"""Weight-only symmetric quantization; float32 dequantization precedes every Linear."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from prism.sut import SUT, FloatArray


@dataclass(frozen=True)
class QuantizedWeight:
    """Owning integer payload and fp32 scales, with implicit zero-point zero.

    INT8 scales are per output row. INT4 groups run along each input row;
    incomplete groups are zero-padded. Flattened two's-complement nibbles are
    packed low nibble first, including one trailing zero nibble if necessary.
    Shapes/group sizes are Python metadata, excluded from array payload counts.
    """

    bits: Literal[8, 4]
    shape: tuple[int, int]
    group_size: int
    payload: NDArray[Any]
    scales: FloatArray

    @property
    def nbytes(self) -> int:
        """Count stored integer and scale bytes; no zero-point array is stored."""
        return self.payload.nbytes + self.scales.nbytes

    def dequantize(self) -> FloatArray:
        """Allocate float32 weights without retaining a dequantized cache."""
        if self.bits == 8:
            return self.payload.astype(np.float32) * self.scales
        rows, columns = self.shape
        count = rows * self.scales.shape[1] * self.group_size
        nibbles = np.empty(self.payload.size * 2, dtype=np.uint8)
        nibbles[0::2] = self.payload & 15
        nibbles[1::2] = self.payload >> 4
        signed = nibbles[:count].astype(np.int8)
        signed[signed >= 8] -= 16
        grouped = signed.reshape(rows, -1, self.group_size).astype(np.float32)
        return (grouped * self.scales[:, :, None]).reshape(rows, -1)[:, :columns].copy()


def quantize_weight(
    weight: FloatArray, *, bits: Literal[8, 4], group_size: int = 32
) -> QuantizedWeight:
    """Quantize finite 2D fp32 weights with ties-to-even and symmetric ranges.

    Ranges are [-127,127] and [-7,7]. All-zero rows/groups use scale 1;
    underflowing scales are clamped to the smallest positive float32 value.
    """
    if (
        not isinstance(weight, np.ndarray)
        or weight.dtype != np.float32
        or weight.ndim != 2
        or min(weight.shape) < 1
        or not np.isfinite(weight).all()
    ):
        raise ValueError("weight must be a finite nonempty 2D float32 array")
    if type(bits) is not int or bits not in (8, 4):
        raise ValueError("bits must be 8 or 4")
    if type(group_size) is not int or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    rows, columns = weight.shape
    if bits == 8:
        grouped = weight
        maximum = np.max(np.abs(grouped), axis=1, keepdims=True)
    else:
        groups = (columns + group_size - 1) // group_size
        grouped = np.pad(weight, ((0, 0), (0, groups * group_size - columns))).reshape(
            rows, groups, group_size
        )
        maximum = np.max(np.abs(grouped), axis=2)
    limit = 127 if bits == 8 else 7
    scales = np.maximum(
        maximum.astype(np.float64) / limit, float(np.nextafter(np.float32(0), np.float32(1)))
    ).astype(np.float32)
    # Rounding max_float / limit upward can overflow during fp32 reconstruction.
    too_large = scales.astype(np.float64) * limit > float(np.finfo(np.float32).max)
    scales[too_large] = np.nextafter(scales[too_large], np.float32(0))
    scales[maximum == 0] = 1
    divisor = scales if bits == 8 else scales[:, :, None]
    integers = np.clip(np.rint(grouped / divisor), -limit, limit).astype(np.int8)
    if bits == 8:
        payload = integers.copy()
    else:
        nibbles = (integers.ravel() & 15).astype(np.uint8)
        if nibbles.size % 2:
            nibbles = np.pad(nibbles, (0, 1))
        payload = nibbles[0::2] | (nibbles[1::2] << 4)
    payload.flags.writeable = False
    scales.flags.writeable = False
    return QuantizedWeight(bits, (rows, columns), group_size, payload, scales)


class QuantizedSUT(SUT):
    """Reuse the unchanged decoder/cache, replacing only block Linear weights.

    Embeddings (including the tied output head), biases and normalization remain
    fp32. The source SUT is never mutated or retained by this wrapper.
    """

    def __init__(
        self,
        sut: SUT,
        *,
        bits: Literal[8, 4],
        group_size: int = 32,
        quantize_embedding: bool = False,
    ) -> None:
        if type(quantize_embedding) is not bool or quantize_embedding:
            raise ValueError("quantize_embedding must be false; tied embedding/head stays fp32")
        names = {
            f"blocks.{layer}.{part}.weight"
            for layer in range(sut.config.layers)
            for part in ("attention.qkv", "attention.output", "up", "down")
        }
        self.weights = {
            name: quantize_weight(sut.arrays[name], bits=bits, group_size=group_size)
            for name in sorted(names)
        }
        super().__init__(
            sut.config, {k: v for k, v in sut.arrays.items() if k not in names}, sut.tokenizer
        )

    def _linear(self, x: FloatArray, name: str) -> FloatArray:
        return x @ self.weights[f"{name}.weight"].dequantize().T + self.arrays[f"{name}.bias"]

    def memory_ledger(self) -> dict[str, int | float]:
        """Count persistent model array payloads, counting the tied head once.

        Excludes Python metadata, source comparison model, transient dequantized
        weights, activations and KV cache. measure() peaks are separate evidence.
        """
        retained = sum(array.nbytes for array in self.arrays.values())
        integer = sum(weight.payload.nbytes for weight in self.weights.values())
        scales = sum(weight.scales.nbytes for weight in self.weights.values())
        linear_fp32 = sum(int(np.prod(weight.shape)) * 4 for weight in self.weights.values())
        baseline = retained + linear_fp32
        total = retained + integer + scales
        return {
            "fp32_baseline_bytes": baseline,
            "linear_fp32_bytes": linear_fp32,
            "retained_fp32_bytes": retained,
            "integer_bytes_including_padding": integer,
            "scale_bytes": scales,
            "zero_point_bytes": 0,
            "total_bytes": total,
            "compression_ratio": baseline / total,
            "linear_compression_ratio": linear_fp32 / (integer + scales),
        }
