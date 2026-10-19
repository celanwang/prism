"""Independent NumPy operator ablations; the production SUT stays unchanged."""

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from prism.kvcache import KVCache
from prism.sut import SUT, FloatArray

type Variant = Literal["baseline", "qkv_split", "attention_bt", "mlp_reuse"]
VARIANTS = ("baseline", "qkv_split", "attention_bt", "mlp_reuse")


class AblatingSUT(SUT):
    """Select exactly one ablation, sharing all other forward operations.

    Scratch is private to an instance, holds only the most recent (batch,time)
    shape, and is reused across layers/calls. Instances are not reentrant.
    Returned logits and KV caches never alias scratch.
    """

    def __init__(self, sut: SUT, variant: Variant = "baseline") -> None:
        if variant not in VARIANTS:
            raise ValueError("select exactly one supported variant")
        super().__init__(sut.config, sut.arrays, sut.tokenizer)
        self.variant = variant
        self._scratch: tuple[FloatArray, FloatArray, FloatArray] | None = None

    def _qkv(self, x: FloatArray, prefix: str) -> tuple[FloatArray, ...]:
        name = f"{prefix}.attention.qkv"
        if self.variant == "qkv_split":
            # Reverse ablation: three GEMMs, same weight/bias slices, no repacking.
            weights = np.split(self.arrays[f"{name}.weight"], 3)
            biases = np.split(self.arrays[f"{name}.bias"], 3)
            return tuple(x @ w.T + b for w, b in zip(weights, biases, strict=True))
        return tuple(np.split(self._linear(x, name), 3, axis=-1))

    def _mlp(self, x: FloatArray, prefix: str) -> FloatArray:
        if self.variant != "mlp_reuse":
            hidden = self._linear(x, f"{prefix}.up")
            hidden = (
                0.5
                * hidden
                * (
                    1
                    + np.tanh(np.float32(math.sqrt(2 / math.pi)) * (hidden + 0.044715 * hidden**3))
                )
            )
            return self._linear(hidden, f"{prefix}.down")
        # Only this arm reuses up/GELU/down destinations, preserving GELU order.
        shape = (*x.shape[:-1], 4 * self.config.width)
        if self._scratch is None or self._scratch[0].shape != shape:
            self._scratch = (
                np.empty(shape, dtype=np.float32),
                np.empty(shape, dtype=np.float32),
                np.empty(x.shape, dtype=np.float32),
            )
        hidden, activation, down = self._scratch
        np.matmul(x, self.arrays[f"{prefix}.up.weight"].T, out=hidden)
        np.add(hidden, self.arrays[f"{prefix}.up.bias"], out=hidden)
        np.power(hidden, 3, out=activation)
        np.multiply(activation, 0.044715, out=activation)
        np.add(hidden, activation, out=activation)
        np.multiply(np.float32(math.sqrt(2 / math.pi)), activation, out=activation)
        np.tanh(activation, out=activation)
        np.add(1, activation, out=activation)
        np.multiply(0.5, hidden, out=hidden)
        np.multiply(hidden, activation, out=hidden)
        np.matmul(hidden, self.arrays[f"{prefix}.down.weight"].T, out=down)
        np.add(down, self.arrays[f"{prefix}.down.bias"], out=down)
        return down

    def _forward(
        self, tokens: NDArray[np.int64], past: KVCache | None, *, cached: bool
    ) -> tuple[FloatArray, KVCache]:
        if (
            not isinstance(tokens, np.ndarray)
            or tokens.ndim != 2
            or tokens.dtype != np.int64
            or min(tokens.shape) < 1
        ):
            raise ValueError("tokens must be a non-empty int64 (batch,time) array")
        if np.any(tokens < 0) or np.any(tokens >= self.config.vocab_size):
            raise ValueError("token ID outside vocabulary")
        batch, time = tokens.shape
        heads = self.config.heads
        head_width = self.config.width // heads
        offset = 0
        if past is not None:
            if len(past) != self.config.layers:
                raise ValueError("cache layer count mismatch")
            lengths = []
            for key, value in past:
                if (
                    key.ndim != 4
                    or key.shape != value.shape
                    or key.shape[:2] != (batch, heads)
                    or key.shape[3] != head_width
                    or key.dtype != np.float32
                    or value.dtype != np.float32
                ):
                    raise ValueError("cache shape or dtype mismatch")
                lengths.append(key.shape[2])
            if len(set(lengths)) != 1:
                raise ValueError("cache positions must agree across layers")
            offset = lengths[0]
        if offset + time > self.config.context_length:
            raise ValueError("context limit exceeded; rebuild with a fresh cropped window")
        x = (
            self.arrays["token_embedding.weight"][tokens]
            + self.arrays["position_embedding.weight"][offset : offset + time]
        )
        visible = np.arange(offset + time)[None, :] <= offset + np.arange(time)[:, None]
        presents = []
        for layer in range(self.config.layers):
            prefix = f"blocks.{layer}"
            parts = self._qkv(self._norm(x, f"{prefix}.ln1"), prefix)
            if self.variant == "attention_bt":
                # Pack each Q/K/V in batch×time-major storage, then expose head
                # views to the same matmul. Baseline uses strided heads-first
                # views of merged QKV. Copy costs stay inside measured forward.
                parts = tuple(np.ascontiguousarray(part) for part in parts)
            q, k, v = (
                part.reshape(batch, time, heads, head_width).transpose(0, 2, 1, 3) for part in parts
            )
            if cached:
                if past is None:
                    k, v = k.copy(), v.copy()
                else:
                    k = np.concatenate((past[layer][0], k), axis=2)
                    v = np.concatenate((past[layer][1], v), axis=2)
                presents.append((k, v))
            scores = q @ k.swapaxes(-2, -1) / np.float32(math.sqrt(head_width))
            scores = np.where(visible, scores, -np.inf)
            probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probability /= probability.sum(axis=-1, keepdims=True)
            attended = (probability @ v).transpose(0, 2, 1, 3).reshape(batch, time, -1)
            x = x + self._linear(attended, f"{prefix}.attention.output")
            x = x + self._mlp(self._norm(x, f"{prefix}.ln2"), prefix)
        logits = self._norm(x, "final_norm") @ self.arrays["token_embedding.weight"].T
        return logits, tuple(presents)

    def forward(self, tokens: NDArray[np.int64]) -> FloatArray:
        """Use the selected ablation without caching."""
        return self._forward(tokens, None, cached=False)[0]

    def forward_cached(
        self, tokens: NDArray[np.int64], past: KVCache | None = None
    ) -> tuple[FloatArray, KVCache]:
        """Use the same ablation for prefill and incremental cached steps."""
        return self._forward(tokens, past, cached=True)

    def generate(
        self,
        prompt: list[int],
        *,
        max_new_tokens: int,
        eos_id: int | None = None,
        use_cache: bool = False,
    ) -> list[int]:
        """Preserve SUT validation, EOS and renumbered rollover semantics."""
        if not use_cache:
            return super().generate(prompt, max_new_tokens=max_new_tokens, eos_id=eos_id)
        # Reuse production input validation without entering its cached dispatch.
        tokens = super().generate(prompt, max_new_tokens=0, eos_id=eos_id)
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        cache = None
        for _ in range(max_new_tokens):
            if cache is None or cache[0][0].shape[2] >= self.config.context_length:
                cache = None
                incoming = np.array([tokens[-self.config.context_length :]], dtype=np.int64)
            else:
                incoming = np.array([[tokens[-1]]], dtype=np.int64)
            logits, cache = self.forward_cached(incoming, cache)
            following = int(logits[0, -1].argmax())
            tokens.append(following)
            if following == eos_id:
                break
        return tokens
