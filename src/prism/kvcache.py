"""NumPy KV caching with full rebuilds for renumbered sliding windows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from prism.sut import FloatArray

if TYPE_CHECKING:
    from prism.sut import SUT

type KVCache = tuple[tuple[FloatArray, FloatArray], ...]


@dataclass(frozen=True)
class _GenerationResult:
    """Internal diagnostics without changing SUT.generate's token-list contract.

    cache_bytes counts final live K/V array payloads, excluding Python metadata,
    temporary concatenations and forward buffers. No cache survives on the SUT.
    The last emitted token is not forwarded unless another token is requested.
    """

    token_ids: list[int]
    cache_rebuilds: int
    cache_bytes: int


def _forward_cached(
    sut: SUT, tokens: NDArray[np.int64], past: KVCache | None = None
) -> tuple[FloatArray, KVCache]:
    """Forward only new tokens at cached positions; reject overflowing caches."""
    if (
        not isinstance(tokens, np.ndarray)
        or tokens.ndim != 2
        or tokens.dtype != np.int64
        or min(tokens.shape) < 1
    ):
        raise ValueError("tokens must be a non-empty int64 (batch,time) array")
    if np.any(tokens < 0) or np.any(tokens >= sut.config.vocab_size):
        raise ValueError("token ID outside vocabulary")
    batch, time = tokens.shape
    head_width = sut.config.width // sut.config.heads
    offset = 0
    if past is not None:
        if len(past) != sut.config.layers:
            raise ValueError("cache layer count mismatch")
        lengths = []
        for key, value in past:
            if (
                key.ndim != 4
                or key.shape != value.shape
                or key.shape[:2] != (batch, sut.config.heads)
                or key.shape[3] != head_width
                or key.dtype != np.float32
                or value.dtype != np.float32
            ):
                raise ValueError("cache shape or dtype mismatch")
            lengths.append(key.shape[2])
        if len(set(lengths)) != 1:
            raise ValueError("cache positions must agree across layers")
        offset = lengths[0]
    if offset + time > sut.config.context_length:
        raise ValueError("context limit exceeded; rebuild cache with a fresh cropped window")
    x = (
        sut.arrays["token_embedding.weight"][tokens]
        + sut.arrays["position_embedding.weight"][offset : offset + time]
    )
    visible = np.arange(offset + time)[None, :] <= offset + np.arange(time)[:, None]
    presents = []
    for layer in range(sut.config.layers):
        prefix = f"blocks.{layer}"
        qkv = sut._linear(sut._norm(x, f"{prefix}.ln1"), f"{prefix}.attention.qkv")
        q, k, v = (
            part.reshape(batch, time, sut.config.heads, head_width).transpose(0, 2, 1, 3)
            for part in np.split(qkv, 3, axis=-1)
        )
        if past is None:
            # Own K/V storage rather than keeping the full QKV allocation alive.
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
        x = x + sut._linear(attended, f"{prefix}.attention.output")
        hidden = sut._linear(sut._norm(x, f"{prefix}.ln2"), f"{prefix}.up")
        hidden = (
            0.5
            * hidden
            * (1 + np.tanh(np.float32(math.sqrt(2 / math.pi)) * (hidden + 0.044715 * hidden**3)))
        )
        x = x + sut._linear(hidden, f"{prefix}.down")
    logits = sut._norm(x, "final_norm") @ sut.arrays["token_embedding.weight"].T
    return logits, tuple(presents)


def _generate_cached(
    sut: SUT, prompt: list[int], *, max_new_tokens: int, eos_id: int | None = None
) -> _GenerationResult:
    """Generate validated SUT inputs, counting rebuilds after initial prefill."""
    tokens = list(prompt)
    cache = None
    rebuilds = 0
    for _ in range(max_new_tokens):
        if cache is None or cache[0][0].shape[2] >= sut.config.context_length:
            if cache is not None:
                rebuilds += 1
            window = np.array([tokens[-sut.config.context_length :]], dtype=np.int64)
            # Release the obsolete window before constructing its replacement.
            cache = None
            logits, cache = _forward_cached(sut, window)
        else:
            logits, cache = _forward_cached(sut, np.array([[tokens[-1]]], dtype=np.int64), cache)
        following = int(logits[0, -1].argmax())
        tokens.append(following)
        if following == eos_id:
            break
    cache_bytes = 0 if cache is None else sum(array.nbytes for pair in cache for array in pair)
    return _GenerationResult(tokens, rebuilds, cache_bytes)
