"""Bounded, exclusive-owner KV pages and exact renumbered-window decoding.

The budget covers owning float32 K/V page arrays only, not the process, page-table
metadata, gather copies or attention buffers. Pages allocate lazily; no prefix
sharing or references to physical storage escape. LRU is memory simulation only.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from prism.sut import SUT, DecoderConfig, FloatArray

type Policy = Literal["sliding", "lru"]


class InfeasibleBudget(ValueError):
    """The K/V pool cannot retain the complete correct attention window."""


@dataclass(frozen=True)
class PageStats:
    """Payload statistics; requests count block lookups on reads and writes."""

    resident_pages: int
    allocated_bytes: int
    used_bytes: int
    peak_allocated_bytes: int
    peak_used_bytes: int
    allocations: int
    releases: int
    evictions: int
    hits: int
    misses: int

    @property
    def fragmentation(self) -> float:
        """Return allocated-but-unused payload divided by allocated payload."""
        return 0.0 if not self.allocated_bytes else 1 - self.used_bytes / self.allocated_bytes

    @property
    def hit_rate(self) -> float | None:
        """Return block lookup hit rate, or None if no lookups occurred."""
        requests = self.hits + self.misses
        return self.hits / requests if requests else None


class PagedKVCache:
    """Map logical blocks to exclusive physical pages within a K/V byte budget.

    Allocation returns no storage reference. gather() always copies, so releasing
    or recycling a physical slot cannot change a previously gathered value.
    Sliding chooses the oldest logical block; LRU chooses the least recent lookup.
    Neither eviction rule makes surviving model K/V valid after a window shift:
    generation invalidates every page and rebuilds with positions starting at zero.
    """

    def __init__(
        self,
        config: DecoderConfig,
        *,
        block_size: int,
        budget_bytes: int,
        batch_size: int = 1,
        policy: Policy = "sliding",
    ) -> None:
        for name, value, minimum in (
            ("block_size", block_size, 1),
            ("budget_bytes", budget_bytes, 0),
            ("batch_size", batch_size, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if policy not in ("sliding", "lru"):
            raise ValueError("policy must be sliding or lru")
        self.config = config
        self.block_size = block_size
        self.budget_bytes = budget_bytes
        self.batch_size = batch_size
        self.policy = policy
        self.bytes_per_token = 2 * config.layers * batch_size * config.width * 4
        self.page_bytes = self.block_size * self.bytes_per_token
        self.capacity = budget_bytes // self.page_bytes
        self.length = 0
        self._table: dict[int, int] = {}
        self._pages: dict[int, FloatArray] = {}
        self._used: dict[int, int] = {}
        self._valid: dict[int, list[int]] = {}
        self._touched: dict[int, int] = {}
        self._clock = 0
        self._allocations = self._releases = self._evictions = 0
        self._hits = self._misses = 0
        self._peak_allocated = self._peak_used = 0
        # Slots are recycled after release, without a budget-sized Python list.
        self._free: list[int] = []
        self._next_slot = 0

    @property
    def page_table(self) -> dict[int, int]:
        """Return a detached logical-to-physical mapping, never storage aliases."""
        return dict(self._table)

    @property
    def stats(self) -> PageStats:
        """Read actual owning ndarray nbytes, not a reserved-budget estimate."""
        allocated = sum(page.nbytes for page in self._pages.values())
        used = sum(self._used.values()) * self.bytes_per_token
        return PageStats(
            len(self._pages),
            allocated,
            used,
            self._peak_allocated,
            self._peak_used,
            self._allocations,
            self._releases,
            self._evictions,
            self._hits,
            self._misses,
        )

    def _track_peak(self) -> None:
        self._peak_allocated = max(self._peak_allocated, self.stats.allocated_bytes)
        self._peak_used = max(self._peak_used, self.stats.used_bytes)

    def _lookup(self, logical: int) -> int | None:
        self._clock += 1
        if logical not in self._table:
            self._misses += 1
            return None
        self._hits += 1
        self._touched[logical] = self._clock
        return self._table[logical]

    def release(self, logical: int, *, eviction: bool = False) -> None:
        """Release one block; a second release raises KeyError (no shared owners)."""
        physical = self._table.pop(logical)
        del self._pages[physical]
        del self._used[logical]
        del self._valid[logical]
        del self._touched[logical]
        self._free.append(physical)
        self._releases += 1
        self._evictions += int(eviction)
        # Direct releases invalidate the contiguous model-cache length.
        self.length = 0

    def clear(self, *, eviction: bool = False) -> None:
        """Invalidate the entire window, releasing oldest logical blocks first."""
        for logical in sorted(self._table):
            self.release(logical, eviction=eviction)
        self.length = 0

    def _allocate(self, logical: int, *, evict: bool) -> int:
        if self.capacity == 0:
            raise InfeasibleBudget("K/V budget cannot hold one page")
        if len(self._table) == self.capacity:
            if not evict:
                raise InfeasibleBudget("K/V budget cannot hold the complete correct window")
            oldest = (
                min(self._table)
                if self.policy == "sliding"
                else min(self._touched, key=self._touched.__getitem__)
            )
            self.release(oldest, eviction=True)
        physical = self._free.pop() if self._free else self._next_slot
        if physical == self._next_slot:
            self._next_slot += 1
        c = self.config
        self._pages[physical] = np.empty(
            (2, c.layers, self.batch_size, c.heads, self.block_size, c.width // c.heads),
            dtype=np.float32,
        )
        self._table[logical] = physical
        self._used[logical] = 0
        self._valid[logical] = [0] * c.layers
        self._touched[logical] = self._clock
        self._allocations += 1
        self._track_peak()
        return physical

    def reserve(self, length: int) -> None:
        """Reserve a complete generation window atomically with respect to budget.

        Only forward_paged writes these pages. Simulation-loaded blocks cannot be
        mixed with generation: a model forward must start with an empty cache.
        """
        if type(length) is not int or not self.length < length <= self.config.context_length:
            raise ValueError("length must grow within the context limit")
        required = math.ceil(length / self.block_size)
        if required > self.capacity:
            raise InfeasibleBudget("K/V budget cannot hold the complete correct window")
        if set(self._table) != set(range(math.ceil(self.length / self.block_size))):
            raise ValueError("generation requires an intact contiguous cache")
        for logical in range(required):
            physical = self._lookup(logical)
            if physical is None:
                self._allocate(logical, evict=False)
            self._used[logical] = min(self.block_size, length - logical * self.block_size)
        self._track_peak()

    def write(self, layer: int, start: int, key: FloatArray, value: FloatArray) -> None:
        """Copy a layer's new positions into already reserved pages."""
        time = key.shape[2] if key.ndim == 4 else 0
        expected = (
            self.batch_size,
            self.config.heads,
            time,
            self.config.width // self.config.heads,
        )
        if (
            type(layer) is not int
            or not 0 <= layer < self.config.layers
            or key.shape != expected
            or value.shape != expected
            or time < 1
            or key.dtype != np.float32
            or value.dtype != np.float32
            or type(start) is not int
            or start != self.length
        ):
            raise ValueError("invalid layer, position or K/V shape/dtype")
        end = start + time
        for position in range(start, end):
            logical, offset = divmod(position, self.block_size)
            if (
                logical not in self._table
                or offset >= self._used[logical]
                or self._valid[logical][layer]
                != max(0, min(start - logical * self.block_size, self.block_size))
            ):
                raise ValueError("write exceeds reserved pages")
        for position in range(start, end):
            logical, offset = divmod(position, self.block_size)
            page = self._pages[self._table[logical]]
            page[0, layer, :, :, offset] = key[:, :, position - start]
            page[1, layer, :, :, offset] = value[:, :, position - start]
            self._valid[logical][layer] = offset + 1

    def gather(self, layer: int, length: int) -> tuple[FloatArray, FloatArray]:
        """Copy valid positions in logical order, excluding padding and stale slots."""
        if type(layer) is not int or not 0 <= layer < self.config.layers:
            raise ValueError("invalid layer")
        if type(length) is not int or length < 1 or length > self.config.context_length:
            raise ValueError("invalid gather length")
        parts = []
        for logical in range(math.ceil(length / self.block_size)):
            physical = self._lookup(logical)
            if physical is None:
                raise KeyError(f"missing logical block {logical}; full window rebuild required")
            count = min(self.block_size, length - logical * self.block_size)
            if self._valid[logical][layer] < count:
                raise ValueError("gather exceeds initialized positions")
            parts.append(self._pages[physical][:, layer, :, :, :count])
        payload = np.concatenate(parts, axis=3)
        return payload[0], payload[1]

    def access_block(
        self, logical: int, used_tokens: int, recompute: Callable[[int], FloatArray]
    ) -> bool:
        """Simulate a block request; recompute exact supplied payload on every miss.

        This never forwards model tokens. Arbitrary layer-wise model K/V cannot
        be recomputed independently after eviction under renumbered windows.
        The return value reports a hit; exceptions do not install partial pages.
        """
        if type(logical) is not int or logical < 0:
            raise ValueError("logical block must be a nonnegative integer")
        if type(used_tokens) is not int or not 1 <= used_tokens <= self.block_size:
            raise ValueError("invalid used_tokens")
        physical = self._lookup(logical)
        if physical is not None:
            if self._used[logical] != used_tokens:
                raise ValueError("block length changed; invalidate before recomputation")
            return True
        if not self.capacity:
            raise InfeasibleBudget("K/V budget cannot hold one page")
        payload = recompute(logical)
        c = self.config
        expected = (2, c.layers, self.batch_size, c.heads, used_tokens, c.width // c.heads)
        if payload.shape != expected or payload.dtype != np.float32:
            raise ValueError("invalid recomputed payload")
        physical = self._allocate(logical, evict=True)
        self._pages[physical][:, :, :, :, :used_tokens] = payload
        self._used[logical] = used_tokens
        self._valid[logical] = [used_tokens] * c.layers
        self._track_peak()
        return False


def forward_paged(sut: SUT, tokens: NDArray[np.int64], cache: PagedKVCache) -> FloatArray:
    """Forward new tokens through page-table gathers, preserving T1 arithmetic."""
    if cache.policy != "sliding":
        raise ValueError("LRU is memory simulation only")
    if cache.config != sut.config:
        raise ValueError("cache/model configuration mismatch")
    if (
        not isinstance(tokens, np.ndarray)
        or tokens.ndim != 2
        or tokens.dtype != np.int64
        or min(tokens.shape) < 1
        or tokens.shape[0] != cache.batch_size
    ):
        raise ValueError("tokens must be a non-empty int64 (batch,time) array matching the cache")
    if np.any(tokens < 0) or np.any(tokens >= sut.config.vocab_size):
        raise ValueError("token ID outside vocabulary")
    batch, time = tokens.shape
    offset = cache.length
    end = offset + time
    if end > sut.config.context_length:
        raise ValueError("context limit exceeded; invalidate and rebuild the full window")
    cache.reserve(end)
    try:
        x = (
            sut.arrays["token_embedding.weight"][tokens]
            + sut.arrays["position_embedding.weight"][offset:end]
        )
        visible = np.arange(end)[None, :] <= offset + np.arange(time)[:, None]
        head_width = sut.config.width // sut.config.heads
        for layer in range(sut.config.layers):
            prefix = f"blocks.{layer}"
            qkv = sut._linear(sut._norm(x, f"{prefix}.ln1"), f"{prefix}.attention.qkv")
            q, k, v = (
                part.reshape(batch, time, sut.config.heads, head_width).transpose(0, 2, 1, 3)
                for part in np.split(qkv, 3, axis=-1)
            )
            cache.write(layer, offset, k, v)
            k, v = cache.gather(layer, end)
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
                * (
                    1
                    + np.tanh(np.float32(math.sqrt(2 / math.pi)) * (hidden + 0.044715 * hidden**3))
                )
            )
            x = x + sut._linear(hidden, f"{prefix}.down")
        logits = sut._norm(x, "final_norm") @ sut.arrays["token_embedding.weight"].T
    except Exception:
        cache.clear()
        raise
    cache.length = end
    return logits


@dataclass(frozen=True)
class PagedGenerationResult:
    """Detached generation diagnostics; no physical cache survives the call."""

    token_ids: list[int]
    cache_rebuilds: int
    stats: PageStats


def generate_paged(
    sut: SUT,
    prompt: list[int],
    *,
    max_new_tokens: int,
    block_size: int,
    budget_bytes: int,
    eos_id: int | None = None,
) -> PagedGenerationResult:
    """Generate exactly, rejecting undersized pools without shrinking or fallback.

    Preflight assumes no early EOS: reserve feasibility for every requested step.
    The last emitted token is not forwarded. Zero-token requests allocate no pages.
    """
    if not prompt or any(type(t) is not int or not 0 <= t < sut.config.vocab_size for t in prompt):
        raise ValueError("prompt must contain valid integer IDs")
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if eos_id is not None and (type(eos_id) is not int or not 0 <= eos_id < sut.config.vocab_size):
        raise ValueError("invalid EOS ID")
    cache = PagedKVCache(sut.config, block_size=block_size, budget_bytes=budget_bytes)
    needed = min(sut.config.context_length, len(prompt) + max_new_tokens - 1)
    if max_new_tokens and math.ceil(needed / block_size) > cache.capacity:
        raise InfeasibleBudget("K/V budget cannot hold the complete correct window")
    tokens = list(prompt)
    rebuilds = 0
    for _ in range(max_new_tokens):
        if cache.length == 0 or cache.length >= sut.config.context_length:
            if cache.length:
                rebuilds += 1
                cache.clear(eviction=True)
            window = np.array([tokens[-sut.config.context_length :]], dtype=np.int64)
            logits = forward_paged(sut, window, cache)
        else:
            logits = forward_paged(sut, np.array([[tokens[-1]]], dtype=np.int64), cache)
        following = int(logits[0, -1].argmax())
        tokens.append(following)
        if following == eos_id:
            break
    return PagedGenerationResult(tokens, rebuilds, cache.stats)
