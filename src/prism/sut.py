"""Integrity-checked kaleido P3 decoder inference using only NumPy."""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

DEFAULT_SUT_PATH = Path(__file__).resolve().parents[2] / "data" / "sut" / "kaleido-p3"
_PATTERN = r"\s+|\w+|[^\w\s]+"
type FloatArray = NDArray[np.float32]


def array_digest(*arrays: NDArray[Any]) -> str:
    """Reproduce kaleido.experiments.array_digest, including its JSON separators."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("object arrays do not have a reproducible byte representation")
        digest.update(json.dumps([array.dtype.str, array.shape]).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DecoderConfig:
    """Architecture dimensions read from the exported checkpoint metadata."""

    vocab_size: int
    context_length: int
    width: int
    layers: int
    heads: int
    dropout: float

    def __post_init__(self) -> None:
        for name in ("vocab_size", "context_length", "width", "layers", "heads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("width must divide into heads; dropout must lie in [0,1)")


class ByteBPE:
    """Apply the exported byte BPE merges within kaleido's regex partitions."""

    def __init__(self, value: dict[str, Any]) -> None:
        if (
            set(value) != {"version", "pattern", "eos_id", "vocab_size", "merges"}
            or value["version"] != 1
            or value["pattern"] != _PATTERN
            or value["eos_id"] != 256
        ):
            raise ValueError("unsupported tokenizer format")
        self.eos_id: int = value["eos_id"]
        self.pieces = [bytes([i]) for i in range(256)] + [b""]
        self.ranks: dict[tuple[int, int], int] = {}
        for item in value["merges"]:
            pair = tuple(item)
            new_id = len(self.pieces)
            if (
                len(pair) != 2
                or any(type(i) is not int or not 0 <= i < new_id or i == self.eos_id for i in pair)
                or pair in self.ranks
            ):
                raise ValueError("invalid or duplicate BPE merge")
            self.ranks[pair] = new_id
            self.pieces.append(self.pieces[pair[0]] + self.pieces[pair[1]])
        if value["vocab_size"] != len(self.pieces):
            raise ValueError("vocabulary length mismatch")

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        """Encode Unicode text, optionally appending EOS."""
        if not isinstance(text, str):
            raise ValueError("encode requires a Unicode string")
        result = []
        for chunk in re.findall(_PATTERN, text):
            ids = list(chunk.encode("utf-8"))
            while len(ids) > 1:
                pairs = list(zip(ids[:-1], ids[1:], strict=True))
                pair = min(pairs, key=lambda p: self.ranks.get(p, math.inf))
                if pair not in self.ranks:
                    break
                merged = []
                i = 0
                while i < len(ids):
                    if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                        merged.append(self.ranks[pair])
                        i += 2
                    else:
                        merged.append(ids[i])
                        i += 1
                ids = merged
            result.extend(ids)
        return result + [self.eos_id] if add_eos else result

    def decode(self, ids: list[int], *, errors: str = "strict") -> str:
        """Decode byte pieces, treating EOS as an empty piece."""
        if any(
            isinstance(t, bool) or not isinstance(t, Integral) or not 0 <= t < len(self.pieces)
            for t in ids
        ):
            raise ValueError("token ID out of range")
        return b"".join(self.pieces[t] for t in ids).decode("utf-8", errors=errors)


@dataclass
class SUT:
    """A validated float32 decoder with tied token embedding/output weights."""

    config: DecoderConfig
    arrays: dict[str, FloatArray]
    tokenizer: ByteBPE

    def _linear(self, x: FloatArray, name: str) -> FloatArray:
        return x @ self.arrays[f"{name}.weight"].T + self.arrays[f"{name}.bias"]

    def _norm(self, x: FloatArray, name: str) -> FloatArray:
        centered = x - x.mean(axis=-1, keepdims=True)
        variance = (centered * centered).mean(axis=-1, keepdims=True)
        normalized = centered / np.sqrt(variance + np.float32(1e-5))
        return normalized * self.arrays[f"{name}.weight"] + self.arrays[f"{name}.bias"]

    def forward(self, tokens: NDArray[np.int64]) -> FloatArray:
        """Return (batch, time, vocab) logits without KV caching or dropout."""
        if not isinstance(tokens, np.ndarray) or tokens.ndim != 2 or tokens.dtype != np.int64:
            raise ValueError("tokens must be a non-empty int64 (batch,time) array")
        if min(tokens.shape) < 1:
            raise ValueError("tokens must be a non-empty int64 (batch,time) array")
        if np.any(tokens < 0) or np.any(tokens >= self.config.vocab_size):
            raise ValueError("token ID outside vocabulary")
        batch, time = tokens.shape
        if time > self.config.context_length:
            raise ValueError("context limit exceeded; use a fresh cropped window")
        x = (
            self.arrays["token_embedding.weight"][tokens]
            + self.arrays["position_embedding.weight"][:time]
        )
        visible = np.arange(time)[None, :] <= np.arange(time)[:, None]
        head_width = self.config.width // self.config.heads
        for layer in range(self.config.layers):
            prefix = f"blocks.{layer}"
            qkv = self._linear(self._norm(x, f"{prefix}.ln1"), f"{prefix}.attention.qkv")
            q, k, v = (
                part.reshape(batch, time, self.config.heads, head_width).transpose(0, 2, 1, 3)
                for part in np.split(qkv, 3, axis=-1)
            )
            scores = q @ k.swapaxes(-2, -1) / np.float32(math.sqrt(head_width))
            scores = np.where(visible, scores, -np.inf)
            probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probability /= probability.sum(axis=-1, keepdims=True)
            attended = (probability @ v).transpose(0, 2, 1, 3).reshape(batch, time, -1)
            x = x + self._linear(attended, f"{prefix}.attention.output")
            hidden = self._linear(self._norm(x, f"{prefix}.ln2"), f"{prefix}.up")
            hidden = (
                0.5
                * hidden
                * (
                    1
                    + np.tanh(np.float32(math.sqrt(2 / math.pi)) * (hidden + 0.044715 * hidden**3))
                )
            )
            x = x + self._linear(hidden, f"{prefix}.down")
        return self._norm(x, "final_norm") @ self.arrays["token_embedding.weight"].T

    def generate(
        self,
        prompt: list[int],
        *,
        max_new_tokens: int,
        eos_id: int | None = None,
        use_cache: bool = False,
    ) -> list[int]:
        """Greedily append tokens with optional caching and renumbered cropped windows."""
        if not prompt or any(
            type(t) is not int or not 0 <= t < self.config.vocab_size for t in prompt
        ):
            raise ValueError("prompt must contain valid integer IDs")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if eos_id is not None and (
            type(eos_id) is not int or not 0 <= eos_id < self.config.vocab_size
        ):
            raise ValueError("invalid EOS ID")
        if use_cache:
            from prism.kvcache import _generate_cached

            return _generate_cached(
                self, prompt, max_new_tokens=max_new_tokens, eos_id=eos_id
            ).token_ids
        tokens = list(prompt)
        for _ in range(max_new_tokens):
            window = np.array([tokens[-self.config.context_length :]], dtype=np.int64)
            following = int(self.forward(window)[0, -1].argmax())
            tokens.append(following)
            if following == eos_id:
                break
        return tokens


def _verify_file(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"file sha256 mismatch: {path.name}")


def load_sut(path: Path) -> SUT:
    """Load local artifacts, failing closed on array or file integrity mismatches."""
    manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported SUT manifest schema")
    with np.load(path / "model-42.npz", allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != set(
            manifest["arrays"]
        ):
            raise ValueError("checkpoint array names mismatch")
        arrays = {}
        for name in archive.files:
            value = archive[name]
            if array_digest(value) != manifest["arrays"][name]:
                raise ValueError(f"checkpoint array digest mismatch: {name}")
            arrays[name] = value
    for name in ("model-42.npz", "tokenizer.json"):
        _verify_file(path / name, manifest["source_files"][name]["source_sha256"])
    config = DecoderConfig(**manifest["config"])
    width = config.width
    shapes = {
        "token_embedding.weight": (config.vocab_size, width),
        "position_embedding.weight": (config.context_length, width),
        "final_norm.weight": (width,),
        "final_norm.bias": (width,),
    }
    for layer in range(config.layers):
        for name, out_width, in_width in (
            ("ln1", width, None),
            ("ln2", width, None),
            ("attention.qkv", 3 * width, width),
            ("attention.output", width, width),
            ("up", 4 * width, width),
            ("down", width, 4 * width),
        ):
            prefix = f"blocks.{layer}.{name}"
            shapes[f"{prefix}.weight"] = (out_width, in_width) if in_width else (out_width,)
            shapes[f"{prefix}.bias"] = (out_width,)
    if set(arrays) != set(shapes):
        raise ValueError("checkpoint arrays do not match configured architecture")
    for name, shape in shapes.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"invalid checkpoint shape, dtype or values: {name}")
        value.flags.writeable = False
    tokenizer = ByteBPE(json.loads((path / "tokenizer.json").read_text(encoding="utf-8")))
    if len(tokenizer.pieces) != config.vocab_size:
        raise ValueError("tokenizer/model vocabulary mismatch")
    return SUT(config, arrays, tokenizer)


def check_sut(path: Path = DEFAULT_SUT_PATH) -> dict[str, Any]:
    """Check independent greedy goldens, next-token argmax and tokenizer prompts."""
    sut = load_sut(path)
    manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))
    for name in ("reference_greedy.json", "reference_logits.npz"):
        _verify_file(path / name, manifest["reference_sha256"][name])
    fixture = json.loads((path / "reference_greedy.json").read_text(encoding="utf-8"))
    if set(fixture["cases"]) != {"within_window", "window_rollover"}:
        raise ValueError("expected both greedy reference cases")
    cases = {}
    for name, case in fixture["cases"].items():
        actual = sut.generate(
            case["prompt_ids"], max_new_tokens=case["max_new_tokens"], eos_id=case["eos_id"]
        )
        if actual != case["token_ids"]:
            raise ValueError(f"greedy token sequence mismatch: {name}")
        cases[name] = {"passed": True, "total_tokens": len(actual), "token_ids": actual}
    within = fixture["cases"]["within_window"]
    tokens = np.array(within["token_ids"], dtype=np.int64)
    with np.load(path / "reference_logits.npz", allow_pickle=False) as reference:
        if not np.array_equal(reference["token_ids"], tokens):
            raise ValueError("logits reference token sequence mismatch")
        logits = sut.forward(tokens[None, :])[0]
        expected = reference["logits"]
        if expected.shape != logits.shape or not np.isfinite(expected).all():
            raise ValueError("invalid reference logits")
        error = float(np.max(np.abs(logits - expected)))
    start = len(within["prompt_ids"]) - 1
    predictions = logits[start:-1].argmax(axis=-1)
    if not np.array_equal(predictions, tokens[start + 1 :]):
        raise ValueError("generated-position logits argmax mismatch")
    prompts = manifest["tokenizer_checks"]
    if len(prompts) != 3:
        raise ValueError("expected three sampled prompt tokenizer checks")
    for sample in prompts:
        if [sut.tokenizer.eos_id] + sut.tokenizer.encode(sample["prompt"]) != sample["prompt_ids"]:
            raise ValueError("sampled prompt tokenizer mismatch")
    return {
        "passed": True,
        "greedy_cases": cases,
        "greedy_cases_passed": len(cases),
        "logits_max_abs_error": error,
        "logits_argmax_positions_passed": len(predictions),
        "tokenizer_prompts_passed": len(prompts),
    }
