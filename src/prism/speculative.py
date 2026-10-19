"""Greedy self-speculation with INT8 drafts and one cached fp32 block verification."""

from dataclasses import dataclass

import numpy as np

from prism.kvcache import KVCache, _forward_cached
from prism.quant import QuantizedSUT
from prism.sut import SUT


@dataclass(frozen=True)
class SpeculativeStep:
    """One round; targets contains only examined positions, through rejection/EOS.

    start is an absolute output index. A bonus is emitted only after all drafts
    were accepted, with budget remaining and no EOS. Window steps have no drafts.
    Candidate tokens after rejection/EOS are computed work, never accepted work.
    """

    start: int
    candidates: tuple[int, ...]
    targets: tuple[int, ...]
    accepted: int
    emitted: tuple[int, ...]
    bonus: bool
    mode: str


@dataclass(frozen=True)
class SpeculativeResult:
    """Tokens and reproducible per-round diagnostics; no cache survives a call."""

    token_ids: list[int]
    steps: tuple[SpeculativeStep, ...]

    def statistics(self, k: int) -> dict[str, object]:
        """Separate proposed, examined and accepted counts, excluding window fallback."""
        rounds = [step for step in self.steps if step.mode == "speculative"]
        proposed = [sum(len(step.candidates) > i for step in rounds) for i in range(k)]
        examined = [sum(len(step.targets) > i for step in rounds) for i in range(k)]
        accepted = [sum(step.accepted > i for step in rounds) for i in range(k)]
        total_proposed, total_examined, total_accepted = (
            sum(values) for values in (proposed, examined, accepted)
        )
        advance = sum(len(step.emitted) for step in rounds)
        return {
            "speculative_rounds": len(rounds),
            "fallback_tokens": sum(len(s.emitted) for s in self.steps if s.mode == "target_only"),
            "proposed": total_proposed,
            "examined": total_examined,
            "accepted": total_accepted,
            "acceptance_rate": total_accepted / total_proposed if total_proposed else None,
            "conditional_acceptance_rate": total_accepted / total_examined
            if total_examined
            else None,
            "accepted_per_round_histogram": {
                str(i): sum(step.accepted == i for step in rounds) for i in range(k + 1)
            },
            "mean_speculative_advance": advance / len(rounds) if rounds else None,
            "mean_all_step_advance": sum(len(s.emitted) for s in self.steps) / len(self.steps)
            if self.steps
            else None,
            "positions": [
                {
                    "position": i + 1,
                    "proposed": proposed[i],
                    "examined": examined[i],
                    "accepted": accepted[i],
                    "conditional_rate": accepted[i] / examined[i] if examined[i] else None,
                    "accepted_per_proposed": accepted[i] / proposed[i] if proposed[i] else None,
                }
                for i in range(k)
            ],
        }


def _truncate(cache: KVCache, length: int) -> KVCache:
    """Own the retained prefix so rejected tails and their base allocations are released."""
    return tuple((key[:, :, :length].copy(), value[:, :, :length].copy()) for key, value in cache)


def generate_speculative(
    target: SUT,
    draft: QuantizedSUT,
    prompt: list[int],
    *,
    max_new_tokens: int,
    k: int,
    eos_id: int | None = None,
) -> SpeculativeResult:
    """Append greedy tokens; exact agreement is gated on the prescribed test matrix.

    At round entry both caches cover all output except its last (pending) token.
    Draft k candidates serially, then verify [pending, *candidates] in one fp32
    forward. Logit i predicts candidate i; logit k predicts the optional bonus.
    On rejection retain only the accepted prefix, leaving the replacement pending.
    After a bonus, forward the last draft token once to align the draft cache.
    This synchronization is real overhead, included in end-to-end measurements.

    Shorten k at budget/context boundaries. Once len(output) reaches context_length,
    use a freshly renumbered fp32 window for every remaining token. No speculative
    speedup is claimed for this interval. EOS in the prompt does not stop generation.
    """
    if isinstance(target, QuantizedSUT) or not isinstance(draft, QuantizedSUT):
        raise ValueError("requires fp32 target and INT8 QuantizedSUT draft")
    if draft.config != target.config or any(weight.bits != 8 for weight in draft.weights.values()):
        raise ValueError("draft must be INT8 with the same decoder configuration")
    if not prompt or any(
        type(t) is not int or not 0 <= t < target.config.vocab_size for t in prompt
    ):
        raise ValueError("prompt must contain valid integer IDs")
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if eos_id is not None and (
        type(eos_id) is not int or not 0 <= eos_id < target.config.vocab_size
    ):
        raise ValueError("invalid EOS ID")
    tokens = list(prompt)
    steps: list[SpeculativeStep] = []
    target_cache = draft_cache = None
    limit = len(prompt) + max_new_tokens
    context = target.config.context_length
    if max_new_tokens and 1 < len(tokens) < context:
        prefix = np.array([tokens[:-1]], dtype=np.int64)
        _, target_cache = _forward_cached(target, prefix)
        _, draft_cache = _forward_cached(draft, prefix)
    while len(tokens) < limit:
        start = len(tokens)
        if start >= context:
            target_cache = draft_cache = None
            logits, _ = _forward_cached(target, np.array([tokens[-context:]], dtype=np.int64))
            following = int(logits[0, -1].argmax())
            tokens.append(following)
            steps.append(SpeculativeStep(start, (), (), 0, (following,), False, "target_only"))
            if following == eos_id:
                break
            continue
        count = min(k, limit - start, context - start)
        candidates = []
        pending = tokens[-1]
        for _ in range(count):
            logits, draft_cache = _forward_cached(
                draft, np.array([[pending]], dtype=np.int64), draft_cache
            )
            pending = int(logits[0, -1].argmax())
            candidates.append(pending)
        logits, target_cache = _forward_cached(
            target, np.array([[tokens[-1], *candidates]], dtype=np.int64), target_cache
        )
        targets = []
        emitted = []
        accepted = 0
        bonus = False
        for i, candidate in enumerate(candidates):
            following = int(logits[0, i].argmax())
            targets.append(following)
            emitted.append(following)
            if following == candidate:
                accepted += 1
            if following != candidate or following == eos_id:
                break
        if accepted == count and emitted[-1] != eos_id and start + count < limit:
            following = int(logits[0, count].argmax())
            emitted.append(following)
            bonus = True
        tokens.extend(emitted)
        steps.append(
            SpeculativeStep(
                start,
                tuple(candidates),
                tuple(targets),
                accepted,
                tuple(emitted),
                bonus,
                "speculative",
            )
        )
        if len(tokens) == limit or tokens[-1] == eos_id:
            break
        if len(tokens) >= context:
            target_cache = draft_cache = None
            continue
        if bonus:
            # The draft loop forwarded pending + k-1 drafts; the kth draft is missing.
            _, draft_cache = _forward_cached(
                draft, np.array([[candidates[-1]]], dtype=np.int64), draft_cache
            )
        target_cache = _truncate(target_cache, len(tokens) - 1)
        draft_cache = _truncate(draft_cache, len(tokens) - 1)
    return SpeculativeResult(tokens, tuple(steps))
