"""解析模型 for dense, tied-embedding GPT training; predictions are not measurements.

Symbols (counts are dimensionless): L layers, h hidden width, f FFN width,
V vocabulary, S sequence length, C position capacity, H attention heads,
P parameters; d/t/p DP/TP/PP degrees, b sequences per microbatch per DP rank,
m microbatches per optimizer step, z ZeRO stage. w/g/o/a/k are bytes per
weight/gradient/Adam state/activation/dropout mask. o includes master weights
and both Adam moments (12 bytes for FP32). q=P/(t*p) is the ideal balanced
parameter count per device; N=d*b*m*S is global tokens per optimizer step.

Resident states, in bytes: weights=q*w/(d if z=3 else 1), gradients=
q*g/(d if z>=2 else 1), optimizer=q*o/(d if z>=1 else 1). In particular,
ZeRO-2 = q*[w+(g+o)/d]. This follows partitioning Adam first, then gradients:
https://arxiv.org/abs/1910.02054 (sections 3 and 7).

Retained activation bytes per layer per microbatch without sequence parallelism:
A=S*b*[h*(4*a+2*k)+(4*a*h+2*a*f+(2*a+k)*H*S)/t]. At f=4h,a=2,k=1,
this is S*b*h*[10+24/t+5*H*S/(h*t)], equation 2 of
https://arxiv.org/abs/2205.05198. Full layer checkpointing retains S*b*h*a
bytes instead. Flush 1F1B retains at most min(m,p) microbatches on stage 0;
activation estimate = (L/p)*min(m,p)*A, a worst-stage retained approximation.

For a ring over n ranks and X bytes of full payload, R(n)=(n-1)/n,
all-reduce sends 2*R(n)*X bytes/rank, all-gather and reduce-scatter each
send R(n)*X. Volumes count SEND only (receive separately would double them).
DP z=0: one gradient all-reduce per step. z=1: one gradient reduce-scatter
and one updated-weight all-gather per step. z=2: m gradient reduce-scatters
and one weight all-gather. z=3: m gradient reduce-scatters and 2*m weight
all-gathers (forward/backward, weights released between uses). Gradient
reduction per microbatch at z>=2 avoids retaining full accumulated gradients.
TP uses 4 all-reduces/layer/microbatch (6 with full recomputation).
PP sends one forward activation and one backward activation-gradient per
boundary/microbatch, each S*b*h*a bytes. Adjacent stages split the sends;
rank mean is 2*(p-1)/p*m*S*b*h*a, with edge/interior volumes also reported.

Bubble idle fraction beta=(p-1)/(m+p-1). With stage forward/backward time
u [seconds], useful=m*u, idle=(p-1)*u, elapsed=(m+p-1)*u. The fraction
relative to useful time in https://arxiv.org/abs/2104.04473 is (p-1)/m;
normalizing by elapsed time gives the requested beta.

Training work F=6*P*N+12*L*N*S*h [FLOP]; full checkpointing multiplies by
4/3 (one extra forward). Hypothetical per-device compute rate c [FLOP/s]
gives T=F/[d*t*p*c*(1-beta)] [s] and throughput=N/T [tokens/s]. This is
compute-only: no communication/computation overlap, kernel efficiency,
network time, latency, load imbalance or measured hardware performance.
Parameter placement assumes perfect balance/sharding including embeddings,
biases and norms; actual Megatron placement can differ. Memory excludes live
ZeRO gathers, workspaces, logits/loss, allocator fragmentation and temporary
recomputed activations. Thus it is retained memory, NOT peak device memory.
ZeRO conserves model states; total memory conservation also needs a fixed
global batch (activations are DP replicated, not sharded by ZeRO).
"""

import math
from dataclasses import asdict, dataclass
from typing import Any

MODE = "解析模型"


def _positive_int(name: str, value: int) -> None:
    """Reject nonintegral, boolean and nonpositive counts."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{MODE}: {name} must be a positive integer")


@dataclass(frozen=True)
class ModelSpec:
    """Dense GPT dimensions; tied token embeddings, learned positions and biases."""

    layers: int
    hidden: int
    ffn: int
    vocab: int
    seq: int
    heads: int
    position_capacity: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _positive_int(name, value)
        if self.hidden % self.heads or self.seq > self.position_capacity:
            raise ValueError(f"{MODE}: heads must divide hidden; seq exceeds position capacity")

    @property
    def parameters(self) -> int:
        """P=V*h+C*h+2*h+L*(4*h²+2*h*f+9*h+f), in scalar parameters.

        Per layer: attention weights 4*h², MLP weights 2*h*f; attention
        biases 4*h, MLP biases f+h, two affine norms 4*h. Final norm 2*h.
        """
        h = self.hidden
        return (
            self.vocab * h
            + self.position_capacity * h
            + 2 * h
            + self.layers * (4 * h * h + 2 * h * self.ffn + 9 * h + self.ffn)
        )


@dataclass(frozen=True)
class Strategy:
    """Parallel degrees and accumulation policy; z=0 means ordinary DP."""

    dp: int = 1
    tp: int = 1
    pp: int = 1
    zero: int = 0
    micro_batch: int = 1
    micro_batches: int = 1
    recompute: bool = False

    def __post_init__(self) -> None:
        for name in ("dp", "tp", "pp", "micro_batch", "micro_batches"):
            _positive_int(name, getattr(self, name))
        if type(self.zero) is not int or self.zero not in range(4):
            raise ValueError(f"{MODE}: zero must be 0, 1, 2 or 3")
        if type(self.recompute) is not bool:
            raise ValueError(f"{MODE}: recompute must be boolean")


@dataclass(frozen=True)
class Precision:
    """Storage widths, all in bytes/scalar; optimizer includes FP32 master copy."""

    weight_bytes: int = 2
    gradient_bytes: int = 2
    optimizer_bytes: int = 12
    activation_bytes: int = 2
    mask_bytes: int = 1

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _positive_int(name, value)


def pipeline_bubble(pp: int, micro_batches: int) -> float:
    """Return dimensionless idle/elapsed fraction for balanced flush pipelines."""
    _positive_int("pp", pp)
    _positive_int("micro_batches", micro_batches)
    return (pp - 1) / (micro_batches + pp - 1)


def predict(
    model: ModelSpec,
    strategy: Strategy,
    precision: Precision | None = None,
    *,
    compute_flops_per_s: float = 1e12,
) -> dict[str, Any]:
    """Return 解析模型 resident bytes, send bytes/step and compute-only throughput.

    compute_flops_per_s is an explicitly hypothetical rate, never a device spec.
    States are ideal per-device means; activation bytes are a worst-stage proxy.
    """
    if (
        type(compute_flops_per_s) not in (int, float)
        or not math.isfinite(compute_flops_per_s)
        or compute_flops_per_s <= 0
    ):
        raise ValueError(f"{MODE}: hypothetical compute rate must be finite and positive")
    precision = Precision() if precision is None else precision
    d, t, p = strategy.dp, strategy.tp, strategy.pp
    b, m, z = strategy.micro_batch, strategy.micro_batches, strategy.zero
    if model.layers % p or any(value % t for value in (model.hidden, model.ffn, model.heads)):
        raise ValueError(f"{MODE}: pp must divide layers; tp must divide hidden, ffn and heads")
    h, s = model.hidden, model.seq
    w, g, o = precision.weight_bytes, precision.gradient_bytes, precision.optimizer_bytes
    a, k = precision.activation_bytes, precision.mask_bytes
    q = model.parameters / (t * p)
    replicated = s * b * h * (4 * a + 2 * k)
    sharded = s * b * (4 * a * h + 2 * a * model.ffn + (2 * a + k) * model.heads * s) / t
    if strategy.recompute:
        replicated, sharded = s * b * h * a, 0.0
    retained_layers = model.layers / p * min(m, p)
    memory = {
        "parameters": q * w / (d if z == 3 else 1),
        "gradients": q * g / (d if z >= 2 else 1),
        "optimizer": q * o / (d if z >= 1 else 1),
        "activations": retained_layers * (replicated + sharded),
    }
    memory["model_states"] = memory["parameters"] + memory["gradients"] + memory["optimizer"]
    memory["total"] = memory["model_states"] + memory["activations"]
    rd, rt = (d - 1) / d, (t - 1) / t
    dp_ar = 2 * rd * q * g if z == 0 else 0.0
    dp_rs = rd * q * g * (m if z >= 2 else 1) if z else 0.0
    dp_ag = rd * q * w * (2 * m if z == 3 else 1) if z else 0.0
    tp_ar = (6 if strategy.recompute else 4) * model.layers / p * m * 2 * rt * s * b * h * a
    p2p = [((rank > 0) + (rank < p - 1)) * m * s * b * h * a for rank in range(p)]
    communication = {
        "all_reduce": dp_ar + tp_ar,
        "all_gather": dp_ag,
        "reduce_scatter": dp_rs,
        "p2p": sum(p2p) / p,
        "dp_all_reduce": dp_ar,
        "tp_all_reduce": tp_ar,
        "p2p_by_stage": p2p,
    }
    communication["total"] = dp_ar + tp_ar + dp_ag + dp_rs + sum(p2p) / p
    tokens = d * b * m * s
    flops = (6 * model.parameters * tokens + 12 * model.layers * tokens * s * h) * (
        4 / 3 if strategy.recompute else 1
    )
    bubble = pipeline_bubble(p, m)
    # Direct ratio avoids cancellation of 1-beta for very small m/p.
    utilization = m / (m + p - 1)
    seconds = flops / (d * t * p * compute_flops_per_s * utilization)
    return {
        "mode": MODE,
        "parameters": model.parameters,
        "devices": d * t * p,
        "memory_bytes": memory,
        "activation_replicated_bytes": retained_layers * replicated,
        "activation_sharded_bytes": retained_layers * sharded,
        "communication_send_bytes_per_step": communication,
        "bubble_fraction": bubble,
        "global_tokens_per_step": tokens,
        "training_flops_per_step": flops,
        "hypothetical_compute_flops_per_s": compute_flops_per_s,
        "compute_only_step_s": seconds,
        "compute_only_tokens_per_s": tokens / seconds,
        "assumptions": __doc__,
    }
