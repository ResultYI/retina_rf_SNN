from dataclasses import asdict, dataclass
import torch

@dataclass(frozen=True, slots=True)
class TraceComparison:
    equal: bool
    max_abs: float
    mean_on_change: float
    raw_mean_sign_flip: bool
    reported_mean_sign_flip: bool
    resolved_mean_sign_flip: bool
    peak_index_changed: bool

def trace_comparison(reference: torch.Tensor, changed: torch.Tensor, active: torch.Tensor, bound: float) -> TraceComparison:
    left, right = float(reference[active].mean()), float(changed[active].mean())
    report_left = int(left > 1e-9) - int(left < -1e-9)
    report_right = int(right > 1e-9) - int(right < -1e-9)
    return TraceComparison(
        torch.equal(reference, changed),
        float((reference - changed).abs().max()),
        abs(left - right),
        left * right < 0,
        report_left != report_right,
        left * right < 0 and min(abs(left), abs(right)) > bound,
        int(reference[active.nonzero()[0].item():].abs().argmax()) != int(changed[active.nonzero()[0].item():].abs().argmax()),
    )

def pair_rows(context, reference, changed):
    rows = []
    for pair in context["pairs"]:
        a, b = pair["a"], pair["b"]
        for channel in ("logit", "probability"):
            normal = reference[channel][a] - reference[channel][b]
            current = changed[channel][a] - changed[channel][b]
            values = asdict(trace_comparison(normal, current, context["active"], context["bound"]))
            rows.append({**context["identity"], "pair": pair["name"], "family": pair["family"],
                         "control": pair["control"], "channel": channel, **values})
    return rows
