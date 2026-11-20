"""E2: hidden defects against a scripted detector and norm-matched random edits.

Families are rank-one, amplified-direction, suppressed-feature, and single-expert.
The single-expert family is blocked on a dense checkpoint. Recovery is scored on
prompts the editor is not shown. No editor is fit on these families.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from selfedit.config import DEFAULT_MODEL, Gates
from selfedit.editors import run_model
from selfedit.plant import Plant, find_plant
from selfedit.subject import Subject

ROOT = Path(__file__).resolve().parents[1]

HIDDEN = (
    "A hidden change was made to one MLP down-projection you are running on. "
    "Restore the next tokens. The harness multiplies one column by a factor you choose, "
    "between 0.05 and 20. Reply with one JSON object and no other text. "
    "Call column_stats with no layer. It lists unusual columns from every layer. "
    "Then finish the propose call with a column and a factor from that list. "
    "The layer is taken from the matching row. If a proposal is rejected, choose a different column."
)


@dataclass
class Defect:
    family: str
    name: str
    column: int
    scale: float
    dev: list[int]
    held: list[int]
    guard: list[int]
    restore: torch.Tensor

    def as_plant(self) -> Plant:
        return Plant(self.name, self.column, self.scale, 1, self.dev, self.held, self.guard)


def run_e2(
    model_id: str = DEFAULT_MODEL,
    device: str = "auto",
    out_dir: Path | None = None,
    live: bool = True,
    subject=None,
) -> dict:
    torch.set_grad_enabled(False)
    gates = Gates()
    if subject is None:
        from selfedit.hf_subject import HFSubject

        subject = HFSubject(model_id, device=device)
        subject.origin = subject.clone_state()
    else:
        subject.load_state(subject.origin)
    if not subject.editable_names():
        raise SystemExit("this checkpoint has no mlp.down_proj.weight tensors")
    origin = subject.origin
    amplified = _plant_amplified(subject, gates, origin)
    rows = [
        _evaluate(subject, gates, origin, amplified, live),
        _evaluate(subject, gates, origin, _plant_suppressed(subject, gates, origin, amplified), live),
        _evaluate(subject, gates, origin, _plant_rank_one(subject, gates, origin), live),
        _single_expert(subject),
    ]
    report = {
        "stage": "e2",
        "model": model_id,
        "device": str(subject.device),
        "dtype": subject.dtype,
        "k": [1, 3],
        "families": rows,
    }
    destination = out_dir or (ROOT / "runs" / "e2")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {destination / 'report.json'}", flush=True)
    subject.load_state(origin)
    return report


def run_amplified_one_row(
    out_dir: Path | None = None,
    shown_rows: int = 1,
    reverse_rows: bool = False,
    row_order: tuple[int, ...] | None = None,
    say: str | None = None,
) -> dict:
    """Show the live editor the top column_stats rows on the amplified plant."""
    torch.set_grad_enabled(False)
    from selfedit.hf_subject import HFSubject

    gates = Gates()
    subject = HFSubject(DEFAULT_MODEL, device="auto")
    subject.origin = subject.clone_state()
    origin = subject.origin
    defect = _plant_amplified(subject, gates, origin)
    if isinstance(defect, dict):
        raise SystemExit(defect)
    planted = subject.clone_state()
    planted_weight = planted[defect.name].float().cpu()
    subject.load_state(planted)
    outcome = run_model(
        subject,
        gates,
        defect.as_plant(),
        "model",
        blind=False,
        frozen=False,
        system=HIDDEN,
        free_layer=True,
        shown_rows=shown_rows,
        reverse_rows=reverse_rows,
        row_order=row_order,
        say=say,
    )
    scores = _score(subject, defect, planted_weight, outcome.journal, outcome.held_correct, outcome.held_total)
    subject.load_state(origin)
    report = {
        "family": defect.family,
        "parameter": defect.name,
        "column": defect.column,
        "scale": defect.scale,
        "shown_rows": shown_rows,
        "reverse_rows": reverse_rows,
        "row_order": list(row_order) if row_order else None,
        "say": say,
        "model": scores,
        "raw": [entry.get("raw") for entry in outcome.journal if entry.get("raw")],
    }
    folder = "e2-one-row" if shown_rows == 1 else f"e2-{shown_rows}-rows"
    if row_order:
        folder += "-" + "-".join(str(index) for index in row_order)
    elif reverse_rows:
        folder += "-reversed"
    if say:
        folder += "-largest-z"
    destination = out_dir or (ROOT / "runs" / folder)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"planted {defect.name} column {defect.column}  "
        f"proposal {scores['proposals']}  "
        f"localization@1={scores['localization_at_1']} recovery={scores['behavioral_recovery']:.2f}",
        flush=True,
    )
    print(f"wrote {destination / 'report.json'}", flush=True)
    return report


def trace_counts(root: Path | None = None) -> dict:
    """Count tool calls recorded in run reports and journals."""
    base = root or (ROOT / "runs")
    counts: dict[str, int] = {}
    files = 0
    calls = 0
    for path in sorted(base.rglob("report.json")):
        if path.parent.name == "e36":
            continue
        payload = json.loads(path.read_text())
        raw_lines = _raw_lines(payload)
        if not raw_lines:
            continue
        files += 1
        for line in raw_lines:
            tool = _tool_name(line)
            if tool is None:
                continue
            counts[tool] = counts.get(tool, 0) + 1
            calls += 1
    journals = 0
    for path in sorted(base.rglob("journal.jsonl")):
        journals += 1
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            tool = (row.get("action") or {}).get("tool")
            if not tool:
                continue
            counts[tool] = counts.get(tool, 0) + 1
            calls += 1
    return {"trace_files": files, "journals": journals, "calls": calls, "tool_counts": counts}


def run_e36(out_dir: Path | None = None, row_order: tuple[int, ...] | None = None) -> dict:
    """Count recorded tool calls, then show one earlier trace on a fresh amplified plant."""
    torch.set_grad_enabled(False)
    from selfedit.hf_subject import HFSubject

    counts = trace_counts()
    earlier_path = ROOT / "runs" / "e2-5-rows-largest-z" / "report.json"
    earlier = json.loads(earlier_path.read_text())
    raw = [line for line in earlier.get("raw") or [] if isinstance(line, str) and line.strip()]
    prior = [
        {
            "role": "user",
            "content": "Earlier trace from this editor on the same kind of hidden column change.",
        }
    ]
    prior.extend({"role": "assistant", "content": line} for line in raw)
    earlier_proposal = None
    for line in raw:
        action = _tool_name(line)
        if action != "propose":
            continue
        body = json.loads(line)
        earlier_proposal = {
            "column": body.get("column"),
            "factor": body.get("factor"),
            "layer": body.get("layer"),
        }
    print(f"traces: {counts}", flush=True)
    gates = Gates()
    subject = HFSubject(DEFAULT_MODEL, device="auto")
    subject.origin = subject.clone_state()
    origin = subject.origin
    defect = _plant_amplified(subject, gates, origin)
    if isinstance(defect, dict):
        raise SystemExit(defect)
    planted = subject.clone_state()
    planted_weight = planted[defect.name].float().cpu()
    subject.load_state(planted)
    outcome = run_model(
        subject,
        gates,
        defect.as_plant(),
        "model",
        blind=False,
        frozen=False,
        system=HIDDEN,
        free_layer=True,
        shown_rows=5,
        row_order=row_order,
        prior=prior,
    )
    scores = _score(subject, defect, planted_weight, outcome.journal, outcome.held_correct, outcome.held_total)
    subject.load_state(origin)
    proposal = scores["proposals"][0] if scores["proposals"] else None
    report = {
        "family": defect.family,
        "parameter": defect.name,
        "column": defect.column,
        "scale": defect.scale,
        "traces": counts,
        "earlier_trace": str(earlier_path.relative_to(ROOT)),
        "earlier_raw": raw,
        "row_order": list(row_order) if row_order else None,
        "earlier_proposal": earlier_proposal,
        "same_column_as_earlier": bool(
            proposal and earlier_proposal and proposal["column"] == earlier_proposal["column"]
        ),
        "model": scores,
        "raw": [entry.get("raw") for entry in outcome.journal if entry.get("raw")],
    }
    folder = "e36" if not row_order else "e36-" + "-".join(str(index) for index in row_order)
    destination = out_dir or (ROOT / "runs" / folder)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "journal.jsonl").write_text(
        "".join(json.dumps(entry, ensure_ascii=True) + "\n" for entry in outcome.journal)
    )
    print(
        f"planted {defect.name} column {defect.column}  "
        f"proposal {scores['proposals']}  "
        f"same_as_earlier={report['same_column_as_earlier']} "
        f"localization@1={scores['localization_at_1']} recovery={scores['behavioral_recovery']:.2f}",
        flush=True,
    )
    print(f"wrote {destination / 'report.json'}", flush=True)
    return report


def _raw_lines(payload: object) -> list[str]:
    found: list[str] = []
    if isinstance(payload, dict):
        raw = payload.get("raw")
        if isinstance(raw, list):
            found.extend(line for line in raw if isinstance(line, str))
        for value in payload.values():
            if isinstance(value, (dict, list)) and value is not raw:
                found.extend(_raw_lines(value))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_raw_lines(item))
    return found


def _tool_name(line: str) -> str | None:
    try:
        action = json.loads(line)
    except json.JSONDecodeError:
        return None
    tool = action.get("tool") if isinstance(action, dict) else None
    return tool if isinstance(tool, str) else None


def _evaluate(subject: Subject, gates: Gates, origin: dict, defect: Defect | dict, live: bool) -> dict:
    if isinstance(defect, dict):
        subject.load_state(origin)
        return defect
    print(f"{defect.family}: {defect.name} column {defect.column}", flush=True)
    planted = subject.clone_state()
    planted_weight = planted[defect.name].float().cpu()
    editors = {
        "detector": _run_detector(subject, gates, defect, planted, planted_weight, origin),
        "random": _run_random(subject, gates, defect, planted, planted_weight, seed=0),
        "top_row": _run_top_row(subject, gates, defect, planted, planted_weight),
    }
    if live:
        subject.load_state(planted)
        outcome = run_model(
            subject,
            gates,
            defect.as_plant(),
            "model",
            blind=False,
            frozen=False,
            system=HIDDEN,
            free_layer=True,
        )
        editors["model"] = _score(
            subject, defect, planted_weight, outcome.journal, outcome.held_correct, outcome.held_total
        )
    for name, scores in editors.items():
        print(
            f"  {name}: localization@1={scores['localization_at_1']} "
            f"@3={scores['localization_at_3']} cosine={scores['direction_cosine']:.3f} "
            f"recovery={scores['behavioral_recovery']:.2f}",
            flush=True,
        )
    subject.load_state(origin)
    return {
        "family": defect.family,
        "status": "ran",
        "held_out": True,
        "parameter": defect.name,
        "column": defect.column,
        "scale": defect.scale,
        "held_prompts": len(defect.held),
        "editors": editors,
    }


def _plant_amplified(subject: Subject, gates: Gates, origin: dict) -> Defect | dict:
    subject.load_state(origin)
    plant = find_plant(subject, gates, banned=set())
    if plant is None:
        return {"family": "amplified-direction", "status": "plant_failed"}
    return _from_weights("amplified-direction", subject, origin, plant.name, plant.column, plant.scale)


def _plant_suppressed(subject: Subject, gates: Gates, origin: dict, amplified: Defect | dict) -> Defect | dict:
    """Shrink columns, including down to zero, until one changes enough answers."""
    from selfedit.plant import _ranked_columns

    subject.load_state(origin)
    subject.gold = subject.predict_map()
    names = subject.editable_names()
    ranked = {name: _ranked_columns(subject, name, gates.columns_per_layer) for name in names}
    ordered: list[tuple[str, int]] = []
    if isinstance(amplified, Defect):
        ordered.append((amplified.name, amplified.column))
    for rank in range(gates.columns_per_layer):
        for name in names:
            if rank >= len(ranked[name]):
                continue
            pair = (name, ranked[name][rank])
            if pair not in ordered:
                ordered.append(pair)
    best: dict | None = None
    zero_only: dict | None = None
    trials = 0
    for name, column in ordered:
        if trials >= gates.max_trials:
            break
        strong_flips = None
        strong_stable = None
        for scale in (0.05, 0.1, 0.25, 0.5, 0.0):
            if scale not in (0.05, 0.0) and strong_flips is not None:
                too_few = strong_flips < gates.min_dev + gates.min_held
                enough_stable = strong_stable is not None and strong_stable >= gates.min_guard
                if too_few and enough_stable:
                    continue
            if trials >= gates.max_trials:
                break
            trials += 1
            subject.load_state(origin)
            subject.scale_column(name, column, scale)
            preds = subject.predict_map()
            flipped = [i for i, pred in enumerate(preds) if pred != subject.gold[i]]
            stable = [i for i, pred in enumerate(preds) if pred == subject.gold[i]]
            print(
                f"plant suppressed trial {trials}: {name} col {column} x{scale:g} "
                f"flipped={len(flipped)} stable={len(stable)}",
                flush=True,
            )
            if best is None or len(flipped) > best["flipped"]:
                best = {
                    "parameter": name,
                    "column": column,
                    "scale": scale,
                    "flipped": len(flipped),
                    "stable": len(stable),
                }
            if scale == 0.05:
                strong_flips = len(flipped)
                strong_stable = len(stable)
            dev, held = flipped[0::2], flipped[1::2]
            passed = (
                len(dev) >= gates.min_dev
                and len(held) >= gates.min_held
                and len(stable) >= gates.min_guard
            )
            if not passed:
                continue
            if scale == 0.0:
                zero_only = {
                    "parameter": name,
                    "column": column,
                    "flipped": len(flipped),
                    "stable": len(stable),
                }
                continue
            subject.dev, subject.held, subject.guard = dev, held, stable
            print(
                f"plant suppressed: {name} col {column} x{scale:g} flipped={len(flipped)}",
                flush=True,
            )
            return _from_weights("suppressed-feature", subject, origin, name, column, scale)
    subject.load_state(origin)
    reason = "No shrunk column changed enough answers."
    if zero_only is not None:
        reason = (
            "Zeroing a column changed enough answers, but multiplying a zero column "
            "cannot restore it. No smaller shrink passed."
        )
    report: dict = {
        "family": "suppressed-feature",
        "status": "no_defect",
        "trials": trials,
        "reason": reason,
    }
    if best is not None:
        report["best"] = best
    if zero_only is not None:
        report["zero_only"] = zero_only
    return report


def _plant_rank_one(subject: Subject, gates: Gates, origin: dict) -> Defect | dict:
    subject.load_state(origin)
    subject.gold = subject.predict_map()
    name = subject.editable_names()[0]
    weight = subject.param(name)
    median = float(weight.detach().float().norm(dim=0).median())
    generator = torch.Generator().manual_seed(0)
    out, width = weight.shape
    for scale in (4.0, 8.0, 16.0):
        for _ in range(4):
            u = torch.randn(out, generator=generator)
            v = torch.randn(width, generator=generator)
            u = u / u.norm()
            v = v / v.norm()
            update = (scale * median * u.unsqueeze(1) * v.unsqueeze(0)).to(
                device=weight.device, dtype=weight.dtype
            )
            subject.load_state(origin)
            subject.param(name).data.add_(update)
            column = int(v.abs().argmax())
            defect = _from_behavior(subject, gates, origin, "rank-one", name, column, scale)
            if defect is not None:
                print(
                    f"plant rank-one: {name} dominant col {column} "
                    f"flipped={len(defect.dev) + len(defect.held)}",
                    flush=True,
                )
                return defect
    subject.load_state(origin)
    return {"family": "rank-one", "status": "plant_failed"}


def _single_expert(subject: Subject) -> dict:
    config = subject.model.config
    count = getattr(config, "num_experts", None) or getattr(config, "num_local_experts", None) or 0
    named = any("expert" in name for name, _module in subject.model.named_modules())
    if count or named:
        return {
            "family": "single-expert",
            "status": "plant_failed",
            "reason": "Expert modules are present, but this run does not plant one.",
        }
    return {
        "family": "single-expert",
        "status": "blocked",
        "reason": "This checkpoint is dense. It has no expert modules, so a single-expert defect cannot be planted.",
    }


def _from_weights(family, subject, origin, name, column, scale) -> Defect:
    dev, held, guard = list(subject.dev), list(subject.held), list(subject.guard)
    restore = origin[name].float().cpu() - subject.param(name).detach().float().cpu()
    return Defect(family, name, column, scale, dev, held, guard, restore)


def _from_behavior(subject, gates, origin, family, name, column, scale) -> Defect | None:
    preds = subject.predict_map()
    flipped = [i for i, pred in enumerate(preds) if pred != subject.gold[i]]
    stable = [i for i, pred in enumerate(preds) if pred == subject.gold[i]]
    dev, held = flipped[0::2], flipped[1::2]
    if len(dev) < gates.min_dev or len(held) < gates.min_held or len(stable) < gates.min_guard:
        return None
    subject.dev, subject.held, subject.guard = dev, held, stable
    return _from_weights(family, subject, origin, name, column, scale)


def _run_top_row(subject, gates, defect, planted_state, planted_weight) -> dict:
    """Propose the first column_stats row, the one the model was told to use."""
    from selfedit.toolbox import Toolbox

    subject.load_state(planted_state)
    box = Toolbox(subject, gates, defect.as_plant(), blind=False)
    stats = box.call({"tool": "column_stats"})
    top = (stats.get("top") or [None])[0]
    if isinstance(top, dict):
        box.call(
            {
                "tool": "propose",
                "layer": top.get("layer"),
                "column": top.get("column"),
                "factor": top.get("factor"),
            }
        )
    return _score(subject, defect, planted_weight, box.journal, *subject.counts(subject.held))


def _run_detector(subject, gates, defect, planted_state, planted_weight, origin) -> dict:
    if defect.family == "rank-one":
        return _run_rank_one_detector(subject, defect, planted_state, origin)
    from selfedit.toolbox import Toolbox

    subject.load_state(planted_state)
    box = Toolbox(subject, gates, defect.as_plant(), blind=False)
    ranked = []
    for name in subject.editable_names():
        norms = subject.column_norms(name)
        std = norms.std(unbiased=False).clamp(min=1e-8)
        score = ((norms - norms.mean()) / std).abs()
        median = float(norms.median())
        top = torch.topk(score, k=min(3, int(score.numel())))
        for value, index in zip(top.values.tolist(), top.indices.tolist()):
            factor = median / max(float(norms[index]), 1e-8)
            ranked.append((value, name, int(index), factor))
    ranked.sort(key=lambda item: item[0], reverse=True)
    for _value, name, column, factor in ranked[: gates.max_proposals]:
        factor = min(gates.factor_max, max(gates.factor_min, factor))
        result = box.call({"tool": "propose", "name": name, "column": column, "factor": factor})
        if result.get("accepted") or result.get("error") == "proposals budget is spent":
            break
    return _score(subject, defect, planted_weight, box.journal, *subject.counts(subject.held))


def _run_rank_one_detector(subject, defect, planted_state, origin) -> dict:
    """Undo the strongest rank-one change from the clean weights. No language model."""
    subject.load_state(planted_state)
    best = None
    for name in subject.editable_names():
        delta = subject.param(name).detach().float().cpu() - origin[name].float().cpu()
        _u, singular, _vh = torch.linalg.svd(delta, full_matrices=False)
        value = float(singular[0]) if singular.numel() else 0.0
        if best is None or value > best[0]:
            best = (value, name, delta)
    _value, name, delta = best
    _u, singular, vh = torch.linalg.svd(delta, full_matrices=False)
    strength = singular[0]
    direction = vh[0]
    left = _u[:, 0]
    update = strength * left.unsqueeze(1) * direction.unsqueeze(0)
    weight = subject.param(name)
    weight.data.sub_(update.to(device=weight.device, dtype=weight.dtype))
    columns = torch.topk(direction.abs(), k=min(3, direction.numel())).indices.tolist()
    target = update if name == defect.name else torch.zeros_like(defect.restore)
    cosine = _matrix_cosine(-target, defect.restore)
    held_correct, held_total = subject.counts(subject.held)
    recovery = held_correct / held_total if held_total else 0.0
    hit = name == defect.name and int(columns[0]) == defect.column
    return {
        "localization_at_1": hit,
        "localization_at_3": name == defect.name and defect.column in {int(i) for i in columns},
        "direction_cosine": cosine,
        "behavioral_recovery": recovery,
        "proposals": [
            {"name": name, "column": int(column), "edit": "remove strongest rank-one component"}
            for column in columns
        ],
    }


def _matrix_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = left.norm() * right.norm()
    if float(denom) == 0.0:
        return 0.0
    value = float(torch.dot(left.reshape(-1), right.reshape(-1)) / denom)
    return max(-1.0, min(1.0, value))


def _run_random(subject, gates, defect, planted_state, planted_weight, seed: int) -> dict:
    from selfedit.toolbox import Toolbox

    subject.load_state(planted_state)
    box = Toolbox(subject, gates, defect.as_plant(), blind=False)
    true_norm = float(defect.restore.norm())
    rng = random.Random(seed)
    names = subject.editable_names()
    for _ in range(gates.max_proposals):
        name = rng.choice(names)
        column = rng.randrange(subject.column_count(name))
        column_norm = float(subject.param(name).detach().float()[:, column].norm().clamp(min=1e-8))
        factor = 1.0 + rng.choice((-1.0, 1.0)) * true_norm / column_norm
        factor = min(gates.factor_max, max(gates.factor_min, factor))
        result = box.call({"tool": "propose", "name": name, "column": column, "factor": factor})
        if result.get("accepted") or result.get("error") == "proposals budget is spent":
            break
    return _score(subject, defect, planted_weight, box.journal, *subject.counts(subject.held))


def _score(subject, defect, planted_weight, journal, held_correct, held_total) -> dict:
    proposals = _proposals(subject, journal)
    cosines = [
        _cosine(planted_weight, defect.restore, item["column"], item["factor"])
        if item["name"] == defect.name
        else 0.0
        for item in proposals
    ]
    recovery = held_correct / held_total if held_total else 0.0
    return {
        "localization_at_1": _hit(proposals, defect, 1),
        "localization_at_3": _hit(proposals, defect, 3),
        "direction_cosine": max(cosines) if cosines else 0.0,
        "behavioral_recovery": recovery,
        "proposals": proposals,
    }


def _proposals(subject: Subject, journal: list) -> list[dict]:
    found = []
    for entry in journal:
        action = entry.get("action") or {}
        if action.get("tool") != "propose":
            continue
        name = action.get("name")
        if not isinstance(name, str):
            layer = action.get("layer")
            matches = [item for item in subject.editable_names() if f".layers.{layer}." in item]
            name = matches[0] if len(matches) == 1 else None
        column = action.get("column")
        factor = action.get("factor")
        if not isinstance(name, str) or column is None or factor is None:
            continue
        found.append({"name": name, "column": int(column), "factor": float(factor)})
    return found


def _hit(proposals: list[dict], defect: Defect, k: int) -> bool:
    return any(item["name"] == defect.name and item["column"] == defect.column for item in proposals[:k])


def _cosine(planted: torch.Tensor, restore: torch.Tensor, column: int, factor: float) -> float:
    attempt = torch.zeros_like(restore)
    attempt[:, column] = (factor - 1.0) * planted[:, column].to(restore.dtype)
    denom = attempt.norm() * restore.norm()
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(attempt.reshape(-1), restore.reshape(-1)) / denom)
