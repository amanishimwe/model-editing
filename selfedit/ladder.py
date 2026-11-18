"""E3 perturbation staircase and E4 information ladder.

E3 estimates the scale at which a repair succeeds.
E4 repeats one planted column while changing which tools the editor may use.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from selfedit.config import DEFAULT_MODEL, Gates
from selfedit.editors import Outcome, _finish, run_model, run_oracle, run_random
from selfedit.plant import Plant, find_plant
from selfedit.subject import Subject
from selfedit.toolbox import Toolbox

ROOT = Path(__file__).resolve().parents[1]

RUNGS: tuple[tuple[str, frozenset[str]], ...] = (
    ("behavior", frozenset({"manifest", "probes", "propose", "stop"})),
    ("activations", frozenset({"manifest", "probes", "activations", "propose", "stop"})),
    ("norm_stats", frozenset({"manifest", "probes", "column_stats", "propose", "stop"})),
    ("slice", frozenset({"manifest", "probes", "column", "propose", "stop"})),
    ("intervene", frozenset({"manifest", "probes", "intervene", "propose", "stop"})),
)


def plant_scale(
    subject: Subject,
    gates: Gates,
    clean: dict,
    name: str,
    column: int,
    scale: float,
) -> Plant | None:
    """Apply one scale to a known column. Return a plant only if the split is valid."""
    subject.load_state(clean)
    subject.gold = subject.predict_map()
    subject.scale_column(name, column, scale)
    preds = subject.predict_map()
    flipped = [i for i, pred in enumerate(preds) if pred != subject.gold[i]]
    stable = [i for i, pred in enumerate(preds) if pred == subject.gold[i]]
    dev, held = flipped[0::2], flipped[1::2]
    subject.dev, subject.held, subject.guard = dev, held, stable
    valid = (
        len(dev) >= gates.min_dev
        and len(held) >= gates.min_held
        and len(stable) >= gates.min_guard
    )
    if not valid:
        subject.load_state(clean)
        return None
    return Plant(name, column, scale, 1, dev, held, stable)


def _record(kind: str, scale: float, defect: bool, outcome: Outcome | None) -> dict:
    return {
        "editor": kind,
        "scale": scale,
        "defect": defect,
        "accepted": bool(outcome and outcome.accepted),
        "localized": bool(outcome and outcome.localized),
        "held_correct": None if outcome is None else outcome.held_correct,
        "held_total": None if outcome is None else outcome.held_total,
        "guard_correct": None if outcome is None else outcome.guard_correct,
        "guard_total": None if outcome is None else outcome.guard_total,
    }


def _both_sides(rows: list[dict]) -> bool:
    return any(row["defect"] and row["localized"] for row in rows) and any(
        row["defect"] and not row["localized"] for row in rows
    )


def _boundary(rows: list[dict], key: str) -> dict:
    """Geometric boundary between the largest miss and the smallest hit."""
    hits = [row["scale"] for row in rows if row["defect"] and row[key]]
    misses = [row["scale"] for row in rows if row["defect"] and not row[key]]
    silent = [row["scale"] for row in rows if not row["defect"]]
    broken = [row["scale"] for row in rows if row["defect"]]
    out: dict = {
        "hits": hits,
        "misses": misses,
        "threshold": None,
        "bound": None,
        "behavior_threshold": None,
    }
    if silent and broken:
        below = [scale for scale in silent if scale < min(broken)]
        if below:
            out["behavior_threshold"] = math.sqrt(max(below) * min(broken))
    if hits and misses and any(scale < min(hits) for scale in misses):
        below = max(scale for scale in misses if scale < min(hits))
        out["threshold"] = math.sqrt(below * min(hits))
        out["bound"] = "between"
    elif hits:
        out["threshold"] = min(hits)
        out["bound"] = "at_or_below"
    elif misses or broken:
        out["threshold"] = max(misses or broken)
        out["bound"] = "above"
    recoveries = []
    for row in rows:
        total = row.get("held_total") or 0
        correct = row.get("held_correct")
        if row["defect"] and total and correct is not None:
            recoveries.append((row["scale"], correct / total))
    passed = [scale for scale, rate in recoveries if rate >= 0.5]
    failed = [scale for scale, rate in recoveries if rate < 0.5]
    recovery: dict = {"threshold": None, "bound": None}
    if passed and failed and any(scale < min(passed) for scale in failed):
        below = max(scale for scale in failed if scale < min(passed))
        recovery = {"threshold": math.sqrt(below * min(passed)), "bound": "between"}
    elif passed:
        recovery = {"threshold": min(passed), "bound": "at_or_below"}
    elif failed:
        recovery = {"threshold": max(failed), "bound": "above"}
    out["recovery_50"] = recovery
    return out


def run_e3(
    subject: Subject,
    gates: Gates,
    clean: dict,
    name: str,
    column: int,
    live: bool,
) -> dict:
    """Adaptive staircase on the norm oracle, then the same column for the live model."""
    scale = 4.0
    step = 2.0
    seen: set[float] = set()
    rows: list[dict] = []
    print("E3 oracle staircase", flush=True)
    for _ in range(8):
        scale = round(scale, 4)
        if scale in seen or not (1.25 <= scale <= 32):
            break
        seen.add(scale)
        plant = plant_scale(subject, gates, clean, name, column, scale)
        if plant is None:
            print(f"  x{scale:g}: no selective defect", flush=True)
            rows.append(_record("oracle", scale, False, None))
            scale *= step
            continue
        outcome = run_oracle(subject, gates, plant)
        ok = outcome.accepted and outcome.localized
        print(
            f"  x{scale:g}: oracle accepted={outcome.accepted} localized={outcome.localized}",
            flush=True,
        )
        rows.append(_record("oracle", scale, True, outcome))
        step = max(math.sqrt(step), 1.25) if _both_sides(rows) else step
        scale = min(32.0, max(1.25, scale / step if ok else scale * step))
    oracle_bound = _boundary(rows, "localized")

    live_rows: list[dict] = []
    if live:
        candidates = [row["scale"] for row in rows if row["defect"] and row["localized"]]
        probe_scales = []
        if candidates:
            probe_scales.append(max(candidates))
        if 16.0 not in probe_scales:
            probe_scales.append(16.0)
        live_gates = replace(gates, max_turns=8, max_new_tokens=128)
        print("E3 live probe", flush=True)
        for scale in probe_scales:
            plant = plant_scale(subject, live_gates, clean, name, column, scale)
            if plant is None:
                live_rows.append(_record("live", scale, False, None))
                print(f"  x{scale:g}: no selective defect", flush=True)
                continue
            outcome = run_model(
                subject, live_gates, plant, "live", blind=False, frozen=False
            )
            print(
                f"  x{scale:g}: live accepted={outcome.accepted} localized={outcome.localized}",
                flush=True,
            )
            live_rows.append(_record("live", scale, True, outcome))
            if outcome.accepted and outcome.localized:
                break
    subject.load_state(clean)
    return {
        "column": {"name": name, "column": column},
        "oracle": rows,
        "oracle_bound": oracle_bound,
        "live": live_rows,
        "live_bound": _boundary(live_rows, "localized") if live_rows else None,
    }


def _activation_editor(subject: Subject, gates: Gates, plant: Plant) -> Outcome:
    allowed = set(RUNGS[1][1])
    box = Toolbox(subject, gates, plant, blind=False, allowed=allowed)
    probes = box.call({"tool": "probes"})
    failures = probes.get("dev_failures") or []
    manifest = box.call({"tool": "manifest"})
    best: tuple[float, str, int] | None = None
    if failures:
        prompt = failures[0]["prompt"]
        for row in manifest.get("editable", []):
            stats = box.call({"tool": "activations", "name": row["name"], "prompt": prompt})
            for item in stats.get("top", []):
                rank = (float(item["activation"]), row["name"], int(item["column"]))
                if best is None or rank[0] > best[0]:
                    best = rank
    if best is not None:
        for factor in (0.25, 0.5, 0.125):
            result = box.call(
                {"tool": "propose", "name": best[1], "column": best[2], "factor": factor}
            )
            if result.get("accepted"):
                break
    return _finish("activations", subject, box, box.proposals, [])


def _slice_editor(subject: Subject, gates: Gates, plant: Plant) -> Outcome:
    allowed = set(RUNGS[3][1])
    box = Toolbox(subject, gates, plant, blind=False, allowed=allowed)
    manifest = box.call({"tool": "manifest"})
    name = manifest["editable"][0]["name"]
    for column in (0, 1, 2):
        box.call({"tool": "column", "name": name, "column": column})
        result = box.call({"tool": "propose", "name": name, "column": column, "factor": 0.25})
        if result.get("accepted"):
            break
    return _finish("slice", subject, box, box.proposals, [])


def _intervene_editor(subject: Subject, gates: Gates, plant: Plant, seed: int) -> Outcome:
    allowed = set(RUNGS[4][1])
    box = Toolbox(subject, gates, plant, blind=False, allowed=allowed)
    rng = random.Random(seed)
    manifest = box.call({"tool": "manifest"})
    names = [row["name"] for row in manifest.get("editable", [])]
    found: tuple[str, int, float] | None = None
    for _ in range(gates.max_interventions):
        name = rng.choice(names)
        column = rng.randrange(subject.column_count(name))
        factor = rng.choice((0.25, 0.125, 0.0625))
        result = box.call(
            {"tool": "intervene", "name": name, "column": column, "factor": factor}
        )
        if result.get("dev_correct") == result.get("dev_total") and result.get("dev_total"):
            found = (name, column, factor)
            break
    if found is not None:
        box.call(
            {"tool": "propose", "name": found[0], "column": found[1], "factor": found[2]}
        )
    return _finish("intervene", subject, box, box.proposals, [])


def _logit_lens(subject: Subject, name: str, column: int) -> list[str]:
    vector = subject.param(name).detach().float().cpu()[:, column]
    embeddings = getattr(subject, "model", None)
    head = None
    if embeddings is not None and hasattr(embeddings, "get_output_embeddings"):
        output = embeddings.get_output_embeddings()
        if output is not None:
            head = output.weight.detach().float().cpu()
    if head is None:
        return []
    scores = head @ vector.to(head.dtype)
    indices = torch.topk(scores, k=5).indices.tolist()
    return [subject.label(int(index)) for index in indices]


def run_e4(
    subject: Subject,
    gates: Gates,
    clean: dict,
    name: str,
    column: int,
    scale: float,
    live: bool,
    seed: int,
) -> dict:
    plant = plant_scale(subject, gates, clean, name, column, scale)
    if plant is None:
        subject.load_state(clean)
        return {"status": "no_defect", "scale": scale}
    planted = subject.clone_state()
    scripted: dict[str, dict] = {}
    print(f"E4 scripted ladder at x{scale:g}", flush=True)
    runners = {
        "behavior": lambda: run_random(subject, gates, plant, seed),
        "activations": lambda: _activation_editor(subject, gates, plant),
        "norm_stats": lambda: run_oracle(subject, gates, plant),
        "slice": lambda: _slice_editor(subject, gates, plant),
        "intervene": lambda: _intervene_editor(subject, gates, plant, seed),
    }
    for rung, _allowed in RUNGS:
        subject.load_state(planted)
        outcome = runners[rung]()
        scripted[rung] = {
            "accepted": outcome.accepted,
            "localized": outcome.localized,
            "held_correct": outcome.held_correct,
            "held_total": outcome.held_total,
        }
        print(
            f"  {rung}: accepted={outcome.accepted} localized={outcome.localized}",
            flush=True,
        )
    lens = _logit_lens(subject, name, column)
    weight = subject.param(name).detach().float().cpu()
    column_vector = weight[:, column]
    formats = {
        "raw": {
            "mean": round(float(column_vector.mean()), 4),
            "std": round(float(column_vector.std(unbiased=False)), 4),
            "norm": round(float(column_vector.norm()), 4),
        },
        "spectral_column_share": round(float(column_vector.norm() / weight.norm().clamp(min=1e-8)), 4),
        "logit_lens": lens,
        "live_format_sweep": "not_run",
    }
    live_rows: dict[str, dict] = {}
    if live:
        live_gates = replace(gates, max_turns=8, max_new_tokens=128)
        print("E4 live ladder", flush=True)
        for rung, allowed in RUNGS:
            subject.load_state(planted)
            outcome = run_model(
                subject,
                live_gates,
                plant,
                f"live-{rung}",
                blind=False,
                frozen=False,
                allowed=set(allowed),
            )
            live_rows[rung] = {
                "accepted": outcome.accepted,
                "localized": outcome.localized,
                "turns": outcome.turns,
                "proposals": outcome.proposals,
            }
            print(
                f"  live {rung}: accepted={outcome.accepted} localized={outcome.localized}",
                flush=True,
            )
    subject.load_state(clean)
    return {
        "scale": scale,
        "column": {"name": name, "column": column},
        "scripted": scripted,
        "logit_lens_top_tokens": lens,
        "formats": formats,
        "live": live_rows,
    }


def _load_plant(model_id: str, device: str, subject=None):
    """Load the checkpoint and find one selective column, then restore the clean weights."""
    torch.set_grad_enabled(False)
    gates = Gates()
    if subject is None:
        from selfedit.hf_subject import HFSubject

        subject = HFSubject(model_id, device=device)
        subject.origin = subject.clone_state()
    else:
        subject.load_state(subject.origin)
    clean = subject.origin
    plant = find_plant(subject, gates, banned=set())
    if plant is None:
        raise SystemExit("no selective column")
    subject.load_state(clean)
    return subject, gates, clean, plant


def _save(prefix: str, report: dict, out_dir: Path | None) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = out_dir or (ROOT / "runs" / f"{prefix}-{stamp}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {destination / 'report.json'}", flush=True)
    return report


def run_e3_only(
    model_id: str = DEFAULT_MODEL,
    device: str = "auto",
    live: bool = True,
    out_dir: Path | None = None,
    subject=None,
) -> dict:
    subject, gates, clean, plant = _load_plant(model_id, device, subject)
    e3 = run_e3(subject, gates, clean, plant.name, plant.column, live=live)
    decision = _e3_text(e3)
    print(decision, flush=True)
    return _save(
        "e3",
        {"model": model_id, "experiment": "E3", "e3": e3, "decision": decision},
        out_dir,
    )


def run_e4_only(
    model_id: str = DEFAULT_MODEL,
    device: str = "auto",
    live: bool = True,
    out_dir: Path | None = None,
    subject=None,
) -> dict:
    """Run E4 on its own, at the scale the plant search found.

    The combined e3e4 command instead uses the largest scale where the oracle succeeded.
    """
    subject, gates, clean, plant = _load_plant(model_id, device, subject)
    e4 = run_e4(subject, gates, clean, plant.name, plant.column, plant.scale, live=live, seed=0)
    decision = _e4_text(e4)
    print(decision, flush=True)
    return _save(
        "e4",
        {
            "model": model_id,
            "experiment": "E4",
            "scale_source": "plant",
            "e4": e4,
            "decision": decision,
        },
        out_dir,
    )


def run_e3_e4(
    model_id: str = DEFAULT_MODEL,
    device: str = "auto",
    live: bool = True,
    out_dir: Path | None = None,
) -> dict:
    subject, gates, clean, plant = _load_plant(model_id, device)
    name, column = plant.name, plant.column
    e3 = run_e3(subject, gates, clean, name, column, live=live)
    e4_scale = 4.0
    hits = [row["scale"] for row in e3["oracle"] if row["localized"]]
    if hits:
        e4_scale = max(hits)
    e4 = run_e4(subject, gates, clean, name, column, e4_scale, live=live, seed=0)
    e3_decision = _e3_text(e3)
    e4_decision = _e4_text(e4)
    print(e3_decision, flush=True)
    print(e4_decision, flush=True)
    return _save(
        "e3-e4",
        {
            "model": model_id,
            "e3": e3,
            "e4": e4,
            "e3_decision": e3_decision,
            "e4_decision": e4_decision,
        },
        out_dir,
    )


def _fmt_bound(bound: dict | None, label: str) -> str:
    if not bound or bound.get("threshold") is None:
        return f"{label} has no threshold."
    value = bound["threshold"]
    kind = bound.get("bound")
    if kind == "between":
        return f"{label} crosses 50% near x{value:.2f}."
    if kind == "at_or_below":
        return f"{label} still succeeds at x{value:g}, the smallest hit in the sweep."
    return f"{label} does not succeed at x{value:g}, the largest scale in the sweep."


def _e3_text(e3: dict) -> str:
    behavior = e3["oracle_bound"].get("behavior_threshold")
    behavior_text = (
        f"The column starts changing held-out tokens near x{behavior:.2f}."
        if behavior
        else "Every tested scale that was scored had already changed held-out tokens."
    )
    oracle_recovery = (e3["oracle_bound"].get("recovery_50") or {}).get("threshold")
    recovery_text = (
        f"The oracle 50% held-out recovery point is x{oracle_recovery:.2f}."
        if oracle_recovery
        else "The sweep did not yield a 50% held-out recovery point."
    )
    return " ".join(
        [
            behavior_text,
            recovery_text,
            _fmt_bound(e3["oracle_bound"], "The norm oracle"),
            _fmt_bound(e3.get("live_bound"), "The live model"),
        ]
    )


def _e4_text(e4: dict) -> str:
    if e4.get("status") == "no_defect":
        return "E4 had no selective defect at the chosen scale."
    scripted = e4["scripted"]
    sufficient = [name for name, row in scripted.items() if row["localized"]]
    live = e4.get("live") or {}
    live_hits = [name for name, row in live.items() if row["localized"]]
    channel = ", ".join(sufficient) if sufficient else "none"
    live_text = ", ".join(live_hits) if live_hits else "none"
    return (
        f"Scripted recovery by channel: {channel}. "
        f"Live recovery by channel: {live_text}. "
        f"Logit-lens tokens on the planted column: {e4.get('logit_lens_top_tokens')}."
    )
