"""
Run SELF-EDIT experiments.

Usage:
    python -m selfedit run E1
    python -m selfedit run E12
    python -m selfedit run all
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from selfedit.config import CLAIM_RULE, CONDITIONS, DEFAULT_MODEL, Gates
from selfedit.editors import run_model, run_no_edit, run_oracle, run_random
from selfedit.plant import Plant, find_plant
from selfedit.subject import Subject, TinySubject

ROOT = Path(__file__).resolve().parents[1]


def run_selftest(out_dir: Path | None = None) -> dict:
    torch.set_grad_enabled(False)
    gates = Gates()
    subject = TinySubject(seed=0)
    subject.origin = subject.clone_state()

    first = subject.checksum()
    subject.predict_map()
    if subject.checksum() != first:
        raise AssertionError("M0 failed: a forward pass changed the weights")

    subject.scale_column("down_proj", 0, 8.0)
    if subject.checksum() == first:
        raise AssertionError("M2 failed: scaling a column left the checksum unchanged")
    subject.load_state(subject.origin)
    if subject.checksum() != first:
        raise AssertionError("M2 failed: rollback did not restore the weights")

    plant = find_plant(subject, gates, banned=set())
    if plant is None:
        raise AssertionError("plant search found no selective column on the tiny map")
    planted = subject.clone_state()
    planted_checksum = subject.checksum()

    _assert_heldout_hidden(subject, gates, plant)

    before = subject.checksum()
    box_probe = _toolbox(subject, gates, plant)
    box_probe.call(
        {"tool": "intervene", "name": plant.name, "column": plant.column, "factor": 0.5}
    )
    if subject.checksum() != before:
        raise AssertionError("intervene did not restore the column")

    subject.load_state(planted)
    oracle = run_oracle(subject, gates, plant)
    if not oracle.accepted or not oracle.localized:
        raise AssertionError(
            f"norm oracle failed to undo the planted column: {oracle.to_dict()}"
        )
    if subject.checksum() == planted_checksum:
        raise AssertionError("accepted oracle edit did not change the planted weights")

    subject.load_state(planted)
    no_edit = run_no_edit(subject, plant)
    if no_edit.accepted:
        raise AssertionError("no-edit passed the gate on a planted defect")

    wins = 0
    for seed in range(40):
        subject.load_state(planted)
        if run_random(subject, gates, plant, seed).accepted:
            wins += 1
    if wins > 6:
        raise AssertionError(f"random editor accepted {wins}/40, above the 6/40 cap")

    subject.load_state(subject.origin)
    report = {
        "stage": "selftest",
        "claim_rule": CLAIM_RULE,
        "gates": gates.to_dict(),
        "plant": plant.to_dict(subject.prompts),
        "oracle": oracle.to_dict(),
        "no_edit_accepted": no_edit.accepted,
        "random_accepted_out_of_40": wins,
        "rollback_checksum": subject.checksum(),
    }
    destination = out_dir or (ROOT / "runs" / "selftest")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"selftest passed: oracle localized the x{plant.scale:g} column, "
        f"random accepted {wins}/40, rollback restored the checksum",
        flush=True,
    )
    print(f"wrote {destination / 'report.json'}", flush=True)
    return report


def _down_module(subject, name: str):
    module_name = name.rsplit(".", 1)[0]
    return dict(subject.model.named_modules())[module_name]


def _capture_down(subject, name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward pass. Return next-token logits, the down-projection input, and its output."""
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, args, output):
        captured["x"] = args[0].detach()
        captured["y"] = output.detach()

    handle = _down_module(subject, name).register_forward_hook(hook)
    try:
        logits = subject.next_logits()
    finally:
        handle.remove()
    return logits, captured["x"], captured["y"]


def open_model(model_id: str = DEFAULT_MODEL, device: str = "auto"):
    """Load the checkpoint once. Later experiments reuse this object."""
    torch.set_grad_enabled(False)
    from selfedit.hf_subject import HFSubject

    subject = HFSubject(model_id, device=device)
    subject.origin = subject.clone_state()
    return subject


def run_e1(
    model_id: str = DEFAULT_MODEL,
    device: str = "auto",
    out_dir: Path | None = None,
    column: int = 0,
    delta: float = 2.0,
    subject=None,
) -> dict:
    """Change one column by a known factor through the tool.

    The down-projection output must equal the output predicted from that factor.
    The hash of the parameter read back must equal the hash of the edited copy.
    Rollback must restore bitwise-identical next-token logits.
    """
    torch.set_grad_enabled(False)
    from selfedit.subject import tensor_hash

    if subject is None:
        subject = open_model(model_id, device)
    else:
        subject.load_state(subject.origin)
    try:
        return _run_e1_body(subject, model_id, out_dir, column, delta, tensor_hash)
    finally:
        subject.load_state(subject.origin)


def _run_e1_body(subject, model_id, out_dir, column, delta, tensor_hash) -> dict:
    name = subject.editable_names()[0]
    clean_logits, clean_input, clean_output = _capture_down(subject, name)
    weight = subject.param(name).detach()
    edited = weight.clone()
    edited[:, column].mul_(float(delta))
    bias = _down_module(subject, name).bias
    predicted = torch.nn.functional.linear(clean_input, edited.to(dtype=clean_input.dtype))
    if bias is not None:
        predicted = predicted + bias
    if torch.equal(predicted, clean_output):
        raise SystemExit(f"multiplying {name} column {column} by {delta:g} predicts no change")
    expected_hash = tensor_hash(edited)
    original_hash = tensor_hash(weight)

    plant = Plant(name, column, delta, 0, [], [], [])
    box = _toolbox(subject, Gates(), plant)
    applied = box.call({"tool": "apply", "name": name, "column": column, "factor": delta})
    readback_hash = tensor_hash(subject.param(name).data)
    if applied.get("readback_hash") != expected_hash or readback_hash != expected_hash:
        raise AssertionError("read-back hash does not match the known edit")

    _edited_logits, edited_input, edited_output = _capture_down(subject, name)
    if not torch.equal(edited_input, clean_input):
        raise AssertionError("the edit changed the input of the layer it was applied to")
    if not torch.equal(edited_output, predicted):
        raise AssertionError("forward output does not match the prediction for this delta")

    restored = box.call({"tool": "rollback"})
    if restored.get("readback_hash") != original_hash or tensor_hash(subject.param(name).data) != original_hash:
        raise AssertionError("rollback did not restore the parameter bytes")
    if not torch.equal(subject.next_logits(), clean_logits):
        raise AssertionError("rollback did not restore bitwise-identical logits")

    report = {
        "stage": "e1",
        "model": model_id,
        "device": str(subject.device),
        "dtype": subject.dtype,
        "parameter": name,
        "column": column,
        "delta": delta,
        "applied_through_tool": True,
        "forward_matches_prediction": True,
        "expected_hash": expected_hash,
        "readback_hash": readback_hash,
        "readback_hash_matches": True,
        "rollback_hash": original_hash,
        "rollback_logits_identical": True,
        "runtimes": {
            "this_harness": "passed",
            "production": {
                "status": "not_run",
                "reason": "No second runtime is configured.",
            },
        },
    }
    destination = out_dir or (ROOT / "runs" / "e1")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"E1 passed on {model_id}: {name} column {column} x{delta:g} through the tool, "
        f"forward matched the prediction, read-back hash matched, "
        f"rollback restored the logits",
        flush=True,
    )
    print(f"wrote {destination / 'report.json'}", flush=True)
    return report


def run_m3(
    model_id: str = DEFAULT_MODEL,
    rounds: int = 3,
    seed: int = 0,
    conditions: tuple[str, ...] = CONDITIONS,
    device: str = "auto",
    out_dir: Path | None = None,
) -> dict:
    torch.set_grad_enabled(False)
    from selfedit.hf_subject import HFSubject

    unknown = [name for name in conditions if name not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown condition: {', '.join(unknown)}")
    gates = Gates()
    subject = HFSubject(model_id, device=device)
    if not subject.editable_names():
        raise SystemExit("this checkpoint has no mlp.down_proj.weight tensors")
    subject.origin = subject.clone_state()
    banned: set[tuple[str, int]] = set()
    round_reports = []
    journals = []
    for round_index in range(rounds):
        print(f"round {round_index + 1}/{rounds}", flush=True)
        before = subject.clone_state()
        clean_checksum = subject.checksum()
        plant = find_plant(subject, gates, banned)
        if plant is None:
            round_reports.append(
                {
                    "round": round_index,
                    "status": "plant_failed",
                    "clean_checksum": clean_checksum,
                }
            )
            continue
        banned.add((plant.name, plant.column))
        planted_state = subject.clone_state()
        planted_checksum = subject.checksum()
        condition_reports = {}
        for name in conditions:
            subject.load_state(planted_state)
            outcome = _run_condition(name, subject, gates, plant, seed + 1000 * round_index)
            condition_reports[name] = outcome.to_dict()
            for entry in outcome.journal:
                journals.append({"round": round_index, "condition": name, **entry})
            print(
                f"  {name}: accepted={outcome.accepted} localized={outcome.localized} "
                f"held={outcome.held_correct}/{outcome.held_total} "
                f"guard={outcome.guard_correct}/{outcome.guard_total}",
                flush=True,
            )
        if "live" in condition_reports and condition_reports["live"]["accepted"]:
            subject.load_state(planted_state)
            op = condition_reports["live"]["op"]
            subject.scale_column(op["name"], op["column"], op["factor"])
            status = "committed"
        else:
            subject.load_state(before)
            status = "rolled_back"
        round_reports.append(
            {
                "round": round_index,
                "status": status,
                "clean_checksum": clean_checksum,
                "planted_checksum": planted_checksum,
                "plant": plant.to_dict(subject.prompts),
                "conditions": condition_reports,
            }
        )
    summary = _summarize(round_reports)
    report = {
        "stage": "M3",
        "model": model_id,
        "commit": subject.commit,
        "device": str(subject.device),
        "dtype": subject.dtype,
        "gates": gates.to_dict(),
        "claim_rule": CLAIM_RULE,
        "conditions": list(conditions),
        "rounds": round_reports,
        "summary": summary,
        "decision": _decision(summary),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = out_dir or (ROOT / "runs" / f"m3-{stamp}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "journal.jsonl").write_text(
        "".join(json.dumps(entry, ensure_ascii=True) + "\n" for entry in journals)
    )
    print(_decision(summary), flush=True)
    print(f"wrote {destination / 'report.json'}", flush=True)
    return report


def _run_condition(name: str, subject: Subject, gates: Gates, plant: Plant, seed: int):
    if name == "no_edit":
        return run_no_edit(subject, plant)
    if name == "random":
        return run_random(subject, gates, plant, seed)
    if name == "oracle":
        return run_oracle(subject, gates, plant)
    if name == "blind":
        return run_model(subject, gates, plant, "blind", blind=True, frozen=False)
    if name == "frozen":
        return run_model(subject, gates, plant, "frozen", blind=False, frozen=True)
    if name == "live":
        return run_model(subject, gates, plant, "live", blind=False, frozen=False)
    raise KeyError(name)


def _toolbox(subject: Subject, gates: Gates, plant: Plant):
    from selfedit.toolbox import Toolbox

    return Toolbox(subject, gates, plant, blind=False)


def _assert_heldout_hidden(subject: Subject, gates: Gates, plant: Plant) -> None:
    box = _toolbox(subject, gates, plant)
    probes = box.call({"tool": "probes"})
    blob = json.dumps(probes)
    if "localized" in probes or "secret" in blob:
        raise AssertionError("probe payload exposes the plant")
    for index in plant.held:
        if subject.prompts[index] in blob:
            raise AssertionError("probe payload includes a held-out prompt")


def _summarize(rounds: list[dict]) -> dict:
    planted = [row for row in rounds if row.get("plant")]
    summary = {
        "rounds": len(rounds),
        "planted_rounds": len(planted),
        "plant_failed": sum(row.get("status") == "plant_failed" for row in rounds),
    }
    for name in CONDITIONS:
        present = [row["conditions"][name] for row in planted if name in row.get("conditions", {})]
        summary[f"{name}_accepted"] = sum(item["accepted"] for item in present)
        summary[f"{name}_localized"] = sum(item["localized"] for item in present)
        summary[f"{name}_rounds"] = len(present)
    later = [row for row in planted if row["round"] > 0 and "live" in row["conditions"]]
    summary["live_later_accepted"] = sum(row["conditions"]["live"]["accepted"] for row in later)
    summary["frozen_later_accepted"] = sum(
        row["conditions"]["frozen"]["accepted"] for row in later if "frozen" in row["conditions"]
    )
    summary["later_rounds"] = len(later)
    return summary


def _decision(summary: dict) -> str:
    if summary["planted_rounds"] == 0:
        return "No selective column was found, so no editor was scored."
    labels = (
        ("live", "Live"),
        ("random", "Random"),
        ("oracle", "The norm oracle"),
        ("blind", "Weight-blind"),
        ("frozen", "The frozen editor"),
        ("no_edit", "No-edit"),
    )
    parts = []
    for name, label in labels:
        n = summary.get(f"{name}_rounds", 0)
        if not n:
            continue
        localized = ""
        if name == "live":
            localized = f" and localized {summary['live_localized']}"
        parts.append(f"{label} accepted {summary[f'{name}_accepted']} of {n}{localized}")
    lines = [". ".join(parts) + "."]
    if summary.get("live_rounds", 0) and summary.get("random_rounds", 0) and summary.get("blind_rounds", 0):
        live = summary["live_accepted"]
        ahead = (
            live > summary["random_accepted"]
            and live > summary["blind_accepted"]
            and summary["live_localized"] > 0
        )
        if ahead:
            lines.append("This run meets the M3 bar for a scaled-column defect.")
        else:
            lines.append("This run is below the M3 bar for a scaled-column defect.")
    if (
        summary.get("live_rounds", 0)
        and summary.get("oracle_rounds", 0)
        and summary.get("random_rounds", 0)
        and summary["oracle_accepted"] >= summary["live_accepted"]
        and summary["oracle_accepted"] > summary["random_accepted"]
    ):
        lines.append("Repairs that succeeded line up with the norm shortcut.")
    if summary["later_rounds"] > 0 and summary.get("frozen_rounds", 0) > 0:
        if summary["live_later_accepted"] > summary["frozen_later_accepted"]:
            lines.append("Later rounds favor the updated model over the frozen original editor.")
        else:
            lines.append("Later rounds keep the frozen original editor level with, or ahead of, the updated model.")
    return " ".join(lines)


def experiment_key(name: str) -> str:
    """Accept E12, e12, or 12. Return E12. 'all' is returned as ALL."""
    text = name.strip().upper()
    if text == "ALL":
        return "ALL"
    number = text[1:] if text.startswith("E") else text
    if number.isdigit() and 1 <= int(number) <= 37:
        return f"E{int(number)}"
    raise SystemExit(f"unknown experiment: {name}. Use E1 through E37, or all.")


def _stamp_dir(key: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "runs" / f"{key.lower()}-{stamp}"


def dispatch(
    name: str | list[str],
    model: str = DEFAULT_MODEL,
    device: str = "auto",
    out_dir: Path | None = None,
    live: bool = True,
    rounds: int = 3,
) -> None:
    names = [name] if isinstance(name, str) else list(name)
    keys = [experiment_key(item) for item in names]
    if "ALL" in keys:
        run_all(model=model, device=device, out_dir=out_dir, live=live, rounds=rounds)
        return
    subject = None
    if len(keys) > 1:
        print(f"loading {model} once for {', '.join(keys)}", flush=True)
        subject = open_model(model, device)
    for key in keys:
        destination = out_dir if len(keys) == 1 else (out_dir / key if out_dir else _stamp_dir(key))
        if out_dir is not None and len(keys) > 1:
            destination = out_dir / key
        print(f"=== {key} ===", flush=True)
        _run_key(key, model=model, device=device, out_dir=destination, live=live, subject=subject)
        if subject is not None:
            subject.load_state(subject.origin)


def _run_key(key, model, device, out_dir, live, subject) -> None:
    if key == "E1":
        run_e1(model_id=model, device=device, out_dir=out_dir, subject=subject)
        return
    if key == "E2":
        from selfedit.e2 import run_e2

        run_e2(model_id=model, device=device, out_dir=out_dir, live=live, subject=subject)
        return
    if key == "E3":
        from selfedit.ladder import run_e3_only

        run_e3_only(model_id=model, device=device, live=live, out_dir=out_dir, subject=subject)
        return
    if key == "E4":
        from selfedit.ladder import run_e4_only

        run_e4_only(model_id=model, device=device, live=live, out_dir=out_dir, subject=subject)
        return
    from selfedit.programme import run_programme

    run_programme(model_id=model, device=device, out_dir=out_dir, only=(key,), subject=subject)


def run_all(
    model: str = DEFAULT_MODEL,
    device: str = "auto",
    out_dir: Path | None = None,
    live: bool = True,
    rounds: int = 3,
) -> None:
    """Run E1 through E37 in this process. The checkpoint is loaded once."""
    del rounds
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = out_dir or (ROOT / "runs" / f"all-{stamp}")
    root.mkdir(parents=True, exist_ok=True)
    print(f"loading {model} once", flush=True)
    subject = open_model(model, device)
    status_path = root / "status.tsv"
    with status_path.open("w") as status:
        status.write("experiment\texit\tseconds\n")
        for number in range(1, 38):
            key = f"E{number}"
            folder = root / f"E{number:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            print(f"=== {key} ===", flush=True)
            started = time.monotonic()
            code = 0
            try:
                _run_key(key, model=model, device=device, out_dir=folder, live=live, subject=subject)
            except Exception as exc:
                code = 1
                print(f"    error: {type(exc).__name__}: {exc}", flush=True)
            finally:
                subject.load_state(subject.origin)
            elapsed = round(time.monotonic() - started)
            status.write(f"{key}\t{code}\t{elapsed}\n")
            status.flush()
            print(f"    exit {code} in {elapsed}s", flush=True)
    print(f"status: {status_path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SELF-EDIT controlled self-repair experiment")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one experiment (E1-E37) or all of them")
    run.add_argument("experiment", nargs="+", help="one or more of E1 through E37, or all")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--device", default="auto")
    run.add_argument("--rounds", type=int, default=3, help="E2 only: planted rounds")
    run.add_argument("--no-live", action="store_true", help="E3 and E4 only: skip the live model")
    run.add_argument("--out", type=Path, default=None)
    selftest = sub.add_parser("selftest", help="check the harness on a tiny map (same as run E1)")
    selftest.add_argument("--out", type=Path, default=None)
    m3 = sub.add_parser("m3", help="plant a scaled column inside a live language model (same as run E2)")
    m3.add_argument("--model", default=DEFAULT_MODEL)
    m3.add_argument("--rounds", type=int, default=3)
    m3.add_argument("--seed", type=int, default=0)
    m3.add_argument("--device", default="auto")
    m3.add_argument("--conditions", default=",".join(CONDITIONS))
    m3.add_argument("--out", type=Path, default=None)
    sweep = sub.add_parser("e3e4", help="run E3 and E4 together, with E4 using E3's scale")
    sweep.add_argument("--model", default=DEFAULT_MODEL)
    sweep.add_argument("--device", default="auto")
    sweep.add_argument("--no-live", action="store_true")
    sweep.add_argument("--out", type=Path, default=None)
    programme = sub.add_parser("programme", help="run E5 through E37 in one process")
    programme.add_argument("--model", default=DEFAULT_MODEL)
    programme.add_argument("--device", default="auto")
    programme.add_argument("--out", type=Path, default=None)
    programme.add_argument("--only", default=None, help="comma-separated experiments, for example E6,E12")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        try:
            dispatch(
                args.experiment,
                model=args.model,
                device=args.device,
                out_dir=args.out,
                live=not args.no_live,
                rounds=args.rounds,
            )
        except KeyboardInterrupt:
            sys.exit(130)
        return
    if args.cmd == "selftest":
        run_selftest(args.out)
        return
    if args.cmd == "programme":
        from selfedit.programme import run_programme

        only = None
        if args.only:
            only = tuple(part.strip() for part in args.only.split(",") if part.strip())
        run_programme(model_id=args.model, device=args.device, out_dir=args.out, only=only)
        return
    if args.cmd == "e3e4":
        from selfedit.ladder import run_e3_e4

        run_e3_e4(
            model_id=args.model,
            device=args.device,
            live=not args.no_live,
            out_dir=args.out,
        )
        return
    conditions = tuple(part.strip() for part in args.conditions.split(",") if part.strip())
    try:
        run_m3(
            model_id=args.model,
            rounds=args.rounds,
            seed=args.seed,
            conditions=conditions,
            device=args.device,
            out_dir=args.out,
        )
    except KeyboardInterrupt:
        sys.exit(130)
