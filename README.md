# SELF-EDIT

## Setup

```bash
uv sync --python /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13
```

## Run one experiment

There are 37 experiments, named E1 through E37. Each one writes its own folder under `runs/`.

```bash
.venv/bin/python -m selfedit run E1
.venv/bin/python -m selfedit run E1 E2
```

`run E1 E2` loads Qwen once and runs both experiments on that copy. `run all` does the same for all 37.

E1 through E37 load `Qwen/Qwen2.5-0.5B-Instruct`. Pass `--model` to use another causal LM that has `mlp.down_proj.weight`.

E1 checks one known edit. The tool multiplies one column by a known factor, the layer output must match the prediction, the read-back hash must match, and rollback must restore the original next-token logits. `selftest` is a separate harness check on a tiny map and does not load Qwen.

## E2

E2 plants hidden defects from four families: rank-one, amplified-direction, suppressed-feature, and single-expert. Qwen2.5-0.5B is dense, so the single-expert family is blocked. Each other family is held out. The editors are not fit on it, and recovery is scored on prompts they are not shown.

Three editors see each planted defect:

- **detector**, a script with no language model. For a scaled column, it tries the columns whose norms are farthest from the mean and sets the factor to the median norm divided by that column's norm. For a rank-one change, it removes the strongest direction of the change. This is the upper bound.
- **random** makes three edits whose size matches the defect, on columns it does not choose for a reason.
- **model** is the live model. It is not told which family was planted.

The report gives localization@1, localization@3, the direction cosine of the closest proposal, and behavioral recovery on the held-out prompts.

## Run all 37

This runs E1, then E2, and so on, each in its own process. A failed experiment is recorded and the next one still runs. Reports go in `runs/all-<timestamp>/E01` through `E37`, with a `status.tsv` of exit codes.

```bash
.venv/bin/python -m selfedit run all
```
