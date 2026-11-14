# SELF-EDIT

One loaded checkpoint inspects a hidden change in its own weights and proposes a one-column repair. The harness commits the edit or rolls it back. No gradient runs.

The defect is pre-registered. One MLP down-projection column is multiplied by 4, 8, or 16. The column has to change the greedy next token on some prompts and leave others alone. The editor can see the prompts that changed. The gate uses a held-out half of those prompts, plus the prompts that stayed the same. Every one of those must match the token from before the plant.

## Setup

```bash
uv sync --python /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13
```

## Harness check

This uses a tiny map, with no download. The norm oracle must undo the planted column. A random editor with the same budget of three proposals must accept at most 6 of 40 seeds.

```bash
.venv/bin/python -m selfedit selftest
```

## M3 on a small model

Default checkpoint is `Qwen/Qwen2.5-0.5B-Instruct`, which fits on this Mac. The same command takes any causal LM that has `mlp.down_proj.weight`.

```bash
.venv/bin/python -m selfedit m3 --rounds 1 --conditions live,oracle,random,blind,no_edit
.venv/bin/python -m selfedit m3
```

Each round plants a new column. If the live editor's proposal passes, that repaired checkpoint is the next round's model. Otherwise the round rolls back to the previous checkpoint. `frozen` generates with the original weights while the tools read the planted ones.

Reports go to `runs/m3-<timestamp>/report.json`. The decision at the end of that file is the M3 bar: the live model accepts and localizes more rounds than both random and weight-blind. A tie with the norm oracle means the repair followed the loudest column norm.

A ~30B checkpoint, and Kimi K3.1 Pro after its files and license are confirmed, are the same command with `--model`. They are not the first run.
