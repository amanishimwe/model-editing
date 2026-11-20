# E4. Presentation ladder

From the proposed programme in `self_editing_experiments.pdf`. One column of `mlp.down_proj` is multiplied by 4. Three similar columns are left alone. The model has to say which candidate was scaled. Chance is 25%.

The same four columns are shown in the three formats E4 names:

- raw numbers
- a spectral summary
- the logit-lens token projection
- four plain L2 norms, with the question "which number is largest?"

A scripted rule that picks the largest shown norm is the ceiling. The score for each digit subtracts its log-probability on the same question with the numbers removed, so a preference for answering 4 does not count. The earlier run of E4 did not sweep these formats.

## Setup

```bash
uv pip install --python .venv/bin/python torch transformers numpy
```

## Run

```bash
.venv/bin/python src/reader.py --n 8
.venv/bin/python src/reader.py --canary
```

`--n` is how many scaled columns to score, one per layer. The report is written to `runs/e4/report.json`.
