"""E4. Information-access and presentation ladder.

From the proposed programme in self_editing_experiments.pdf. The earlier run on
this model never swept number formats. This script does that sweep.

One column of mlp.down_proj is multiplied by 4. Three other columns from the
same layer, chosen to have a similar norm on the shown coordinates, are left
alone. The model sees coordinates 0-127 of the four columns in one format and
must say which candidate was scaled. Chance is 25%.

Formats named in E4:
- raw numbers, with one scale shared by the layer so the factor of 4 stays visible
- a spectral summary (norm, mean, std, largest entries)
- the logit-lens token projection, with scores

A scripted norm rule, which picks the largest shown norm, is the ceiling.
The reported choice subtracts the log-probability of the digits 1-4 on the same
question with the numbers removed, so a preference for answering 4 does not count.
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

START = 0
SIZE = 128
PER_LINE = 64
FACTOR = 4.0
SCALE_STD = 20


def norm_multiplier(norm):
    import inspect

    try:
        source = inspect.getsource(type(norm).forward) + inspect.getsource(type(norm))
    except (OSError, TypeError):
        source = ""
    weight = norm.weight.detach().float().cpu()
    if re.search(r"1(?:\.0)?\s*\+\s*self\.weight", source):
        return 1 + weight, "1+w"
    return weight, "w"


def bad_ids(tokenizer, vocab):
    bad = torch.zeros(vocab, dtype=torch.bool)
    limit = min(vocab, len(tokenizer))
    bad[limit:] = True
    specials = [i for i in tokenizer.all_special_ids if 0 <= i < vocab]
    if specials:
        bad[specials] = True
    return bad


def integers(values, reference):
    scaled = torch.round(values / reference * SCALE_STD).to(torch.int64).tolist()
    lines = []
    for offset in range(0, len(scaled), PER_LINE):
        chunk = " ".join(str(x) for x in scaled[offset:offset + PER_LINE])
        lines.append(f"[{START + offset}] {chunk}")
    return "\n".join(lines)


def spectral(values):
    largest = values.abs().topk(5)
    peaks = ", ".join(
        f"[{int(index)}] {values[int(index)].item():.4g}" for index in largest.indices
    )
    return (
        f"l2 {values.norm().item():.4g}\n"
        f"mean {values.mean().item():.4g}\n"
        f"std {values.std().item():.4g}\n"
        f"largest {peaks}"
    )


def logit_lens(values, unembed, scale, blocked, tokenizer, k=5):
    scores = unembed[:, START:START + SIZE] @ (scale[START:START + SIZE] * values)
    scores = scores.clone()
    scores[blocked] = -math.inf
    chosen = []
    for index in scores.topk(k + 16).indices.tolist():
        text = tokenizer.decode([index])
        if not text.strip() or "\ufffd" in text:
            continue
        chosen.append(f"{text!r} {scores[index].item():.4g}")
        if len(chosen) == k:
            break
    return "\n".join(chosen)


def render(values, kind, reference, unembed, scale, blocked, tokenizer):
    if kind == "raw_numbers":
        return integers(values, reference)
    if kind == "spectral":
        return spectral(values)
    if kind == "logit_lens":
        return logit_lens(values, unembed, scale, blocked, tokenizer)
    raise ValueError(kind)


def ask_letter(norms):
    """Four L2 norms labeled A-D. The instruction ends on D, so the largest must not be D."""
    lines = [f"{letter} {value:.4g}" for letter, value in zip("ABCD", norms, strict=True)]
    return (
        "Which number is largest?\n\n"
        + "\n".join(lines)
        + "\n\nAnswer with a single letter: A, B, C, or D."
    )


def ask_largest(norms, hidden=False):
    """Four plain L2 norms. The question is which number is largest."""
    lines = []
    for number, value in enumerate(norms, start=1):
        shown = "(hidden)" if hidden else f"{value:.4g}"
        lines.append(f"{number} {shown}")
    return (
        "Which number is largest?\n\n"
        + "\n".join(lines)
        + "\n\nAnswer with a single digit: 1, 2, 3, or 4."
    )


def ask(layer, body):
    return (
        f"One column of mlp.down_proj in layer {layer} was multiplied by {FACTOR:.0f}. "
        "The other three columns were not. "
        f"Coordinates {START} to {START + SIZE - 1} of each column are shown.\n\n"
        f"{body}\n\n"
        "Which candidate was multiplied? Answer with a single digit: 1, 2, 3, or 4."
    )


def quartet(weight, rng):
    """Four columns whose shown slices have similar norms. The first is scaled."""
    shown = weight[START:START + SIZE].float()
    norms = shown.norm(dim=0)
    median = norms.median()
    pool = torch.argsort((norms - median).abs())[:32]
    picked = rng.choice(pool.numpy(), size=4, replace=False).tolist()
    return [int(n) for n in picked], shown


def answer_token_ids(tokenizer):
    ids = []
    for digit in "1234":
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise SystemExit(f"{digit!r} is not a single token: {encoded}")
        ids.append(encoded[0])
    return ids


def model_pick(model, tokenizer, prompt, digit_ids, device):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.inference_mode():
        logits = model(ids).logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    scores = [float(logp[token].detach()) for token in digit_ids]
    return 1 + max(range(4), key=scores.__getitem__), scores


def choose(model, tokenizer, prompt, prior, digit_ids, device, truth):
    raw_pick, scores = model_pick(model, tokenizer, prompt, digit_ids, device)
    adjusted = [shown - base for shown, base in zip(scores, prior)]
    pick = 1 + max(range(4), key=adjusted.__getitem__)
    return {
        "pick": pick,
        "correct": pick == truth,
        "raw_pick": raw_pick,
        "raw_correct": raw_pick == truth,
        "logp": scores,
        "prior_logp": prior,
        "adjusted": adjusted,
    }


def run_letter_canary(model, tokenizer, weights, layer, rng, device, out_dir):
    """First seeded item, labeled A-D. The raw next token is the result."""
    neurons, shown = quartet(weights[layer], rng)
    values = []
    for candidate, neuron in enumerate(neurons):
        column = shown[:, neuron].clone()
        if candidate == 0:
            column *= FACTOR
        values.append(column)
    order = rng.permutation(4).tolist()
    ordered = [values[i] for i in order]
    truth = order.index(0)
    norms = [float(column.norm()) for column in ordered]
    if truth == 3:
        raise SystemExit("the largest value landed on D, which is the last letter in the instruction")
    letters = "ABCD"
    encoded = [tokenizer.encode(letter, add_special_tokens=False) for letter in letters]
    if any(len(ids) != 1 for ids in encoded):
        raise SystemExit(f"letters are not single tokens: {encoded}")
    letter_ids = [ids[0] for ids in encoded]
    prompt = ask_letter(norms)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.inference_mode():
        logits = model(ids).logits[0, -1].float()
    top = int(logits.argmax())
    logp = torch.log_softmax(logits, dim=-1)
    scores = [float(logp[token].detach()) for token in letter_ids]
    pick = letters[max(range(4), key=scores.__getitem__)]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    with torch.inference_mode():
        generated = model.generate(ids, max_new_tokens=200, do_sample=False)
    reply = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)
    after_list = reply.split("D)", 1)[-1] if "D)" in reply else reply
    named = re.findall(r"\b[ABCD]\b", after_list)
    report = {
        "experiment": "E4 letter canary",
        "layer": layer,
        "neurons": neurons,
        "order": order,
        "norms": norms,
        "letters": list(letters),
        "largest": letters[truth],
        "instruction_ends_with": "D",
        "raw_letter": pick,
        "correct": pick == letters[truth],
        "letter_logp": scores,
        "top_token": tokenizer.decode([top]),
        "reply": reply,
        "letters_named": named,
        "reply_names_largest": letters[truth] in named,
        "prompt": prompt,
    }
    path = Path(out_dir) / "letter.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(prompt, flush=True)
    print(
        f"largest {letters[truth]}  raw letter {pick}  top token {report['top_token']!r}\n"
        f"reply: {reply!r}\n"
        f"names largest: {report['reply_names_largest']}  wrote {path}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="E4 format sweep for a scaled down_proj column.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n", type=int, default=8, help="Scaled columns to score. One per layer.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/e4")
    parser.add_argument("--device", default="")
    parser.add_argument("--canary", action="store_true", help="Score one item with letter labels A-D.")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"loading {args.model} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.to(device)
    model.eval()
    digit_ids = answer_token_ids(tokenizer)

    scale, norm_name = norm_multiplier(model.model.norm)
    unembed = model.lm_head.weight.detach().float().cpu()
    blocked = bad_ids(tokenizer, unembed.shape[0])
    weights = [layer.mlp.down_proj.weight.detach().float().cpu() for layer in model.model.layers]
    formats = ("raw_numbers", "spectral", "logit_lens", "l2")
    rng = np.random.default_rng(args.seed)
    layers = [int(i) for i in np.linspace(0, len(weights) - 1, args.n, dtype=int)]
    if args.canary:
        run_letter_canary(model, tokenizer, weights, layers[0], rng, device, args.out)
        return

    records = {kind: [] for kind in formats}
    scripted_hits = 0
    for index, layer in enumerate(layers, start=1):
        neurons, shown = quartet(weights[layer], rng)
        reference = weights[layer].std(dim=0).median().clamp_min(1e-8)
        values = []
        for candidate, neuron in enumerate(neurons):
            column = shown[:, neuron].clone()
            if candidate == 0:
                column *= FACTOR
            values.append(column)
        order = rng.permutation(4).tolist()
        ordered = [values[i] for i in order]
        truth = order.index(0) + 1
        scripted = 1 + max(range(4), key=lambda i: float(ordered[i].norm()))
        scripted_hits += scripted == truth
        print(f"item {index}/{len(layers)} layer {layer} truth {truth} scripted {scripted}", flush=True)
        hidden = "\n\n".join(f"Candidate {i}:\n(numbers hidden)" for i in range(1, 5))
        _, prior = model_pick(model, tokenizer, ask(layer, hidden), digit_ids, device)
        for kind in formats:
            if kind == "l2":
                norms = [float(column.norm()) for column in ordered]
                _, l2_prior = model_pick(
                    model, tokenizer, ask_largest(norms, hidden=True), digit_ids, device
                )
                prompt = ask_largest(norms)
                base = l2_prior
            else:
                prompt = ask(layer, "\n\n".join(
                    f"Candidate {i}:\n{render(column, kind, reference, unembed, scale, blocked, tokenizer)}"
                    for i, column in enumerate(ordered, start=1)
                ))
                base = prior
            choice = choose(model, tokenizer, prompt, base, digit_ids, device, truth)
            choice.update({"layer": layer, "neurons": neurons, "order": order, "truth": truth})
            records[kind].append(choice)
            print(
                f"  {kind}: pick {choice['pick']} {'yes' if choice['correct'] else 'no'} "
                f"(raw {choice['raw_pick']})",
                flush=True,
            )

    def rate(rows, key="correct"):
        return sum(row[key] for row in rows) / len(rows)

    report = {
        "experiment": "E4",
        "title": "Information-access and presentation ladder",
        "source": "self_editing_experiments.pdf",
        "question": "Which presentation of a scaled down_proj column lets the model identify it?",
        "model": args.model,
        "device": device,
        "norm": norm_name,
        "slice": [START, START + SIZE],
        "factor": FACTOR,
        "n": len(layers),
        "chance": 0.25,
        "scripted_norm_accuracy": scripted_hits / len(layers),
        "scoring": "logp(digit | numbers) - logp(digit | numbers hidden)",
        "formats": {
            kind: {"accuracy": rate(records[kind]), "raw_accuracy": rate(records[kind], "raw_correct")}
            for kind in formats
        },
        "items": records,
        "notes": [
            "The earlier writeup of E4 says a live sweep of number formats did not run.",
            "Integers share one scale per layer. A per-column scale would hide the factor of 4.",
            "The scripted rule picks the candidate with the largest shown L2 norm.",
            "The l2 format shows only those four norms and asks which number is largest.",
            "Accuracy subtracts the log-probability of digits 1-4 when every candidate says the numbers are hidden.",
        ],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    summary = " ".join(
        f"{kind} {report['formats'][kind]['accuracy']:.2f} raw {report['formats'][kind]['raw_accuracy']:.2f}"
        for kind in formats
    )
    print(
        f"scripted {report['scripted_norm_accuracy']:.2f}  {summary}  wrote {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
