"""Text-only control for the modality-preference experiment.

Symmetric companion to `preference_image_only.py`. Instead of sending
two conflicting instructions (one text block, one image block), this
sends ONLY the text instruction — no image block at all — and measures
how often the model follows it.

Together with `preference_image_only.py` this gives the two
single-modality baselines: the conflict run (`preference.py`) measures
*relative* preference; these two measure each modality's *absolute*
follow-rate when nothing competes. Comparing them isolates how much of
the conflict-run text-bias is "text is easier to act on" vs a genuine
tilt.

Design (no system prompt, mirroring preference.py / the image-only
control):
  - For each of the 21 pairs, BOTH instruction A and instruction B are
    sent alone as a single text block (the `instr` axis cancels
    content bias).
  - `samples_per_cell` independent samples per (pair, instr) cell.
  - 21 pairs × 2 instr × S samples = 42·S trials per model
    (S=5 → 210 trials).
  - Scoring: primary is the single-instruction LLM judge (shared with
    the image-only control); per-trial regex match also recorded.
  - Slicing by `sample_idx` gives 5 pseudo-replications → mean ± std,
    exactly as in motivation_exp §3 / §7.

Usage:
    uv run python preference_text_only.py
    uv run python preference_text_only.py --model anthropic/claude-haiku-4.5
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from preference import PAIRS, _call_model
from preference_image_only import DEFAULT_MODELS, _score_single_judge

console = Console()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _text_only_content(instruction: str) -> list[dict]:
    """A user turn with a single text block — no image block at all."""
    return [{"type": "text", "text": instruction}]


def run_model(
    model_id: str,
    samples_per_cell: int,
    logdir: Path,
    judge_model: str | None = None,
) -> list[dict]:
    model_short = model_id.replace("/", "_")
    model_dir = logdir / model_short
    model_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold]{model_id}[/bold]")
    n_trials = len(PAIRS) * 2 * samples_per_cell
    console.log(
        f"Running {n_trials} text-only trials "
        f"({len(PAIRS)} pairs × 2 instr × {samples_per_cell})."
    )

    results: list[dict] = []
    for pair in PAIRS:
        for instr_label in ("a", "b"):
            instruction = pair[instr_label]
            follows_fn = pair["follows_a"] if instr_label == "a" else pair["follows_b"]
            for i in range(samples_per_cell):
                content = _text_only_content(instruction)
                tag = f"[{pair['name']}] instr={instr_label} s{i}"
                console.print(f"[dim]{tag}[/dim]")

                rec: dict = {
                    "model": model_id,
                    "pair": pair["name"],
                    "instr_label": instr_label,
                    "sample_idx": i,
                    "instruction": instruction,
                }
                try:
                    response = _call_model(model_id, content)
                    rec["response"] = response
                    rec["follows_regex"] = "follows" if follows_fn(response) else "not"
                    if judge_model:
                        rec["follows_judge"] = _score_single_judge(
                            instruction, response, judge_model
                        )
                        rec["follows"] = rec["follows_judge"]
                    else:
                        rec["follows"] = rec["follows_regex"]
                    rec["outcome"] = "text" if rec["follows"] == "follows" else "other"
                except Exception as exc:  # noqa: BLE001
                    rec["response"] = None
                    rec["follows"] = None
                    rec["outcome"] = "error"
                    rec["error"] = str(exc)
                    console.print(f"  [red]error:[/red] {exc}")
                results.append(rec)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = model_dir / f"run_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": model_id,
                "timestamp": ts,
                "mode": "text_only",
                "n_trials": len(results),
                "results": results,
            },
            f,
            indent=2,
        )
    console.log(f"Saved raw results to [bold]{out_path}[/bold]")

    _print_table(model_id, results, samples_per_cell)
    return results


def _print_table(model_id: str, results: list[dict], samples_per_cell: int) -> None:
    """Per-model summary: follow-rate overall, and mean±std over sample slices."""
    table = Table(title=f"Text-only follow-rate — {model_id}", show_lines=True)
    table.add_column("Sample slice", style="bold")
    table.add_column("Follows text", justify="right")
    table.add_column("Other", justify="right")
    table.add_column("Error", justify="right")
    table.add_column("N", justify="right")

    slice_rates: list[float] = []
    for s in range(samples_per_cell):
        rows = [r for r in results if r["sample_idx"] == s]
        c = Counter(r["outcome"] for r in rows)
        n_ok = c["text"] + c["other"]
        rate = c["text"] / n_ok * 100 if n_ok else float("nan")
        if n_ok:
            slice_rates.append(rate)
        table.add_row(
            f"s{s}",
            f"{c['text']}/{n_ok} = {rate:.1f}%" if n_ok else "—",
            str(c["other"]),
            str(c["error"]),
            str(n_ok),
        )

    if slice_rates:
        mean = sum(slice_rates) / len(slice_rates)
        var = sum((x - mean) ** 2 for x in slice_rates) / max(1, len(slice_rates) - 1)
        std = var ** 0.5
        table.add_row(
            "[bold]mean ± std[/bold]",
            f"[bold]{mean:.1f}% ± {std:.1f}%[/bold]",
            "", "", "",
        )
    console.print()
    console.print(table)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        help="Model id to test. Repeat for multiple. If omitted, the 6 defaults.",
    )
    parser.add_argument("--samples-per-cell", type=int, default=5)
    parser.add_argument("--logdir", type=str, default="results/preference_text_only_logs")
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-5.4-mini",
        help="Judge model. Empty string → regex-only scoring.",
    )
    args = parser.parse_args()
    judge_model = args.judge_model or None

    models = args.model if args.model else DEFAULT_MODELS
    logdir = Path(args.logdir)
    console.log(
        f"Models ({len(models)}): {', '.join(models)} | "
        f"samples_per_cell={args.samples_per_cell} | "
        f"judge={judge_model or 'regex-only'}"
    )

    for m in models:
        try:
            run_model(m, args.samples_per_cell, logdir, judge_model=judge_model)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]{m} failed: {exc}[/red]")


if __name__ == "__main__":
    main()
