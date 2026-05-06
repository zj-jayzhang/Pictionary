"""No-injection baseline: measure UT for text vs image on clean documents.

Sends the document (no injection embedded) through both text and image
conditions, grades UT via the same LLM judge used in the injection
experiments. This isolates any UT cost from the modality shift itself,
independent of attack presence.

Usage:
    uv run python baseline_ut.py                           # all 7 models, 5 samples
    uv run python baseline_ut.py --model openai/gpt-5.4-mini --samples 10
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from helpers.runner import (
    create_content_image,
    image_data_url,
    resolve_client_and_model,
)
from injection.data import (
    INJECTION_USER_TASKS,
    grade_user_task,
)

console = Console()

DEFAULT_MODELS = [
    "openai/gpt-5.4-mini",
    "qwen/qwen3.5-flash-02-23",
    "google/gemini-3.1-flash-lite-preview",
]

SYSTEM_PROMPT = "You are a helpful assistant."


def _call_openai_like(model_id, content):
    client, resolved = resolve_client_and_model(model_id)
    resp = client.chat.completions.create(
        model=resolved,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_completion_tokens=1024,
        timeout=60,
    )
    return (resp.choices[0].message.content or "").strip()


def _call_anthropic(model_id, content):
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("Set ANTHROPIC_API_KEY")
    client = Anthropic(api_key=key)
    anth_content = []
    for b in content:
        if b["type"] == "text":
            anth_content.append({"type": "text", "text": b["text"]})
        else:
            url = b["image_url"]["url"]
            data = url.split(",", 1)[1]
            anth_content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": data},
            })
    model = model_id
    if model_id == "anthropic/claude-haiku-4.5":
        model = "claude-haiku-4-5"
    elif model_id.startswith("anthropic/"):
        model = model_id.split("/", 1)[1]
    r = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": anth_content}],
        max_tokens=1024,
        timeout=60,
    )
    return "\n".join(
        getattr(b, "text", "") for b in r.content
        if getattr(b, "type", None) == "text"
    ).strip()


def call_model(model_id, content):
    if model_id.startswith("anthropic/"):
        return _call_anthropic(model_id, content)
    return _call_openai_like(model_id, content)


def build_content(task_prompt, document, condition):
    if condition == "text":
        return [
            {"type": "text", "text": f"{task_prompt}\n\n{document}"},
        ]
    else:
        img = create_content_image(document)
        return [
            {"type": "text", "text": task_prompt},
            {"type": "image_url", "image_url": {"url": image_data_url(img)}},
        ]


def run_baseline(model_id, n_samples, logdir):
    model_short = model_id.replace("/", "_")
    console.rule(f"[bold]{model_id} — no-injection baseline[/bold]")

    results = {"text": [], "image": []}

    for task_type, task_prompt, document in INJECTION_USER_TASKS:
        for condition in ("text", "image"):
            for i in range(n_samples):
                tag = f"[{task_type}][{condition}][s{i}]"
                console.print(f"[dim]{tag}[/dim]")
                try:
                    content = build_content(task_prompt, document, condition)
                    response = call_model(model_id, content)
                    ut = grade_user_task(task_type, response)
                    results[condition].append({"ut": ut, "response": response})
                except Exception as e:
                    console.print(f"  [red]error:[/red] {e}")
                    results[condition].append({"ut": None, "error": str(e)})

    # Summary
    table = Table(title=f"Baseline UT (no injection) — {model_id}", show_lines=True)
    table.add_column("Condition", style="bold")
    table.add_column("UT", justify="right")
    table.add_column("N", justify="right")

    for cond in ("text", "image"):
        valid = [r for r in results[cond] if r.get("ut") is not None]
        n = len(valid)
        passed = sum(1 for r in valid if r["ut"])
        table.add_row(
            cond,
            f"{passed}/{n} = {passed/n*100:.1f}%" if n else "N/A",
            str(n),
        )
    console.print()
    console.print(table)

    # Save
    out_dir = Path(logdir) / model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"baseline_ut_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model_id,
            "timestamp": ts,
            "n_samples": n_samples,
            "results": {
                cond: [{"ut": r.get("ut"), "error": r.get("error")}
                       for r in results[cond]]
                for cond in ("text", "image")
            },
        }, f, indent=2)
    console.log(f"Saved to [bold]{out_path}[/bold]")
    return results


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--logdir", default="results/baseline_ut_logs")
    args = parser.parse_args()

    models = args.model if args.model else DEFAULT_MODELS

    all_results = {}
    for m in models:
        try:
            all_results[m] = run_baseline(m, args.samples, args.logdir)
        except Exception as e:
            console.print(f"[red]{m} failed: {e}[/red]")

    if len(all_results) > 1:
        table = Table(title="Baseline UT — cross-model", show_lines=True)
        table.add_column("Model", style="bold")
        table.add_column("Text UT", justify="right")
        table.add_column("Image UT", justify="right")
        table.add_column("Δ", justify="right")
        for m, res in all_results.items():
            row = [m]
            for cond in ("text", "image"):
                valid = [r for r in res[cond] if r.get("ut") is not None]
                n = len(valid)
                p = sum(1 for r in valid if r["ut"])
                row.append(f"{p}/{n} = {p/n*100:.1f}%" if n else "N/A")
            # compute delta
            t_valid = [r for r in res["text"] if r.get("ut") is not None]
            i_valid = [r for r in res["image"] if r.get("ut") is not None]
            t_rate = sum(1 for r in t_valid if r["ut"]) / max(len(t_valid), 1)
            i_rate = sum(1 for r in i_valid if r["ut"]) / max(len(i_valid), 1)
            row.append(f"{(i_rate - t_rate)*100:+.1f} pp")
            table.add_row(*row)
        console.print()
        console.print(table)


if __name__ == "__main__":
    main()
