"""No-injection baseline: measure UT for text vs image/audio on clean documents.

Sends the document (no injection embedded) through both conditions, grades UT
via the same LLM judge used in the injection experiments. This isolates any UT
cost from the modality shift itself, independent of attack presence.

Usage:
    uv run python baseline_ut.py                           # all 3 default models, text+image
    uv run python baseline_ut.py --model openai/gpt-5.4-mini --samples 10
    uv run python baseline_ut.py --model openai/gpt-audio-mini --conditions text,audio --samples 14
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
    audio_b64,
    create_content_audio,
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


def _has_audio_input(content):
    return any(isinstance(b, dict) and b.get("type") == "input_audio" for b in content)


def _call_openai_like(model_id, content):
    client, resolved = resolve_client_and_model(model_id)
    kwargs: dict = dict(
        model=resolved,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_completion_tokens=8192,
        timeout=120,
    )
    is_audio_model = "audio" in resolved.lower()
    has_audio_in = _has_audio_input(content)
    if is_audio_model:
        if has_audio_in:
            kwargs["modalities"] = ["text"]
        else:
            kwargs["modalities"] = ["text", "audio"]
            kwargs["audio"] = {"voice": "alloy", "format": "wav"}
    elif has_audio_in:
        kwargs["modalities"] = ["text"]
    resp = client.chat.completions.create(**kwargs)
    message = resp.choices[0].message
    parts = []
    if message.content:
        parts.append(message.content.strip())
    audio_out = getattr(message, "audio", None)
    transcript = getattr(audio_out, "transcript", None) if audio_out else None
    if transcript:
        parts.append(transcript.strip())
    text = "\n".join(parts).strip()
    u = resp.usage
    usage = {
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
    }
    return text, usage


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
        max_tokens=8192,
        timeout=120,
    )
    text = "\n".join(
        getattr(b, "text", "") for b in r.content
        if getattr(b, "type", None) == "text"
    ).strip()
    usage = {
        "prompt_tokens": getattr(r.usage, "input_tokens", None),
        "completion_tokens": getattr(r.usage, "output_tokens", None),
    }
    return text, usage


def _messages_to_dashscope(content):
    """Convert input_audio blocks to data-URI form expected by DashScope."""
    new_content = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "input_audio":
            ia = b.get("input_audio", {})
            fmt = ia.get("format", "wav")
            data = ia.get("data", "")
            if not data.startswith("data:"):
                data = f"data:audio/{fmt};base64,{data}"
            new_content.append({"type": "input_audio", "input_audio": {"data": data, "format": fmt}})
        else:
            new_content.append(b)
    return new_content


def _call_dashscope(model_id, content):
    import time as _time
    client, resolved = resolve_client_and_model(model_id)
    kwargs: dict = dict(
        model=resolved,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _messages_to_dashscope(content)},
        ],
        modalities=["text", "audio"],
        audio={"voice": "Tina", "format": "wav"},
        stream=True,
        stream_options={"include_usage": True},
        timeout=180,
    )
    for attempt in range(5):
        try:
            stream = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "rate" in msg.lower() or "quota" in msg.lower():
                _time.sleep(2 ** attempt + 1)
                continue
            raise
    parts = []
    for chunk in stream:
        if not chunk.choices:
            continue
        text = getattr(chunk.choices[0].delta, "content", None)
        if text:
            parts.append(text)
    return "".join(parts).strip(), {"prompt_tokens": None, "completion_tokens": None}


def call_model(model_id, content):
    """Returns (response_text, usage_dict)."""
    if model_id.startswith("anthropic/"):
        return _call_anthropic(model_id, content)
    if model_id.startswith("dashscope/"):
        return _call_dashscope(model_id, content)
    return _call_openai_like(model_id, content)


def build_content(task_prompt, document, condition):
    if condition == "text":
        return [
            {"type": "text", "text": f"{task_prompt}\n\n{document}"},
        ]
    elif condition == "audio":
        audio_bytes = create_content_audio(document)
        return [
            {"type": "text", "text": task_prompt},
            {"type": "input_audio", "input_audio": {"data": audio_b64(audio_bytes), "format": "wav"}},
        ]
    else:
        img = create_content_image(document)
        return [
            {"type": "text", "text": task_prompt},
            {"type": "image_url", "image_url": {"url": image_data_url(img)}},
        ]


def run_baseline(model_id, n_samples, logdir, conditions=("text", "image")):
    import math

    model_short = model_id.replace("/", "_")
    console.rule(f"[bold]{model_id} — no-injection baseline[/bold]")

    # results[condition][task_type] = list of {"ut": bool|None, ...}
    results: dict[str, dict[str, list]] = {
        c: {task_type: [] for task_type, _, _ in INJECTION_USER_TASKS}
        for c in conditions
    }

    for task_type, task_prompt, document in INJECTION_USER_TASKS:
        for condition in conditions:
            for i in range(n_samples):
                tag = f"[{task_type}][{condition}][s{i}]"
                console.print(f"[dim]{tag}[/dim]")
                try:
                    content = build_content(task_prompt, document, condition)
                    response, usage = call_model(model_id, content)
                    ut = grade_user_task(task_type, response)
                    results[condition][task_type].append({
                        "ut": ut, "response": response,
                        "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"],
                    })
                except Exception as e:
                    console.print(f"  [red]error:[/red] {e}")
                    results[condition][task_type].append({"ut": None, "error": str(e)})

    def _mean_std(bits):
        bits = [b for b in bits if b is not None]
        if not bits:
            return None, None
        m = sum(bits) / len(bits)
        if len(bits) > 1:
            s = math.sqrt(sum((b - m) ** 2 for b in bits) / (len(bits) - 1))
        else:
            s = 0.0
        return m, s

    # Per-task table
    table = Table(title=f"Baseline UT (no injection) — {model_id}", show_lines=True)
    table.add_column("Task", style="bold")
    for cond in conditions:
        table.add_column(f"{cond.capitalize()} UT (mean±std)", justify="right")
    table.add_column("N/task", justify="right")

    task_types = [t for t, _, _ in INJECTION_USER_TASKS]
    for task_type in task_types:
        row = [task_type]
        for cond in conditions:
            bits = [r["ut"] for r in results[cond][task_type] if r.get("ut") is not None]
            m, s = _mean_std(bits)
            if m is None:
                row.append("N/A")
            else:
                row.append(f"{m*100:.1f}% ± {s*100:.1f}%")
        row.append(str(n_samples))
        table.add_row(*row)

    # Overall row
    overall_row = ["**Overall**"]
    for cond in conditions:
        all_bits = [
            r["ut"]
            for task_type in task_types
            for r in results[cond][task_type]
            if r.get("ut") is not None
        ]
        m, s = _mean_std(all_bits)
        if m is None:
            overall_row.append("N/A")
        else:
            n = len(all_bits)
            passed = sum(all_bits)
            overall_row.append(f"{passed}/{n} ({m*100:.1f}% ± {s*100:.1f}%)")
    overall_row.append(str(n_samples * len(task_types)))
    table.add_row(*overall_row)

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
            "conditions": list(conditions),
            "results": {
                cond: {
                    task_type: [
                        {"ut": r.get("ut"), "error": r.get("error"),
                         "prompt_tokens": r.get("prompt_tokens"),
                         "completion_tokens": r.get("completion_tokens")}
                        for r in results[cond][task_type]
                    ]
                    for task_type in task_types
                }
                for cond in conditions
            },
        }, f, indent=2)
    console.log(f"Saved to [bold]{out_path}[/bold]")
    return results


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--logdir", default="exp_runs/baseline_ut_logs")
    parser.add_argument(
        "--conditions", default="text,image",
        help="Comma-separated conditions to run: text, image, audio. Default: text,image",
    )
    args = parser.parse_args()

    models = args.model if args.model else DEFAULT_MODELS
    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())

    all_results = {}
    for m in models:
        try:
            all_results[m] = run_baseline(m, args.samples, args.logdir, conditions)
        except Exception as e:
            console.print(f"[red]{m} failed: {e}[/red]")

    if len(all_results) > 1:
        table = Table(title="Baseline UT — cross-model", show_lines=True)
        table.add_column("Model", style="bold")
        for cond in conditions:
            table.add_column(f"{cond.capitalize()} UT", justify="right")
        if len(conditions) == 2:
            table.add_column("Δ (cond2−cond1)", justify="right")
        for m, res in all_results.items():
            row = [m]
            rates = []
            for cond in conditions:
                all_bits = [
                    r["ut"]
                    for task_records in res.get(cond, {}).values()
                    for r in task_records
                    if r.get("ut") is not None
                ]
                n = len(all_bits)
                p = sum(all_bits)
                row.append(f"{p}/{n} = {p/n*100:.1f}%" if n else "N/A")
                rates.append(p / max(n, 1))
            if len(conditions) == 2:
                row.append(f"{(rates[1] - rates[0])*100:+.1f} pp")
            table.add_row(*row)
        console.print()
        console.print(table)


if __name__ == "__main__":
    main()
