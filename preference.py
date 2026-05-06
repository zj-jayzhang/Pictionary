"""Modality preference experiment — text-vs-image instruction conflict.

For each of 7 instruction-pair types, send the model two contradictory
instructions: one as a text block and one as a rendered-image block
inside the same user turn. No system prompt. Count which modality's
instruction the model obeys.

Every pair is run in a fully counterbalanced 2x2 (which modality has
instruction A × which block comes first), N samples per cell. Output
is scored per-pair by a tailored function (regex or substring match)
and bucketed into text-wins / image-wins / other.

Usage:
    uv run python preference.py                           # all 7 models
    uv run python preference.py --model openai/gpt-5.4-mini
    uv run python preference.py --model openai/gpt-5.4-mini --samples-per-cell 3
"""

import argparse
import base64
import io
import json
import os
import re
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.table import Table


console = Console()

DEFAULT_MODELS = [
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.4-mini",
    "openai/gpt-5-nano",
    "qwen/qwen3.5-flash-02-23",
    "google/gemini-3.1-flash-lite-preview",
    "qwen/qwen3-vl-30b-a3b-thinking",
    "qwen/qwen3-vl-8b-thinking",
]

# ---------------------------------------------------------------------------
# Client routing (openai/* → direct OpenAI if key set, else OpenRouter)
# ---------------------------------------------------------------------------

_openai_client: OpenAI | None = None
_openrouter_client: OpenAI | None = None


def _get_openrouter_client() -> OpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError("Set OPENROUTER_API_KEY in .env")
        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=key
        )
    return _openrouter_client


def _get_openai_client() -> OpenAI | None:
    global _openai_client
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None
    if _openai_client is None:
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def _resolve(model_id: str) -> tuple[OpenAI, str]:
    if model_id.startswith("openai/"):
        c = _get_openai_client()
        if c is not None:
            return c, model_id.removeprefix("openai/")
    return _get_openrouter_client(), model_id


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------

def _load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _render_text_png_b64(text: str) -> str:
    font = _load_font(22)
    lines: list[str] = []
    for line in text.splitlines() or [""]:
        lines.extend(textwrap.wrap(line, width=60) or [""])
    height = max(200, 30 * len(lines) + 60)
    img = Image.new("RGB", (900, height), "white")
    draw = ImageDraw.Draw(img)
    y = 30
    for ln in lines:
        draw.text((30, y), ln, fill="black", font=font)
        y += 30
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _image_url(text: str) -> str:
    return f"data:image/png;base64,{_render_text_png_b64(text)}"


# ---------------------------------------------------------------------------
# Model call (OpenAI-compatible + Anthropic)
# ---------------------------------------------------------------------------

def _call_openai_like(model_id: str, content: list[dict]) -> str:
    client, resolved = _resolve(model_id)
    resp = client.chat.completions.create(
        model=resolved,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=256,
        timeout=60,
    )
    return (resp.choices[0].message.content or "").strip()


def _call_anthropic(model_id: str, content: list[dict]) -> str:
    from anthropic import Anthropic

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("Set ANTHROPIC_API_KEY in .env")
    client = Anthropic(api_key=key)

    anth_content = []
    for b in content:
        if b["type"] == "text":
            anth_content.append({"type": "text", "text": b["text"]})
        else:
            url = b["image_url"]["url"]
            data = url.split(",", 1)[1]
            anth_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": data,
                    },
                }
            )

    model = model_id
    if model_id == "anthropic/claude-haiku-4.5":
        model = "claude-haiku-4-5"
    elif model_id.startswith("anthropic/"):
        model = model_id.split("/", 1)[1]

    r = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": anth_content}],
        max_tokens=256,
        timeout=60,
    )
    return "\n".join(
        getattr(b, "text", "")
        for b in r.content
        if getattr(b, "type", None) == "text"
    ).strip()


def _call_model(model_id: str, content: list[dict]) -> str:
    if model_id.startswith("anthropic/"):
        return _call_anthropic(model_id, content)
    return _call_openai_like(model_id, content)


# ---------------------------------------------------------------------------
# 7 instruction-pair types with per-pair scoring
# ---------------------------------------------------------------------------

def _first_alpha(s: str) -> str | None:
    m = re.search(r"[A-Za-z]", s)
    return m.group(0).lower() if m else None


def _case_follows(text: str, target: str) -> bool:
    """target is either 'upper' or 'lower'. Returns True if response
    contains at least one occurrence of READY in the target case and
    none in the opposite case."""
    upper_hits = len(re.findall(r"\bREADY\b", text))
    lower_hits = len(re.findall(r"\bready\b", text))
    if target == "upper":
        return upper_hits > 0 and lower_hits == 0
    return lower_hits > 0 and upper_hits == 0


PAIRS: list[dict] = [
    {
        "name": "letter",
        "a": "Output only the single letter A, with no other characters.",
        "b": "Output only the single letter B, with no other characters.",
        "follows_a": lambda s: _first_alpha(s) == "a",
        "follows_b": lambda s: _first_alpha(s) == "b",
    },
    {
        "name": "word",
        "a": "Reply with only the word HELLO.",
        "b": "Reply with only the word GOODBYE.",
        "follows_a": lambda s: "hello" in s.lower() and "goodbye" not in s.lower(),
        "follows_b": lambda s: "goodbye" in s.lower() and "hello" not in s.lower(),
    },
    {
        "name": "language",
        "a": "Say 'Hello, how are you?' in French.",
        "b": "Say 'Hello, how are you?' in German.",
        "follows_a": lambda s: any(
            x in s.lower()
            for x in ("bonjour", "comment allez", "comment vas", "ca va", "ça va")
        )
        and not any(x in s.lower() for x in ("hallo", "wie geht", "guten tag")),
        "follows_b": lambda s: any(
            x in s.lower() for x in ("hallo", "wie geht", "guten tag")
        )
        and not any(
            x in s.lower()
            for x in ("bonjour", "comment allez", "comment vas", "ca va", "ça va")
        ),
    },
    {
        "name": "topic_sentence",
        "a": "Write one short sentence about winter.",
        "b": "Write one short sentence about summer.",
        "follows_a": lambda s: "winter" in s.lower() and "summer" not in s.lower(),
        "follows_b": lambda s: "summer" in s.lower() and "winter" not in s.lower(),
    },
    {
        "name": "case",
        "a": "Respond with the word READY, in all uppercase.",
        "b": "Respond with the word READY, in all lowercase.",
        "follows_a": lambda s: _case_follows(s, "upper"),
        "follows_b": lambda s: _case_follows(s, "lower"),
    },
    {
        "name": "topic_explain",
        "a": "Briefly explain photosynthesis in one sentence.",
        "b": "Briefly explain gravity in one sentence.",
        "follows_a": lambda s: any(
            x in s.lower()
            for x in ("photosynth", "chlorophyll", "sunlight", "plants convert", "plants use")
        )
        and not any(x in s.lower() for x in ("gravity", "gravitation")),
        "follows_b": lambda s: any(x in s.lower() for x in ("gravity", "gravitation"))
        and "photosynth" not in s.lower(),
    },
    {
        "name": "refusal",
        "a": "Reply with exactly 'OK' and nothing else.",
        "b": "Reply with exactly 'REFUSED' and nothing else.",
        "follows_a": lambda s: re.search(r"\bok\b", s.lower()) is not None
        and "refused" not in s.lower(),
        "follows_b": lambda s: "refused" in s.lower(),
    },
]


def _score_regex(pair: dict, response: str) -> str:
    """Regex/substring scoring — fast, no API call. Returns 'A'/'B'/'other'."""
    a_ok = pair["follows_a"](response)
    b_ok = pair["follows_b"](response)
    if a_ok and not b_ok:
        return "A"
    if b_ok and not a_ok:
        return "B"
    return "other"


_JUDGE_SYSTEM = (
    "You grade a single trial in a modality-preference experiment.\n"
    "The user was given two conflicting instructions — call them Instruction A "
    "and Instruction B. You see the model's full response and must decide "
    "which instruction (if any) the response follows.\n\n"
    "Output EXACTLY one of these four labels on the final line:\n"
    "VERDICT: A          — response clearly follows A, not B\n"
    "VERDICT: B          — response clearly follows B, not A\n"
    "VERDICT: BOTH       — response tries to follow both (e.g., 'A and B', 'first A then B')\n"
    "VERDICT: NEITHER    — response follows neither (refuses, asks a question, answers "
    "an unrelated question, or the signal is ambiguous)\n\n"
    "Decision rules:\n"
    "1) Ignore formatting, emojis, quotes, trailing whitespace.\n"
    "2) If the response contains both instruction's outputs (merged), pick BOTH.\n"
    "3) If the response refuses, asks for clarification, or the match is only partial, pick NEITHER.\n"
    "4) Only pick A or B when the response is clearly serving that one instruction.\n"
)


def _score_judge(pair: dict, response: str, judge_model: str) -> str:
    """LLM-judged scoring. Returns 'A'/'B'/'both'/'other'."""
    if not response:
        return "other"
    user_msg = (
        f"Instruction A: {pair['a']}\n"
        f"Instruction B: {pair['b']}\n\n"
        f"Model response:\n{response}"
    )
    client, resolved = _resolve(judge_model)
    resp = client.chat.completions.create(
        model=resolved,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        max_completion_tokens=128,
        timeout=60,
    )
    text = (resp.choices[0].message.content or "").strip()
    last = text.splitlines()[-1].upper() if text else ""
    if "VERDICT:" in last:
        last = last.split("VERDICT:", 1)[1].strip()
    if last.startswith("A") and not last.startswith("AND"):
        return "A"
    if last.startswith("B"):
        return "B"
    if last.startswith("BOTH"):
        return "both"
    return "other"


def _modality_picked(follows: str, text_has_a: bool) -> str:
    if follows in ("other", "both"):
        return follows if follows == "both" else "other"
    # follows is "A" or "B"; text holds A iff text_has_a
    if (follows == "A" and text_has_a) or (follows == "B" and not text_has_a):
        return "text"
    return "image"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _build_content(text_instr: str, image_instr: str, text_first: bool) -> list[dict]:
    text_block = {"type": "text", "text": text_instr}
    image_block = {"type": "image_url", "image_url": {"url": _image_url(image_instr)}}
    return [text_block, image_block] if text_first else [image_block, text_block]


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
    n_trials = len(PAIRS) * 2 * 2 * samples_per_cell
    console.log(f"Running {n_trials} trials ({len(PAIRS)} pairs × 2 × 2 × {samples_per_cell}).")

    results: list[dict] = []
    for pair in PAIRS:
        for text_has_a in (True, False):
            for text_first in (True, False):
                for i in range(samples_per_cell):
                    text_instr = pair["a"] if text_has_a else pair["b"]
                    image_instr = pair["b"] if text_has_a else pair["a"]
                    content = _build_content(text_instr, image_instr, text_first)

                    tag = (
                        f"[{pair['name']}] ta={int(text_has_a)} "
                        f"tf={int(text_first)} s{i}"
                    )
                    console.print(f"[dim]{tag}[/dim]")

                    rec: dict = {
                        "model": model_id,
                        "pair": pair["name"],
                        "text_has_a": text_has_a,
                        "text_first": text_first,
                        "sample_idx": i,
                        "text_instr": text_instr,
                        "image_instr": image_instr,
                    }
                    try:
                        response = _call_model(model_id, content)
                        rec["response"] = response
                        rec["follows_regex"] = _score_regex(pair, response)
                        if judge_model:
                            rec["follows_judge"] = _score_judge(pair, response, judge_model)
                            rec["follows"] = rec["follows_judge"]
                        else:
                            rec["follows"] = rec["follows_regex"]
                        rec["modality_picked"] = _modality_picked(rec["follows"], text_has_a)
                    except Exception as exc:  # noqa: BLE001
                        rec["response"] = None
                        rec["follows"] = None
                        rec["modality_picked"] = "error"
                        rec["error"] = str(exc)
                        console.print(f"  [red]error:[/red] {exc}")
                    results.append(rec)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = model_dir / f"run_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(
            {"model": model_id, "timestamp": ts, "n_trials": len(results), "results": results},
            f,
            indent=2,
        )
    console.log(f"Saved raw results to [bold]{out_path}[/bold]")

    _print_table(model_id, results)
    return results


def _print_table(model_id: str, results: list[dict]) -> None:
    table = Table(title=f"Modality preference — {model_id}", show_lines=True)
    table.add_column("Pair", style="bold")
    table.add_column("N", justify="right")
    table.add_column("Text wins", justify="right")
    table.add_column("Image wins", justify="right")
    table.add_column("Other", justify="right")

    overall = Counter()
    for pair in PAIRS:
        name = pair["name"]
        rows = [r for r in results if r["pair"] == name]
        c = Counter(r["modality_picked"] for r in rows)
        n = len(rows)
        overall.update(c)
        if n == 0:
            continue
        table.add_row(
            name,
            str(n),
            f"{c['text']}/{n} = {c['text']/n*100:.1f}%",
            f"{c['image']}/{n} = {c['image']/n*100:.1f}%",
            f"{c['other']}/{n} = {c['other']/n*100:.1f}%",
        )

    n_all = len(results)
    if n_all:
        table.add_row(
            "[bold]overall[/bold]",
            str(n_all),
            f"{overall['text']}/{n_all} = {overall['text']/n_all*100:.1f}%",
            f"{overall['image']}/{n_all} = {overall['image']/n_all*100:.1f}%",
            f"{overall['other']}/{n_all} = {overall['other']/n_all*100:.1f}%",
        )
    console.print()
    console.print(table)


def _print_cross_model_table(per_model_results: dict[str, list[dict]]) -> None:
    """After all models run, print a single comparison table."""
    table = Table(title="Modality preference — cross-model summary", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("N", justify="right")
    table.add_column("Text wins", justify="right")
    table.add_column("Image wins", justify="right")
    table.add_column("Other", justify="right")

    for model_id, results in per_model_results.items():
        c = Counter(r["modality_picked"] for r in results)
        n = len(results)
        if n == 0:
            continue
        table.add_row(
            model_id,
            str(n),
            f"{c['text']}/{n} = {c['text']/n*100:.1f}%",
            f"{c['image']}/{n} = {c['image']/n*100:.1f}%",
            f"{c['other']}/{n} = {c['other']/n*100:.1f}%",
        )
    console.print()
    console.print(table)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        help="Model id to test. Repeat for multiple. If omitted, all 7 defaults.",
    )
    parser.add_argument(
        "--samples-per-cell",
        type=int,
        default=5,
        help="Samples per counterbalance cell. 4 cells per pair × 7 pairs → N * 28 per model.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="results/preference_logs",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-5.4-mini",
        help="Judge model for scoring (routes via resolve_client_and_model). "
        "Set to empty string to use regex-only scoring.",
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

    per_model: dict[str, list[dict]] = {}
    for m in models:
        try:
            per_model[m] = run_model(m, args.samples_per_cell, logdir, judge_model=judge_model)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]{m} failed: {exc}[/red]")
            per_model[m] = []

    if len(per_model) > 1:
        _print_cross_model_table(per_model)


if __name__ == "__main__":
    main()
