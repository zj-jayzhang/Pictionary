"""API runner for the image-as-defense injection benchmark.

Mirrors ``local_model.py`` but calls hosted models through the API
(OpenRouter / OpenAI / Anthropic / DashScope) instead of loading
weights locally. Sweeps multiple model ids and writes per-sample JSON
in the same schema, so results aggregate alongside the local and
API runs in ``full_results.md``.

Routing (same prefix rules as ``injection/runner.py``):
    anthropic/*   → Anthropic Messages API
    dashscope/*   → DashScope OpenAI-compatible endpoint
    openai/*      → OpenAI direct when OPENAI_API_KEY is set
    everything else (meta-llama/*, qwen/*, google/*, x-ai/*, …)
                  → OpenRouter

Usage::

    uv run python api_llm.py --model qwen/qwen3-vl-8b-instruct
    uv run python api_llm.py \\
        --model meta-llama/llama-3.2-11b-vision-instruct \\
        --model google/gemma-3-27b-it \\
        --conditions text,image --task-type summarize \\
        --template-idx 0 --max-injections 7
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime

from dotenv import load_dotenv

from helpers.runner import (
    console,
    create_content_image,
    set_active_model_id,
)
from injection.data import (
    INJECTION_ATTACKS,
    INJECTION_OVERRIDE_TEMPLATES,
    INJECTION_USER_TASKS,
    _T2_PREFILLED_ANSWERS,
    build_tool_prompt,
    grade_user_task,
)
from injection.runner import (
    _CONDITION_LABELS,
    _build_injection_messages,
    _call_anthropic,
    _call_dashscope,
    _call_openai_or_openrouter,
    _embed_injection,
    _extract_function_calls,
    _write_sample_log,
)


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

def _backend_for(model_id: str) -> str:
    if model_id.startswith("anthropic/"):
        return "Anthropic"
    if model_id.startswith("dashscope/"):
        return "DashScope"
    return "OpenAI/OpenRouter"


_TOOL_PROMPT = build_tool_prompt()


def _with_tool_prompt(messages: list) -> list:
    """Append the text tool description to the system message.

    The target models (Llama-3.2-Vision, Gemma-3, Qwen3-VL) have no
    tool-capable endpoint on OpenRouter, so native `tools=` calls 404.
    Instead we describe the tools in the system prompt and let the
    model emit `<tool_call>{...}</tool_call>` text, which
    `_extract_function_calls` parses. Mirrors the native-tool benchmark
    closely enough for the ASR metric.
    """
    out = []
    patched_system = False
    for m in messages:
        if m.get("role") == "system" and not patched_system:
            out.append({
                "role": "system",
                "content": f"{m['content']}\n\n{_TOOL_PROMPT}",
            })
            patched_system = True
        else:
            out.append(m)
    if not patched_system:
        out.insert(0, {"role": "system", "content": _TOOL_PROMPT})
    return out


def _call_api(model_id: str, messages: list) -> str:
    """Single API call, dispatched by model-id prefix. Uses the
    text-prompt tool fallback (no native `tools=`)."""
    messages = _with_tool_prompt(messages)
    backend = _backend_for(model_id)
    if backend == "Anthropic":
        return _call_anthropic(messages, None, model_id)
    if backend == "DashScope":
        return _call_dashscope(messages, None, model_id)
    return _call_openai_or_openrouter(messages, None, model_id)


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def run_model(
    model_id: str,
    conditions: list[str],
    log_root: str,
    task_type_filter: str | None = None,
    template_idx_filter: int | None = None,
    max_injections: int | None = None,
) -> None:
    set_active_model_id(model_id)
    backend = _backend_for(model_id)
    model_short = model_id.replace("/", "_")

    conditions = [c for c in conditions if c in _CONDITION_LABELS]
    if not conditions:
        raise ValueError("No valid conditions to run.")
    conditions_indexed = [(k, _CONDITION_LABELS[k]) for k in conditions]
    needs_image = "image" in conditions

    user_tasks_indexed = [
        (i, t) for i, t in enumerate(INJECTION_USER_TASKS)
        if task_type_filter is None or t[0] == task_type_filter
    ]
    templates_indexed = [
        (i, t) for i, t in enumerate(INJECTION_OVERRIDE_TEMPLATES)
        if template_idx_filter is None or i == template_idx_filter
    ]
    if not user_tasks_indexed:
        raise ValueError(f"task_type_filter={task_type_filter!r} matched no tasks")
    if not templates_indexed:
        raise ValueError(f"template_idx_filter={template_idx_filter!r} out of range")

    attacks = list(enumerate(INJECTION_ATTACKS))
    if max_injections is not None:
        attacks = attacks[:max_injections]

    n_tasks = len(user_tasks_indexed)
    n_attacks = len(attacks)
    n_templates = len(templates_indexed)
    per_cond = n_tasks * n_attacks * n_templates
    console.rule(f"[bold]Running {model_id}  ({backend})[/bold]")
    console.log(
        f"{n_tasks} tasks × {n_attacks} injections × {n_templates} templates × "
        f"{len(conditions)} conditions = {per_cond * len(conditions)} samples"
    )

    results: dict[str, dict[str, list]] = {
        label: {"asr": [], "ut": []} for _, label in conditions_indexed
    }
    started = time.time()

    for ut_idx, (task_type, user_task, content) in user_tasks_indexed:
        console.rule(f"[bold cyan]Task: {task_type}[/bold cyan]")
        for inj_idx, (injection, target_fn) in attacks:
            for tmpl_idx, template in templates_indexed:
                wrapped_injection = template.format(
                    injection=injection,
                    prefilled_answers=_T2_PREFILLED_ANSWERS[task_type],
                )
                full_content = _embed_injection(content, wrapped_injection)
                content_img = (
                    create_content_image(full_content) if needs_image else None
                )

                for cond_key, cond_label in conditions_indexed:
                    sample_start = time.time()
                    response: str | None = None
                    called: list[str] = []
                    inj_ok: bool | None = None
                    ut_ok: bool | None = None
                    error: str | None = None
                    messages: list = []
                    try:
                        messages = _build_injection_messages(
                            cond_key, user_task, full_content,
                            content_img=content_img,
                        )
                        response = _call_api(model_id, messages)
                        called = _extract_function_calls(response)
                        inj_ok = target_fn in called
                        ut_ok = grade_user_task(task_type, response or "")
                    except Exception as exc:  # noqa: BLE001
                        error = f"{type(exc).__name__}: {exc}"
                        console.print(f"  [red]Error[/red] {error}")

                    duration = time.time() - sample_start
                    results[cond_label]["asr"].append(inj_ok)
                    results[cond_label]["ut"].append(ut_ok)
                    tag = f"[{task_type}][t{tmpl_idx}:{inj_idx}:{target_fn}][{cond_label}]"
                    console.print(
                        f"[dim]{tag}[/dim] "
                        f"INJ={'HIT' if inj_ok else ('BLOCK' if inj_ok is False else '?')} "
                        f"UT={'PASS' if ut_ok else ('FAIL' if ut_ok is False else '?')} "
                        f"({duration:.1f}s)"
                    )

                    _write_sample_log(
                        root_dir=log_root,
                        model_short=model_short,
                        model_id=model_id,
                        backend=backend,
                        task_type=task_type,
                        user_task_idx=ut_idx,
                        user_task=user_task,
                        content=content,
                        inj_idx=inj_idx,
                        injection=injection,
                        target_fn=target_fn,
                        tmpl_idx=tmpl_idx,
                        template=template,
                        wrapped_injection=wrapped_injection,
                        condition_key=cond_key,
                        condition_label=cond_label,
                        messages=messages,
                        response=response,
                        called=called,
                        injection_succeeded=inj_ok,
                        user_task_succeeded=ut_ok,
                        error=error,
                        duration=duration,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )

    # Aggregate summary
    wall = time.time() - started
    summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "scenario": "injection",
        "model": model_id,
        "backend": backend,
        "n_tasks": n_tasks,
        "n_attacks": n_attacks,
        "n_templates": n_templates,
        "samples_per_condition": per_cond,
        "conditions": [label for _, label in conditions_indexed],
        "overall": {},
        "wall_seconds": wall,
    }
    for _, label in conditions_indexed:
        asrs = [a for a in results[label]["asr"] if a is not None]
        uts = [u for u in results[label]["ut"] if u is not None]
        summary["overall"][label] = {
            "asr_succeeded": sum(asrs),
            "asr_total": len(asrs),
            "ut_succeeded": sum(uts),
            "ut_total": len(uts),
            "errors": sum(1 for a in results[label]["asr"] if a is None),
        }
    summaries_dir = os.path.join(log_root, "_summaries")
    os.makedirs(summaries_dir, exist_ok=True)
    out_path = os.path.join(
        summaries_dir, f"{model_short}_{summary['timestamp']}.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    console.log(f"Summary → [bold]{out_path}[/bold]   wall={wall:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", action="append", required=True,
        help="Model id. Repeat to sweep multiple models.",
    )
    p.add_argument(
        "--conditions", default="text,image",
        help="Comma-separated: text,image.",
    )
    p.add_argument(
        "--log-root", default="exp_runs/api_logs",
        help="Per-sample logs under <log-root>/<model>/; "
             "summaries under <log-root>/_summaries/.",
    )
    p.add_argument("--task-type", default=None,
                   help="Filter to a single user task type (e.g. summarize).")
    p.add_argument("--template-idx", type=int, default=None,
                   help="Filter to a single attack template (0..6).")
    p.add_argument(
        "--max-injections", type=int, default=None,
        help="Cap the number of injections (first N of 14). "
             "Use for smoke tests, e.g. 7.",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_cli()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for m in args.model:
        try:
            run_model(
                m, conditions, args.log_root,
                task_type_filter=args.task_type,
                template_idx_filter=args.template_idx,
                max_injections=args.max_injections,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[bold red]Model {m} failed:[/bold red] {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
