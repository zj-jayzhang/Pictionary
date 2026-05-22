"""Entry point for the image-as-defense injection experiment.

Compares text vs image modality for untrusted document content against
prompt-injection attacks. Writes per-sample JSON logs under
``exp_runs/simple_inj_logs/<model>/...`` and an aggregate summary under
``exp_runs/logs/injection/``.
"""

import argparse

from dotenv import load_dotenv

from helpers.runner import (
    DEFAULT_MODEL_ID,
    console,
    get_active_model_id,
    set_active_model_id,
    set_render_style,
)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prompt-injection benchmark. For each user task × injection × "
            "attack template, runs the model under text and image conditions "
            "and records InjectionTaskSuccess + UserTaskSuccess per sample."
        )
    )
    parser.add_argument(
        "--model",
        "--model-id",
        dest="model_id",
        default=DEFAULT_MODEL_ID,
        help=(
            "Model id. `openai/*` routes to OpenAI directly if "
            "OPENAI_API_KEY is set; `anthropic/*` routes to the Anthropic "
            "API; everything else goes through OpenRouter."
        ),
    )
    parser.add_argument(
        "--chat-ui",
        action="store_true",
        default=False,
        help="Render document images in chat-UI style. Alias for "
             "`--render-style chat`; kept for back-compat.",
    )
    parser.add_argument(
        "--render-style",
        dest="render_style",
        default=None,
        choices=["plain", "chat", "google", "blackboard"],
        help=(
            "Image render style for the `image` condition. 'plain' is "
            "the default white-on-black document. 'chat' mimics ChatGPT's "
            "UI. 'google' mimics a Google Search page (centered logo + "
            "long search-box framing the document). 'blackboard' renders "
            "the document as white chalk text on a cartoon green "
            "chalkboard (with a teacher figure pointing at it)."
        ),
    )
    parser.add_argument(
        "--task-type",
        dest="task_type",
        default=None,
        help=(
            "If set, only run user tasks whose task_type matches this string "
            "(e.g. 'code_reading'). Index in the log dir is preserved from the "
            "original INJECTION_USER_TASKS order."
        ),
    )
    parser.add_argument(
        "--template-idx",
        dest="template_idx",
        type=int,
        default=None,
        help=(
            "If set, only run the attack template at this index (0..5). "
            "Useful for re-running a single template after a template "
            "definition change without redoing the rest of the matrix."
        ),
    )
    parser.add_argument(
        "--log-root",
        dest="log_root",
        default=None,
        help=(
            "Root directory for per-sample and aggregate logs. Defaults "
            "to exp_runs/simple_inj_logs (per-sample) + exp_runs/logs/injection "
            "(aggregate). When set, both go under <log-root>/ and "
            "<log-root>/_summaries/."
        ),
    )
    parser.add_argument(
        "--conditions",
        dest="conditions",
        default="text,image",
        help=(
            "Comma-separated list of conditions to run. Valid values: "
            "text, image, audio. Default: 'text,image'. For the audio "
            "modality-gap test pass 'text,audio' (requires an "
            "audio-capable model like openai/gpt-audio-mini)."
        ),
    )
    parser.add_argument(
        "--defense",
        dest="defense",
        default="none",
        choices=["none", "spotlighting", "secalign", "sandwiching"],
        help=(
            "Optional defense baseline. 'spotlighting' applies the "
            "datamarking-style spotlighting transform from arXiv:2403.14720 "
            "to the text condition. 'secalign' runs the text condition "
            "through facebook/Meta-SecAlign-70B. 'sandwiching' repeats the "
            "user task instruction after the document to counter recency bias."
        ),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_cli_args()
    set_active_model_id(args.model_id)
    if args.render_style:
        set_render_style(args.render_style)
        console.log(f"[yellow]Render style: {args.render_style}[/yellow]")
    elif args.chat_ui:
        set_render_style("chat")
        console.log("[yellow]Render style: chat-UI[/yellow]")

    console.log(f"Using model: [bold]{get_active_model_id()}[/bold]")
    if args.defense != "none":
        console.log(f"[yellow]Defense: {args.defense}[/yellow]")

    from injection.runner import run_injection_scenario
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    run_injection_scenario(
        task_type_filter=args.task_type,
        template_idx_filter=args.template_idx,
        log_root=args.log_root,
        conditions=conditions,
        defense=args.defense,
    )


if __name__ == "__main__":
    main()
