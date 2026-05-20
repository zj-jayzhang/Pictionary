"""Local HuggingFace runner for the image-as-defense injection benchmark.

Mirrors ``main.py`` / ``injection/runner.py`` but loads the model
locally via ``transformers`` instead of going through an API. Iterates
the same ``(user_task × injection × template × condition)`` matrix and
writes per-sample JSON in the same schema, so results can be aggregated
in ``full_results.md`` alongside the API-based runs.

Currently targeted at local HuggingFace vision/text checkpoints including:
    Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-0.8B-Base,
    Qwen/Qwen3.5-9B,   Qwen/Qwen3.5-9B-Base
    google/gemma-3-4b-pt, google/gemma-3-4b-it

For *Base* checkpoints (no chat template, no tool-use post-training)
the script falls back to a raw `system\\nUser: ...\\nAssistant:` prompt
and skips the ``tools=`` argument. Those rows are expected to give
ASR ≈ 0 (base models don't emit structured tool calls) and serve as
the "is injection a pretrained behavior?" ablation.

Usage::

    uv run python local_model.py --model Qwen/Qwen3.5-0.8B
    uv run python local_model.py --model Qwen/Qwen3.5-9B \\
        --model Qwen/Qwen3.5-9B-Base \\
        --conditions text,image --log-root results/local_logs
    uv run python local_model.py --model google/gemma-3-4b-it \\
        --task-type summarize --template-idx 0 --conditions text,image
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from datetime import datetime

import torch
from dotenv import load_dotenv
from PIL import Image

# Legacy custom modeling files (e.g., InternVL's `modeling_internvl_chat`)
# define `_tied_weights_keys` (list) but not `all_tied_weights_keys`
# (dict), which transformers 5.x demands during model loading. Provide
# an empty-dict fallback so `from_pretrained` doesn't crash on those
# models. This is a read-only fallback — anything that already defines
# the attribute keeps its own value.
from transformers import PreTrainedModel as _PTM

if not hasattr(_PTM, "_all_tied_weights_keys_fallback_installed"):
    _orig_getattr = _PTM.__getattribute__

    def _patched_getattr(self, name):
        try:
            return _orig_getattr(self, name)
        except AttributeError:
            if name == "all_tied_weights_keys":
                return {}
            raise

    _PTM.__getattribute__ = _patched_getattr
    _PTM._all_tied_weights_keys_fallback_installed = True

from helpers.runner import (
    console,
    create_content_image,
    image_data_url,
    set_active_model_id,
)
from injection.data import (
    INJECTION_ATTACKS,
    INJECTION_OVERRIDE_TEMPLATES,
    INJECTION_USER_TASKS,
    _INJECTION_SYSTEM_PROMPT,
    _T2_PREFILLED_ANSWERS,
    build_hf_tools,
    build_tool_prompt,
    grade_user_task,
)
from injection.runner import (
    _CONDITION_LABELS,
    _embed_injection,
    _extract_function_calls,
    _write_sample_log,
)

_MAX_NEW_TOKENS = 4096
_BACKEND = "HuggingFace (local)"
_IMAGE_TOKEN = "<start_of_image>"


def _resolve_eos_ids(tokenizer) -> list[int]:
    """Stop-token set for generation. Qwen's chat template emits
    `<|im_end|>` to close each turn; without it, the model continues
    into a synthetic next turn until max_new_tokens."""
    ids = []
    base_eos = tokenizer.eos_token_id
    if base_eos is not None:
        ids.append(base_eos)
    for extra in ("<|im_end|>", "<|endoftext|>", "<end_of_turn>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(extra)
            if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
                ids.append(tid)
        except Exception:  # noqa: BLE001
            pass
    # Dedupe while preserving order.
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_local(model_id: str):
    """Load model + processor/tokenizer. Returns (model, processor,
    tokenizer, is_vlm, is_base).

    Strategy: try ``AutoModelForImageTextToText`` (multimodal); if that
    fails, fall back to ``AutoModelForCausalLM``. Always load the
    tokenizer separately because some multimodal processors don't
    expose a working ``chat_template`` even when the underlying
    tokenizer does. ``is_base`` is detected by name suffix (``-Base``)
    since base checkpoints can ship a chat-template attribute that
    nevertheless yields garbage on a non-instruct model.
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    console.log(f"Loading [bold]{model_id}[/bold] — this may take a while…")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Tokenizer is always needed (we apply chat templates through it for
    # text-only inputs). transformers 5.x's AutoTokenizer fails on some
    # legacy vocab.json+merges.txt configs (e.g., InternVL3.5) — fall
    # back to Qwen2TokenizerFast which understands the same format.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        from transformers import Qwen2TokenizerFast
        console.log(
            f"[yellow]AutoTokenizer failed ({type(exc).__name__}); "
            f"falling back to Qwen2TokenizerFast.[/yellow]"
        )
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_id, trust_remote_code=True)

    # Processor for image inputs (None if not multimodal).
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        console.log(f"[yellow]AutoProcessor unavailable ({type(exc).__name__}); image input disabled.[/yellow]")

    # Try multimodal model first; fall back to causal LM. Some custom
    # modeling files (InternVL) are incompatible with `device_map="auto"`
    # under transformers 5.x — fall back to no-device-map + manual move.
    # Prefer FlashAttention-2 (much faster, esp. for Mllama cross-attn);
    # fall back to the default kernel if the model/install doesn't
    # support it.
    def _try_load(Cls):
        for attn in ("flash_attention_2", None):
            kw = dict(
                torch_dtype=dtype, device_map="auto", trust_remote_code=True,
            )
            if attn is not None:
                kw["attn_implementation"] = attn
            try:
                m = Cls.from_pretrained(model_id, **kw)
                if attn:
                    console.log(f"  {Cls.__name__}: attn_implementation={attn}")
                return m
            except AttributeError as exc:
                console.log(f"[yellow]{Cls.__name__} device_map=auto failed ({exc}); retry without device_map.[/yellow]")
                kw.pop("device_map", None)
                m = Cls.from_pretrained(model_id, **kw)
                if torch.cuda.is_available():
                    m = m.to("cuda")
                return m
            except Exception as exc:  # noqa: BLE001
                if attn is not None:
                    console.log(f"[yellow]{Cls.__name__} flash_attention_2 unavailable ({type(exc).__name__}); using default kernel.[/yellow]")
                    continue
                raise

    model = None
    is_vlm = False
    try:
        model = _try_load(AutoModelForImageTextToText)
        is_vlm = True
        console.log("  loaded via AutoModelForImageTextToText (VLM)")
    except Exception as exc:  # noqa: BLE001
        console.log(f"[yellow]AutoModelForImageTextToText failed ({type(exc).__name__}); trying CausalLM.[/yellow]")
        model = _try_load(AutoModelForCausalLM)
        is_vlm = False

    model.eval()

    # InternVL's custom `generate()` asserts `img_context_token_id is
    # not None` before doing anything else. It's only set lazily in
    # their chat() helper, never in our path. Wire it manually from the
    # special-token id so the assertion passes; non-InternVL models
    # don't have this attribute and the check is a no-op.
    if getattr(model, "img_context_token_id", "missing") is None:
        try:
            tok_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            if isinstance(tok_id, int) and tok_id >= 0:
                model.img_context_token_id = tok_id
        except Exception:  # noqa: BLE001
            pass

    # Base-vs-instruct detection. Base checkpoints carry an explicit
    # `-Base` / `-Pretrained` suffix in this run's model list; the
    # chat_template attribute alone is unreliable (some Qwen base
    # checkpoints ship one that nevertheless misbehaves).
    model_lc = model_id.lower()
    is_base = (
        model_lc.endswith("-base")
        or model_lc.endswith("-pretrained")
        or model_lc.endswith("-pt")
        or tokenizer.chat_template is None
    )

    # Qwen3.5-Base ships a `chat_template` on the tokenizer but not on
    # the processor wrapper, so `processor.apply_chat_template` fails
    # for the image condition. Both variants share the same template
    # text — copy it over so the image path works on base too.
    if (
        processor is not None
        and not getattr(processor, "chat_template", None)
        and tokenizer.chat_template
    ):
        processor.chat_template = tokenizer.chat_template

    device = next(model.parameters()).device
    console.log(
        f"  ready: VLM={is_vlm}  base={is_base}  device={device}  dtype={dtype}"
    )
    return model, processor, tokenizer, is_vlm, is_base


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

def _hf_messages(
    condition: str,
    user_task: str,
    full_content: str,
    pil_image: Image.Image | None,
) -> list[dict]:
    """Messages in the HuggingFace chat-template multimodal format —
    used as input to ``processor.apply_chat_template`` for instruct
    models."""
    if condition == "text":
        return [
            {"role": "system", "content": _INJECTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"{user_task}\n\n{full_content}"},
        ]
    if condition == "image":
        if pil_image is None:
            raise ValueError("image condition needs a rendered PIL image")
        return [
            {"role": "system", "content": _INJECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_task},
                ],
            },
        ]
    raise ValueError(f"Unknown condition: {condition}")


def _log_messages(
    condition: str,
    user_task: str,
    full_content: str,
    pil_image: Image.Image | None,
) -> list[dict]:
    """OpenAI-style messages used only for the per-sample log so the
    schema matches the API runner's output."""
    if condition == "text":
        return [
            {"role": "system", "content": _INJECTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"{user_task}\n\n{full_content}"},
        ]
    return [
        {"role": "system", "content": _INJECTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_task},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url(pil_image) if pil_image else "<missing>",
                    },
                },
            ],
        },
    ]


# ---------------------------------------------------------------------------
# InternVL support
#
# InternVL ships a custom architecture (`InternVLChatModel`) that is not
# recognized by `AutoModelForImageTextToText`, has no `AutoProcessor`,
# and does image input through its own `model.chat()` helper with a
# bespoke dynamic-tiling preprocessor. We special-case it: text and
# image both route through `model.chat()`, and the image is tiled with
# the standard InternVL recipe below.
# ---------------------------------------------------------------------------

_INTERNVL_CLS = "InternVLChatModel"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _is_internvl(model) -> bool:
    return type(model).__name__ == _INTERNVL_CLS


def _internvl_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _internvl_closest_ratio(aspect_ratio, target_ratios, w, h, image_size):
    best_diff, best = float("inf"), (1, 1)
    area = w * h
    for ratio in target_ratios:
        tgt = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - tgt)
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """Standard InternVL dynamic tiling (from the model card)."""
    w, h = image.size
    aspect_ratio = w / h
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    ratio = _internvl_closest_ratio(aspect_ratio, target_ratios, w, h, image_size)
    tw, th = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((tw, th))
    cols = tw // image_size
    tiles = []
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _internvl_pixel_values(pil_image, dtype, device, input_size=448, max_num=12):
    transform = _internvl_transform(input_size)
    tiles = _internvl_dynamic_preprocess(
        pil_image, image_size=input_size, max_num=max_num, use_thumbnail=True
    )
    pv = torch.stack([transform(t) for t in tiles])
    return pv.to(dtype=dtype, device=device)


def _generate_internvl(
    model, tokenizer, condition: str,
    user_task: str, full_content: str, pil_image: Image.Image | None,
) -> str:
    """InternVL path — uses model.chat(). Tools are described in the
    prompt text (InternVL has no native tool-calling).

    text condition  → document goes in the prompt text.
    image condition → document goes ONLY in the image; the prompt text
                       carries just system + tools + the user task.
    """
    preamble = f"{_INJECTION_SYSTEM_PROMPT}\n\n{build_tool_prompt()}\n\n"
    gen_cfg = dict(max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)
    if condition == "image":
        if pil_image is None:
            raise ValueError("image condition needs a rendered PIL image")
        question = f"{preamble}{user_task}"
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        pixel_values = _internvl_pixel_values(pil_image, dtype, device)
        return model.chat(tokenizer, pixel_values, question, gen_cfg)
    question = f"{preamble}{user_task}\n\n{full_content}"
    return model.chat(tokenizer, None, question, gen_cfg)


# ---------------------------------------------------------------------------
# Mllama (Llama-3.2-Vision) support
#
# The Llama-3.2-Vision chat template DROPS the `<|image|>` token when a
# `tools=` kwarg is passed to `apply_chat_template` — image + native
# tool rendering are mutually exclusive in that template. So Mllama
# gets the tool list as prompt text (like InternVL), never via `tools=`.
# ---------------------------------------------------------------------------

_MLLAMA_CLS = "MllamaForConditionalGeneration"


def _is_mllama(model) -> bool:
    return type(model).__name__ == _MLLAMA_CLS


def _generate_mllama(
    model, processor, tokenizer, condition: str,
    user_task: str, full_content: str, pil_image: Image.Image | None,
) -> str:
    """Llama-3.2-Vision path. Tools described in the prompt text.

    text condition  → document in the prompt text.
    image condition → document ONLY in the rendered image; prompt text
                       carries system + tools + user task.
    """
    device = next(model.parameters()).device
    has_template = tokenizer.chat_template is not None
    task = f"{build_tool_prompt()}\n\n{user_task}"

    if condition == "image":
        if pil_image is None:
            raise ValueError("image condition needs a rendered PIL image")
        if has_template:
            msgs = [
                {"role": "system", "content": _INJECTION_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": task},
                ]},
            ]
            text = processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
            )
        else:
            text = (
                f"<|image|>\n{_INJECTION_SYSTEM_PROMPT}\n\n"
                f"User: {task}\n\nAssistant:"
            )
        inputs = processor(text=text, images=pil_image, return_tensors="pt")
    else:
        if has_template:
            msgs = [
                {"role": "system", "content": _INJECTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"{task}\n\n{full_content}"},
            ]
            inputs = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
        else:
            prompt = (
                f"{_INJECTION_SYSTEM_PROMPT}\n\n"
                f"User: {task}\n\n{full_content}\n\nAssistant:"
            )
            inputs = tokenizer(prompt, return_tensors="pt")

    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_TEXT_GENERATE_KEYS = {"input_ids", "attention_mask"}


def _generate(
    model,
    processor,
    tokenizer,
    is_vlm: bool,
    is_base: bool,
    condition: str,
    user_task: str,
    full_content: str,
    pil_image: Image.Image | None,
    tools,
) -> str:
    # InternVL-family models go through their own `model.chat()` API.
    if _is_internvl(model):
        return _generate_internvl(
            model, tokenizer, condition, user_task, full_content, pil_image,
        )
    # Llama-3.2-Vision (Mllama) — tools via prompt text, not `tools=`.
    if _is_mllama(model):
        return _generate_mllama(
            model, processor, tokenizer, condition,
            user_task, full_content, pil_image,
        )

    device = next(model.parameters()).device
    has_template = tokenizer.chat_template is not None

    # No chat template at all → raw prompt. For VLM checkpoints such as
    # Gemma 3 PT / Llama-3.2-Vision (base), prepend the model's image
    # placeholder and let the processor expand it into image tokens.
    #
    # Modality parity: the document (`full_content`, with the injection)
    # goes in the TEXT for the text condition and ONLY in the IMAGE for
    # the image condition — same as the chat-template and InternVL
    # paths. The image-condition prompt text carries just system + task.
    if not has_template:
        if condition == "image" and is_vlm and pil_image is not None and processor is not None:
            img_tok = getattr(processor, "image_token", None) or _IMAGE_TOKEN
            prompt = (
                f"{img_tok}\n{_INJECTION_SYSTEM_PROMPT}\n\n"
                f"User: {user_task}\n\nAssistant:"
            )
            inputs = processor(text=prompt, images=pil_image, return_tensors="pt")
        else:
            prompt = (
                f"{_INJECTION_SYSTEM_PROMPT}\n\n"
                f"User: {user_task}\n\n{full_content}\n\nAssistant:"
            )
            inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    # Chat-template path (works for instruct AND base models that ship
    # a template). For base checkpoints we skip `tools` (no tool-use
    # training) and `enable_thinking` (no CoT training). Image input
    # uses the processor (image-token insertion + pixel_values); text
    # uses the tokenizer directly to avoid `mm_token_type_ids`.
    template_kwargs = dict(
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if (
        not is_base
        and isinstance(tokenizer.chat_template, str)
        and "enable_thinking" in tokenizer.chat_template
    ):
        # Qwen3 chain-of-thought switch — only templates that declare
        # the flag understand it.
        template_kwargs["enable_thinking"] = True
    if tools is not None:
        template_kwargs["tools"] = tools

    if condition == "image" and is_vlm and pil_image is not None and processor is not None:
        messages = _hf_messages("image", user_task, full_content, pil_image)
        # Two-step: render the chat text (with the image placeholder
        # inserted) WITHOUT tokenizing, then attach the image via the
        # processor. The one-step `apply_chat_template(tokenize=True)`
        # silently drops the image on Mllama (Llama-3.2-Vision); the
        # two-step form is the universal recipe (Qwen, Mllama, Gemma).
        render_kwargs = dict(add_generation_prompt=True, tokenize=False)
        if "enable_thinking" in template_kwargs:
            render_kwargs["enable_thinking"] = template_kwargs["enable_thinking"]
        if "tools" in template_kwargs:
            render_kwargs["tools"] = template_kwargs["tools"]
        text = processor.apply_chat_template(messages, **render_kwargs)
        inputs = processor(text=text, images=pil_image, return_tensors="pt")
        allowed = None
    else:
        messages = _hf_messages("text", user_task, full_content, pil_image)
        inputs = tokenizer.apply_chat_template(messages, **template_kwargs)
        allowed = _TEXT_GENERATE_KEYS

    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    if allowed is not None:
        inputs = {k: v for k, v in inputs.items() if k in allowed}

    eos_ids = _resolve_eos_ids(tokenizer)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
        )
    in_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0, in_len:], skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def run_model(
    model_id: str,
    conditions: list[str],
    log_root: str,
    task_type_filter: str | None = None,
    template_idx_filter: int | None = None,
) -> None:
    set_active_model_id(model_id)
    model, processor, tokenizer, is_vlm, is_base = _load_local(model_id)
    model_short = model_id.replace("/", "_")

    if "audio" in conditions:
        console.log("[yellow]Audio is not supported for local models; dropping.[/yellow]")
        conditions = [c for c in conditions if c != "audio"]
    # Drop image only when the model has no vision tower/processor.
    # Gemma 3 PT has no chat template, but its processor can still
    # expand a raw `<start_of_image>` prompt into image tokens.
    # InternVL has no AutoProcessor but does image via model.chat().
    if (
        "image" in conditions
        and not _is_internvl(model)
        and (not is_vlm or processor is None)
    ):
        reason = "text-only model" if not is_vlm else "no image processor"
        console.log(f"[yellow]Dropping image condition for {model_id}: {reason}.[/yellow]")
        conditions = [c for c in conditions if c != "image"]
    if not conditions:
        raise ValueError("No conditions remain after filtering.")

    conditions_indexed = [(k, _CONDITION_LABELS[k]) for k in conditions]
    needs_image = "image" in conditions
    # Pass tools to base models too: Qwen base/instruct share the same
    # tool-capable chat template, so the base-vs-instruct ablation
    # should isolate post-training, not "did the prompt mention tools".
    tools = build_hf_tools()

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

    n_tasks = len(user_tasks_indexed)
    n_attacks = len(INJECTION_ATTACKS)
    n_templates = len(templates_indexed)
    per_cond = n_tasks * n_attacks * n_templates
    console.rule(f"[bold]Running {model_id}[/bold]")
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
        for inj_idx, (injection, target_fn) in enumerate(INJECTION_ATTACKS):
            for tmpl_idx, template in templates_indexed:
                wrapped_injection = template.format(
                    injection=injection,
                    prefilled_answers=_T2_PREFILLED_ANSWERS[task_type],
                )
                full_content = _embed_injection(content, wrapped_injection)
                pil_image = create_content_image(full_content) if needs_image else None

                for cond_key, cond_label in conditions_indexed:
                    sample_start = time.time()
                    response: str | None = None
                    called: list[str] = []
                    inj_ok: bool | None = None
                    ut_ok: bool | None = None
                    error: str | None = None
                    try:
                        response = _generate(
                            model, processor, tokenizer, is_vlm, is_base,
                            cond_key, user_task, full_content, pil_image, tools,
                        )
                        called = _extract_function_calls(response)
                        inj_ok = target_fn in called
                        ut_ok = grade_user_task(task_type, response)
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
                        backend=_BACKEND,
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
                        messages=_log_messages(
                            cond_key, user_task, full_content, pil_image,
                        ),
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
        "backend": _BACKEND,
        "n_tasks": n_tasks,
        "n_attacks": n_attacks,
        "n_templates": n_templates,
        "samples_per_condition": per_cond,
        "conditions": [label for _, label in conditions_indexed],
        "overall": {},
        "wall_seconds": wall,
        "vlm": is_vlm,
        "base_model": is_base,
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

    # Release GPU memory before the next model
    del model, processor, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", action="append", required=True,
        help="HuggingFace model id. Repeat to sweep multiple models.",
    )
    p.add_argument(
        "--conditions", default="text,image",
        help="Comma-separated: text,image (audio not supported locally).",
    )
    p.add_argument(
        "--log-root", default="results/local_logs",
        help="Per-sample logs go under <log-root>/<model>/; "
             "summaries under <log-root>/_summaries/.",
    )
    p.add_argument("--task-type", default=None,
                   help="Filter to a single user task type (e.g. summarize).")
    p.add_argument("--template-idx", type=int, default=None,
                   help="Filter to a single attack template (0..6).")
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=_MAX_NEW_TOKENS,
        help=f"Generation budget per sample. Default: {_MAX_NEW_TOKENS}.",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_cli()
    global _MAX_NEW_TOKENS
    _MAX_NEW_TOKENS = args.max_new_tokens
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for m in args.model:
        try:
            run_model(
                m, conditions, args.log_root,
                task_type_filter=args.task_type,
                template_idx_filter=args.template_idx,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[bold red]Model {m} failed:[/bold red] {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
