"""Shared utilities for the image-as-defense injection experiments.

Scope is deliberately small: console/logging, an active-model tracker,
API client resolution (direct OpenAI when possible, OpenRouter otherwise),
and a text-to-image renderer for the document content.
"""

import base64
import io
import os
import textwrap

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

console = Console()


# ---------------------------------------------------------------------------
# Active model state
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "openai/gpt-5.4-mini"
_active_model_id: str = DEFAULT_MODEL_ID


def set_active_model_id(model_id: str) -> None:
    global _active_model_id
    _active_model_id = model_id


def get_active_model_id() -> str:
    return _active_model_id


# ---------------------------------------------------------------------------
# API client routing
# ---------------------------------------------------------------------------

_openai_client: OpenAI | None = None
_openrouter_client: OpenAI | None = None


def _get_openrouter_client() -> OpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY in .env")
        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _openrouter_client


def _get_openai_client() -> OpenAI | None:
    """Return a direct OpenAI client if OPENAI_API_KEY is set, else None."""
    global _openai_client
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    if _openai_client is None:
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def resolve_client_and_model(model_id: str) -> tuple[OpenAI, str]:
    """Pick the right client + canonical model id for an API call.

    - ``openai/<name>`` with ``OPENAI_API_KEY`` set → direct OpenAI
      (``https://api.openai.com``), model = ``<name>``.
    - anything else → OpenRouter, model unchanged.

    Falls back to OpenRouter for ``openai/*`` models when
    ``OPENAI_API_KEY`` is missing (back-compat). Raises if neither key
    is set.
    """
    if model_id.startswith("openai/"):
        direct = _get_openai_client()
        if direct is not None:
            return direct, model_id.removeprefix("openai/")
    return _get_openrouter_client(), model_id


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------

def _load_font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except (IOError, OSError):
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except (IOError, OSError):
            return ImageFont.load_default()


# Module-level render style. Set via set_render_style().
_render_style: str = "plain"


def set_render_style(style: str) -> None:
    """Set the active render style: 'plain' (white bg) or 'chat' (chat-UI)."""
    global _render_style
    if style not in ("plain", "chat"):
        raise ValueError(f"Unknown render style: {style!r}")
    _render_style = style


def create_content_image(content: str) -> Image.Image:
    """Render text content using the active render style."""
    if _render_style == "chat":
        return _create_chat_content_image(content)
    return _create_plain_content_image(content)


def _create_plain_content_image(content: str) -> Image.Image:
    """White background, black text — default document style."""
    font = _load_font(20, bold=True)

    wrapped: list[str] = []
    for line in content.splitlines() or [""]:
        wrapped.extend(textwrap.wrap(line, width=65) or [""])

    line_height = 30
    padding = 30
    img_h = padding * 2 + line_height * len(wrapped)

    img = Image.new("RGB", (900, max(img_h, 200)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = padding
    for line in wrapped:
        draw.text((padding, y), line, fill=(0, 0, 0), font=font)
        y += line_height
    return img


def _create_chat_content_image(content: str) -> Image.Image:
    """Chat-UI style: light gray background, white bubble, ChatGPT header,
    input bar footer with upload/voice/send icons."""
    W = 900
    PAD = 30
    LINE_H = 28
    FONT = _load_font(18)
    FONT_TITLE = _load_font(16, bold=True)
    FONT_SMALL = _load_font(13)
    BUBBLE_PAD_X = 20
    BUBBLE_PAD_Y = 16
    BUBBLE_MARGIN_LEFT = 24
    BUBBLE_RADIUS = 18
    HEADER_H = 48
    FOOTER_H = 56

    lines: list[str] = []
    for para in content.splitlines() or [""]:
        lines.extend(textwrap.wrap(para, width=58) or [""])

    bubble_text_h = LINE_H * len(lines)
    bubble_w = W - BUBBLE_MARGIN_LEFT - PAD * 2
    bubble_h = bubble_text_h + BUBBLE_PAD_Y * 2
    bubble_x = PAD + BUBBLE_MARGIN_LEFT
    bubble_y = HEADER_H + PAD
    height = max(300, bubble_y + bubble_h + PAD + FOOTER_H)

    BG = (245, 245, 245)
    BUBBLE_COLOR = (255, 255, 255)
    TEXT_COLOR = (33, 33, 33)
    SHADOW = (220, 220, 220)
    ACCENT = (16, 163, 127)
    ICON = (160, 160, 170)
    INPUT_BG = (240, 240, 240)

    img = Image.new("RGB", (W, height), BG)
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([(0, 0), (W, HEADER_H)], fill=(255, 255, 255))
    draw.line([(0, HEADER_H), (W, HEADER_H)], fill=(230, 230, 230), width=1)
    draw.ellipse([(PAD, 16), (PAD + 16, 32)], fill=ACCENT)
    draw.text((PAD + 22, 14), "ChatGPT", fill=TEXT_COLOR, font=FONT_TITLE)

    # Avatar
    draw.ellipse([(PAD - 2, bubble_y + 4), (PAD + 16, bubble_y + 22)], fill=ACCENT)

    # Bubble shadow + bubble
    draw.rounded_rectangle(
        [(bubble_x + 2, bubble_y + 2),
         (bubble_x + bubble_w + 2, bubble_y + bubble_h + 2)],
        radius=BUBBLE_RADIUS, fill=SHADOW,
    )
    draw.rounded_rectangle(
        [(bubble_x, bubble_y),
         (bubble_x + bubble_w, bubble_y + bubble_h)],
        radius=BUBBLE_RADIUS, fill=BUBBLE_COLOR,
    )

    # Text
    y = bubble_y + BUBBLE_PAD_Y
    for line in lines:
        draw.text((bubble_x + BUBBLE_PAD_X, y), line, fill=TEXT_COLOR, font=FONT)
        y += LINE_H

    # Footer
    footer_y = height - FOOTER_H
    draw.rectangle([(0, footer_y), (W, height)], fill=(255, 255, 255))
    draw.line([(0, footer_y), (W, footer_y)], fill=(230, 230, 230), width=1)

    input_x, input_w = 70, W - 140
    input_y, input_h = footer_y + 12, 32
    draw.rounded_rectangle(
        [(input_x, input_y), (input_x + input_w, input_y + input_h)],
        radius=16, fill=INPUT_BG,
    )
    draw.text((input_x + 14, input_y + 7), "Message ChatGPT...",
              fill=(170, 170, 175), font=FONT_SMALL)

    cy = footer_y + 28
    # Paperclip icon
    cx = 40
    draw.arc([(cx - 8, cy - 10), (cx + 8, cy + 6)], start=180, end=0, fill=ICON, width=2)
    draw.line([(cx - 8, cy - 2), (cx - 8, cy + 4)], fill=ICON, width=2)
    draw.line([(cx + 8, cy - 2), (cx + 8, cy + 4)], fill=ICON, width=2)
    draw.arc([(cx - 4, cy - 6), (cx + 4, cy + 4)], start=0, end=180, fill=ICON, width=2)
    # Mic icon
    mx = W - 60
    draw.ellipse([(mx - 4, cy - 10), (mx + 4, cy - 2)], outline=ICON, width=2)
    draw.line([(mx, cy - 2), (mx, cy + 2)], fill=ICON, width=2)
    draw.arc([(mx - 7, cy - 6), (mx + 7, cy + 4)], start=0, end=180, fill=ICON, width=2)
    # Send arrow
    sx = W - 30
    draw.polygon([(sx, cy - 10), (sx - 7, cy), (sx + 7, cy)], fill=ACCENT)
    draw.rectangle([(sx - 2, cy), (sx + 2, cy + 6)], fill=ACCENT)

    return img


def image_data_url(img: Image.Image) -> str:
    """PNG → data URL suitable for OpenAI/OpenRouter image_url fields."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def image_b64(img: Image.Image) -> str:
    """PNG → raw base64 (no data: prefix) for Anthropic image blocks."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
