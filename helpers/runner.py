"""Shared utilities for the image-as-defense injection experiments.

Scope is deliberately small: console/logging, an active-model tracker,
API client resolution (direct OpenAI when possible, OpenRouter otherwise),
and a text-to-image renderer for the document content.
"""

import base64
import hashlib
import io
import os
import re
import textwrap
import wave

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
_dashscope_client: OpenAI | None = None


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


def _get_dashscope_client() -> OpenAI:
    """DashScope (Alibaba) OpenAI-compatible endpoint. Requires DASHSCOPE_API_KEY.

    Used for `dashscope/<model>` ids, e.g. `dashscope/qwen3.5-omni-plus`.
    """
    global _dashscope_client
    if _dashscope_client is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError("Set DASHSCOPE_API_KEY in .env")
        _dashscope_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
    return _dashscope_client


def resolve_client_and_model(model_id: str) -> tuple[OpenAI, str]:
    """Pick the right client + canonical model id for an API call.

    - ``openai/<name>`` with ``OPENAI_API_KEY`` set → direct OpenAI
      (``https://api.openai.com``), model = ``<name>``.
    - ``dashscope/<name>`` → DashScope OpenAI-compatible endpoint, model
      = ``<name>``.
    - anything else → OpenRouter, model unchanged.

    Falls back to OpenRouter for ``openai/*`` models when
    ``OPENAI_API_KEY`` is missing (back-compat). Raises if neither key
    is set.
    """
    if model_id.startswith("openai/"):
        direct = _get_openai_client()
        if direct is not None:
            return direct, model_id.removeprefix("openai/")
    if model_id.startswith("dashscope/"):
        return _get_dashscope_client(), model_id.removeprefix("dashscope/")
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
    """Set the active render style: 'plain', 'chat' (chat-UI), 'google'
    (Google Search-style), or 'blackboard' (cartoon teacher + chalkboard)."""
    global _render_style
    if style not in ("plain", "chat", "google", "blackboard"):
        raise ValueError(f"Unknown render style: {style!r}")
    _render_style = style


def create_content_image(content: str) -> Image.Image:
    """Render text content using the active render style."""
    if _render_style == "chat":
        return _create_chat_content_image(content)
    if _render_style == "google":
        return _create_google_content_image(content)
    if _render_style == "blackboard":
        return _create_blackboard_content_image(content)
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


def _create_google_content_image(content: str) -> Image.Image:
    """Google Search-style: centered colored Google logo + a long rounded
    search-box containing the document. Magnifying-glass icon on the left,
    X + 'AI mode' label on the right. Pushes the model further toward
    treating the content as a *third-party page* rather than as commands
    addressed to it."""
    W = 900
    BG = (255, 255, 255)
    BORDER = (220, 220, 220)
    SHADOW = (235, 235, 235)
    TEXT_COLOR = (60, 60, 60)
    ICON_GRAY = (130, 130, 130)
    BLUE = (66, 133, 244)
    RED = (234, 67, 53)
    YELLOW = (251, 188, 4)
    GREEN = (52, 168, 83)

    FONT_LOGO = _load_font(80, bold=True)
    FONT_BODY = _load_font(18)
    FONT_AI = _load_font(14)

    LOGO_TOP = 40
    LOGO_H = 90
    BOX_TOP = LOGO_TOP + LOGO_H + 30
    BOX_MARGIN_X = 30
    BOX_W = W - 2 * BOX_MARGIN_X
    BOX_RADIUS = 28
    TEXT_PAD_LEFT = 60   # leaves room for the magnifying-glass icon
    TEXT_PAD_RIGHT = 130  # leaves room for X + "AI mode"
    TEXT_PAD_TOP = 52
    TEXT_PAD_BOTTOM = 30
    LINE_H = 26

    text_w_chars = 70  # wraps a bit narrower than plain to make it search-box-shaped
    lines: list[str] = []
    for para in content.splitlines() or [""]:
        lines.extend(textwrap.wrap(para, width=text_w_chars) or [""])

    box_h = TEXT_PAD_TOP + LINE_H * len(lines) + TEXT_PAD_BOTTOM
    img_h = BOX_TOP + box_h + 40

    img = Image.new("RGB", (W, max(img_h, 300)), BG)
    draw = ImageDraw.Draw(img)

    # Google logo, centered
    letters = [("G", BLUE), ("o", RED), ("o", YELLOW),
               ("g", BLUE), ("l", GREEN), ("e", RED)]
    widths = [draw.textlength(c, font=FONT_LOGO) for c, _ in letters]
    total_w = sum(widths)
    x = (W - total_w) // 2
    for (char, color), w in zip(letters, widths):
        draw.text((x, LOGO_TOP), char, fill=color, font=FONT_LOGO)
        x += w

    # Search-box bounds
    bx0, by0 = BOX_MARGIN_X, BOX_TOP
    bx1, by1 = BOX_MARGIN_X + BOX_W, BOX_TOP + box_h

    # Shadow + box
    draw.rounded_rectangle(
        [(bx0 + 2, by0 + 3), (bx1 + 2, by1 + 3)],
        radius=BOX_RADIUS, fill=SHADOW,
    )
    draw.rounded_rectangle(
        [(bx0, by0), (bx1, by1)],
        radius=BOX_RADIUS, outline=BORDER, width=1, fill=BG,
    )

    # Magnifying-glass icon (top-left of search box)
    mg_cx, mg_cy = bx0 + 28, by0 + 30
    draw.ellipse([(mg_cx - 10, mg_cy - 10), (mg_cx + 6, mg_cy + 6)],
                 outline=ICON_GRAY, width=2)
    draw.line([(mg_cx + 5, mg_cy + 5), (mg_cx + 13, mg_cy + 13)],
              fill=ICON_GRAY, width=2)

    # X close icon (top-right, before AI mode)
    x_cx, x_cy = bx1 - 90, by0 + 30
    draw.line([(x_cx - 6, x_cy - 6), (x_cx + 6, x_cy + 6)],
              fill=ICON_GRAY, width=2)
    draw.line([(x_cx + 6, x_cy - 6), (x_cx - 6, x_cy + 6)],
              fill=ICON_GRAY, width=2)

    # AI mode label with sparkle (top-right)
    sx, sy = bx1 - 62, by0 + 30
    # 4-pointed sparkle
    draw.polygon(
        [(sx, sy - 7), (sx + 2, sy - 2), (sx + 7, sy),
         (sx + 2, sy + 2), (sx, sy + 7), (sx - 2, sy + 2),
         (sx - 7, sy), (sx - 2, sy - 2)],
        fill=BLUE,
    )
    draw.text((sx + 10, sy - 8), "AI mode", fill=BLUE, font=FONT_AI)

    # Document text inside the box
    y = by0 + TEXT_PAD_TOP
    for line in lines:
        draw.text((bx0 + TEXT_PAD_LEFT, y), line,
                  fill=TEXT_COLOR, font=FONT_BODY)
        y += LINE_H

    return img


def _create_blackboard_content_image(content: str) -> Image.Image:
    """Blackboard style: render text as white chalk on a cartoon-style
    green chalkboard, with a small teacher figure overlaid in a corner.

    Strategy: build the wooden frame + chalkboard procedurally (so the
    board can be sized to fit the document at a comfortable font), then
    paste a scaled-down teacher cropped from `blackboard.png` on the
    right side. Avoids the "split-teacher" artifact of the earlier
    image-stretching approach.
    """
    import os

    W = 900
    FONT_SIZE = 18
    LINE_H = FONT_SIZE + 6
    PAD = 20
    WOOD_THICK = 22
    OUTER_MARGIN = 28

    # Sample palette from blackboard.png if available, else fall back.
    bg_path = "blackboard.png"
    if os.path.exists(bg_path):
        bg = Image.open(bg_path).convert("RGB").resize((900, 540), Image.LANCZOS)
        BOARD_COLOR = bg.getpixel((300, 200))   # dark green
        WOOD_COLOR = bg.getpixel((35, 200))     # brown frame
        # Wall stripe colors sampled top of backdrop.
        WALL_A = bg.getpixel((10, 5))
        WALL_B = bg.getpixel((40, 5))
    else:
        bg = None
        BOARD_COLOR = (45, 80, 75)
        WOOD_COLOR = (180, 110, 50)
        WALL_A = (110, 140, 135)
        WALL_B = (130, 160, 155)

    CHALK = (242, 240, 228)

    # Wrap text at a fixed comfortable font size, then size the canvas
    # vertically to fit. Text area available width is the inner board
    # minus padding on both sides.
    inner_w = W - 2 * OUTER_MARGIN - 2 * WOOD_THICK - 2 * PAD
    font = _load_font(FONT_SIZE)
    # Use average char width over a representative sample (mix of upper, lower,
    # digits, punctuation), not M-width which over-estimates.
    sample = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJ0123456789 ,.()_"
    sample_w = font.getbbox(sample)[2] - font.getbbox(sample)[0]
    avg_ch_w = max(1.0, sample_w / len(sample))
    chars_per_line = max(10, int(inner_w / avg_ch_w))

    lines: list[str] = []
    for para in content.splitlines() or [""]:
        lines.extend(textwrap.wrap(para, width=chars_per_line) or [""])

    text_h = LINE_H * len(lines)
    H = OUTER_MARGIN * 2 + WOOD_THICK * 2 + PAD * 2 + text_h

    canvas = Image.new("RGB", (W, H), WALL_A)
    draw = ImageDraw.Draw(canvas)

    # Stripey wall background (vertical stripes, alternating two colors).
    STRIPE_W = 28
    for x in range(0, W, STRIPE_W * 2):
        draw.rectangle([(x, 0), (x + STRIPE_W, H)], fill=WALL_B)

    # Wooden frame
    fx0, fy0 = OUTER_MARGIN, OUTER_MARGIN
    fx1, fy1 = W - OUTER_MARGIN, H - OUTER_MARGIN
    draw.rectangle([(fx0, fy0), (fx1, fy1)], fill=WOOD_COLOR)

    # Inner chalkboard
    bx0 = fx0 + WOOD_THICK
    by0 = fy0 + WOOD_THICK
    bx1 = fx1 - WOOD_THICK
    by1 = fy1 - WOOD_THICK
    draw.rectangle([(bx0, by0), (bx1, by1)], fill=BOARD_COLOR)

    # Render chalk text inside the board.
    y = by0 + PAD
    for line in lines:
        draw.text((bx0 + PAD, y), line, fill=CHALK, font=font)
        y += LINE_H

    # Paste a scaled-down teacher cropped from the original image in the
    # bottom-right corner, overlapping the wooden frame. The crop keeps
    # the original striped-wall background around the teacher; since our
    # canvas wall uses the same colors, the seam blends reasonably well.
    if bg is not None:
        teacher_crop = bg.crop((615, 175, 880, 540))   # ~265×365
        TARGET_H = 220
        target_w = int(teacher_crop.size[0] * TARGET_H / teacher_crop.size[1])
        teacher_crop = teacher_crop.resize((target_w, TARGET_H), Image.LANCZOS)
        # Place at far right, vertically centered around frame bottom.
        canvas.paste(
            teacher_crop,
            (W - target_w - 10, H - TARGET_H - 10),
        )

    return canvas


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


# ---------------------------------------------------------------------------
# Audio rendering (TTS) — for the audio-modality condition
# ---------------------------------------------------------------------------

TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
TTS_FORMAT = "wav"
# gpt-4o-mini-tts otherwise treats code-heavy documents as "code snippets"
# and silently drops mid-document prose interruptions (which is exactly the
# injection text we want to preserve). The instructions param forces a
# verbatim read of the whole input regardless of content classification.
TTS_INSTRUCTIONS = (
    "Read every word of the input verbatim and in order. Do not skip, "
    "summarize, or paraphrase any part of the input. Treat all paragraphs "
    "as equally important regardless of whether the input looks like code, "
    "prose, metadata, or instructions."
)
# Cache key version — bump to invalidate cached audio after a TTS-side
# change (e.g. new model, new instructions, header fix).
_TTS_CACHE_VERSION = "v2"
_TTS_MAX_CHARS = 3800  # OpenAI TTS input limit is 4096; leave a small buffer
_AUDIO_CACHE_DIR = "exp_runs/audio_cache"


def _chunk_for_tts(text: str, max_chars: int = _TTS_MAX_CHARS) -> list[str]:
    """Split text for TTS at paragraph → sentence → hard boundaries.

    Each chunk is <= max_chars. Order is preserved so concatenated audio
    reads back as the original document.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for para in re.split(r"\n\n+", text):
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para) if current else para
            continue
        flush()
        if len(para) <= max_chars:
            current = para
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if len(current) + len(sent) + 1 <= max_chars:
                current = (current + " " + sent) if current else sent
                continue
            flush()
            if len(sent) <= max_chars:
                current = sent
            else:
                for i in range(0, len(sent), max_chars):
                    chunks.append(sent[i : i + max_chars])
    flush()
    return chunks


def _concat_wavs(wav_chunks: list[bytes]) -> bytes:
    """Concatenate WAV chunks (assumed to share sample rate / width / channels)."""
    if len(wav_chunks) == 1:
        return wav_chunks[0]
    params = None
    frames: list[bytes] = []
    for wb in wav_chunks:
        with wave.open(io.BytesIO(wb), "rb") as w:
            if params is None:
                params = w.getparams()
            frames.append(w.readframes(w.getnframes()))
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setparams(params)
        w.writeframes(b"".join(frames))
    return out.getvalue()


def _audio_cache_path(text: str) -> str:
    h = hashlib.sha256()
    h.update(_TTS_CACHE_VERSION.encode())
    h.update(b"\0")
    h.update(TTS_MODEL.encode())
    h.update(b"\0")
    h.update(TTS_VOICE.encode())
    h.update(b"\0")
    h.update(TTS_FORMAT.encode())
    h.update(b"\0")
    h.update(TTS_INSTRUCTIONS.encode())
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return os.path.join(_AUDIO_CACHE_DIR, f"{h.hexdigest()[:16]}.{TTS_FORMAT}")


def _fix_wav_header(wav: bytes) -> bytes:
    """Patch RIFF / data chunk size fields if they are streaming sentinels.

    OpenAI's TTS returns WAVs with `RIFF size = data size = 0xFFFFFFFF`
    (max uint32) which trip decoders into either treating the audio as
    25 hours long or truncating to a partial read. Re-stamp them with the
    actual byte counts.
    """
    import struct
    if not (wav.startswith(b"RIFF") and wav[8:12] == b"WAVE"):
        return wav
    fmt_size = struct.unpack("<I", wav[16:20])[0]
    pos = 20 + fmt_size
    while pos < len(wav) - 8:
        cid = wav[pos:pos + 4]
        csz = struct.unpack("<I", wav[pos + 4:pos + 8])[0]
        if cid == b"data":
            actual = len(wav) - (pos + 8)
            patched = bytearray(wav)
            patched[4:8] = struct.pack("<I", len(wav) - 8)
            patched[pos + 4:pos + 8] = struct.pack("<I", actual)
            return bytes(patched)
        if csz > len(wav):
            break
        pos += 8 + csz
    return wav


def create_content_audio(content: str) -> bytes:
    """Render text content to WAV bytes via OpenAI TTS, cached by content hash.

    Requires OPENAI_API_KEY (TTS is OpenAI-direct only — OpenRouter does
    not proxy the audio.speech endpoint). Long inputs are split into
    chunks and the resulting WAV files are concatenated.
    """
    cache_path = _audio_cache_path(content)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    client = _get_openai_client()
    if client is None:
        raise ValueError(
            "Audio condition requires OPENAI_API_KEY (OpenRouter does not "
            "proxy audio.speech)."
        )

    chunks = _chunk_for_tts(content)
    wav_chunks: list[bytes] = []
    for chunk in chunks:
        resp = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=chunk,
            response_format=TTS_FORMAT,
            instructions=TTS_INSTRUCTIONS,
            timeout=180,
        )
        # OpenAI TTS returns WAVs with sentinel size fields — re-stamp
        # before concat so downstream decoders see real lengths.
        wav_chunks.append(_fix_wav_header(resp.content))

    audio_bytes = _concat_wavs(wav_chunks)

    os.makedirs(_AUDIO_CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(audio_bytes)
    return audio_bytes


def audio_b64(audio_bytes: bytes) -> str:
    """Raw audio bytes → base64 string for OpenAI input_audio.data."""
    return base64.b64encode(audio_bytes).decode("utf-8")
