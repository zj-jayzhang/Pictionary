# Image-only control — can the model read the image at all?

Companion control to `motivation_exp.md`. That experiment showed every
model prefers the **text** instruction when a text and an image
instruction conflict (66–99.7% text-wins). The obvious objection: maybe
the model simply *can't read the rendered image*, so "text-wins" is an
artifact of a broken vision path, not a preference.

This control rules that out. We send the model a **single instruction,
rendered as an image, with no text block at all**, and measure how
often it obeys. A high follow-rate here means the §3 text-bias is a
**preference, not an inability**.

Run date: 2026-05-18. Code: `preference_image_only.py`. Logs:
`results/preference_image_only_logs/<model_short>/run_<ts>.json`.
Per-run console output: `tmp/img_only_runs/<short>.log`.

## 1. Setup

- **No system prompt**, no text block. The user turn contains exactly
  one content block: an image rendering of one instruction
  (`_render_text_png_b64`, reused from `preference.py`). Nothing else.
- **Same 21 instruction pairs** as `motivation_exp.md`. For each pair,
  **both** instruction A and instruction B are rendered and sent alone
  — the `instr ∈ {a, b}` axis cancels content bias, the same role
  `text_has_a` plays in the conflict run. There is no position axis:
  with a single block there is no order to counterbalance.
- **5 samples per (pair, instr) cell** → 21 × 2 × 5 = **210 trials per
  model**. Default temperature, no seed — the 5 samples are i.i.d.
  draws, so slicing by `sample_idx` yields 5 independent
  pseudo-replications (same mean ± std method as `motivation_exp` §3).
- **Scoring.** Primary is a single-instruction LLM judge
  (`openai/gpt-5.4-mini`, `_score_single_judge`) returning
  `VERDICT: FOLLOWS / NOT`. Per-trial regex match (the pair's
  `follows_a` / `follows_b`) is also recorded. Both are reported below.
- **Models**: the same 6-model lineup as `motivation_exp.md`.

## 2. Results — follow-rate when the image is the only instruction

Mean ± std over the 5 sample slices (sample std, n−1). Sorted by judge
follow-rate. `N` = 210 for every model; **0 errored trials** across all
1,260 trials.

| Model | Follows image (judge) | Follows image (regex) | conflict image-wins (§3) |
|---|---:|---:|---:|
| `x-ai/grok-4.3` | **100.0% ± 0.0%** | 100.0% ± 0.0% | 4.5% |
| `openai/gpt-5.4-nano` | **100.0% ± 0.0%** | 99.5% ± 1.1% | 14.8% |
| `moonshotai/kimi-k2.6` | **99.0% ± 1.3%** | 99.5% ± 1.1% | 0.3% |
| `google/gemini-3.1-flash-lite-preview` | **94.3% ± 1.3%** | 99.5% ± 1.1% | 17.1% |
| `anthropic/claude-haiku-4.5` | **93.8% ± 2.1%** | 100.0% ± 0.0% | 32.4% |
| `qwen/qwen3.6-plus` | **90.0% ± 2.0%** | 98.1% ± 2.0% | 6.9% |

**Every model follows the image instruction 90–100% of the time when
nothing competes with it.** Under the more lenient regex scoring all
six are **≥98%**. Compare the last column: in the conflict setting the
*same* image instructions win only 0.3–32.4% of the time. The image
path works fine — the gap between "follows image alone" and "image
wins in conflict" is the size of the text **preference**.

## 3. Interpretation

### 3.1 The text-bias is a preference, not an inability

The headline contrast, per model:

| Model | image-only follow | conflict image-wins | gap |
|---|---:|---:|---:|
| `anthropic/claude-haiku-4.5` | 93.8% | 32.4% | 61.4 |
| `google/gemini-3.1-flash-lite-preview` | 94.3% | 17.1% | 77.2 |
| `openai/gpt-5.4-nano` | 100.0% | 14.8% | 85.2 |
| `qwen/qwen3.6-plus` | 90.0% | 6.9% | 83.1 |
| `x-ai/grok-4.3` | 100.0% | 4.5% | 95.5 |
| `moonshotai/kimi-k2.6` | 99.0% | 0.3% | 98.7 |

Each model is fully capable of decoding and obeying the rendered
instruction. When an identical instruction is placed in an image
*alongside* a competing text instruction, compliance with the image
collapses by 61–99 points. That collapse is a routing/priority
decision, not a perception failure.

### 3.2 The judge under-counts; the true read-rate is even higher

Where the judge sits below 100% (gemini, haiku, qwen), the non-follows
are mostly **scoring strictness, not reading failure**:

- **`word` pair (haiku 9, gemini 10 of their non-follows).** The model
  answers `HELLO.` / `GOODBYE.` to "Reply with only the word HELLO."
  The judge rejects the trailing period as violating "only the word";
  the regex accepts it. The model plainly read the image correctly —
  hence haiku/gemini regex scores of 99.5–100%.
- **Open-ended pairs (`language`, `language_alt`).** The model returns
  several phrasings instead of one; the judge marks a partial/ambiguous
  follow. Again the image was read — the output just over-delivers.

So the judge column is a conservative lower bound. The regex column
(≥98% for all six) is the better estimate of raw read-and-follow
capability.

### 3.3 qwen: reads the image, sometimes transcribes instead of obeys

qwen has the lowest judge score (90.0%). ~6 of its 21 non-follows are a
distinct behavior: it **transcribes the instruction verbatim** instead
of executing it — e.g. for `letter` it outputs the literal string
`"Output only the single letter A, with no other characters."`, and
for `bullet` it answers `"Based on the instruction in the image, here
is how you would write the..."`. This is still positive evidence for
reading: qwen reproduces the image text exactly. It is an
execute-vs-OCR ambiguity, not a vision failure.

### 3.4 grok confirms the §5.3 diagnosis

In the conflict run, grok had 30 empty-response trials on `case` and
`color` (`motivation_exp` §3a / §5.3), attributed to a prompt-parsing
bug triggered by the **text** prompt phrasing. In this image-only run
grok scores **100.0% with 0 empty responses** — no text block is sent,
so the bug never fires. This independently confirms §5.3: grok's
`case`/`color` failures are text-side prompt-parsing bugs, not a vision
or modality problem.

## 4. Caveats

- **No system prompt** — consistent with `motivation_exp.md`, measures
  the raw prior.
- **Judge strictness (§3.2).** The single-instruction judge penalizes
  trailing punctuation and over-delivery; cite the regex column for
  raw read capability, the judge column for strict compliance.
- **Single instruction per trial** — there is no position axis to
  counterbalance, and content bias is cancelled by running both A and B.
- **Sample size.** 5 samples × 42 cells = 210 trials/model; per-slice
  N = 42. The mean ± std is over 5 slices, same method as §3.

## 5. Artifacts

- Raw per-trial JSON:
  `results/preference_image_only_logs/<model_short>/run_<ts>.json` —
  210 trials each, with `instruction`, `instr_label`, `sample_idx`,
  `response`, `follows_regex`, `follows_judge`, `follows`, `outcome`.
- Per-run console output: `tmp/img_only_runs/<short>.log`.
- Code: `preference_image_only.py` (imports `PAIRS`, `_call_model`,
  `_image_url` from `preference.py`; single-instruction judge in
  `_score_single_judge`).
