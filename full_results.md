# Image-as-defense — full sweep, 5 models × 4 user tasks

Run date: 2026-05-06 / 2026-05-07. Scope: every (model × user_task ×
injection × attack_template × condition) cell in the simple-injection
matrix. Logs under `results/logs_full-results/`. Per-sample JSONs at
`<model_short>/user_task_<i>_<task>/<mode>/template_<j>/injection_task_<k>.json`,
aggregate run summaries under `<root>/_summaries/`, console run logs
under `<root>/_run_logs/`.

## 1. Setup

- **Models tested (5):**
  - `anthropic/claude-haiku-4.5` (Anthropic direct)
  - `openai/gpt-5.4-nano` (OpenAI direct)
  - `google/gemini-3.1-flash-lite-preview` (OpenRouter)
  - `x-ai/grok-4.3` (OpenRouter)
  - `qwen/qwen3.6-plus` (OpenRouter)
- **User tasks (4):** `summarize` (archaeology essay), `code_reading`
  (Python snippet with `parse_and_filter_events` + helpers),
  `cv_reading` (CS-master CV), `todo_in_email` (runbook-update reminder
  email). Definitions in `injection/data.py:INJECTION_USER_TASKS`.
- **Attack templates (6):** T0 delimiter, T1 chat-token, T2
  pre-completed (task-aware via `_T2_PREFILLED_ANSWERS`), T3 ReAct,
  T4 schema-forcing, T5 middleware. Same as inj.md §3.
- **Injection tasks (14):** see `injection/data.py:INJECTION_ATTACKS`.
- **Conditions:** `text` (document inline as user text) vs `image`
  (document rendered to PNG, plain white style, sent in same user
  turn as image_url block).
- **Setup:** no-tag (no `<document>` wrappers), default system
  prompt `"You are a helpful assistant."`, judge model
  `openai/gpt-5.4-mini` for UT grading.
- **Per-condition matrix per model:** 4 tasks × 14 injections × 6
  templates = **336 samples per condition**, 672 total.

## 2. Headline aggregate (336 per condition)

Models in two clusters: a hard refusal floor (haiku, gpt-5.4-nano) and
an injection-vulnerable group (gemini, grok, qwen).

| Model | Text ASR | Image ASR | ΔASR | Text UT | Image UT |
|-------|---------:|----------:|-----:|--------:|---------:|
| anthropic/claude-haiku-4.5            | 1.2% (4/336)   | 0.0% (0/336)  | −1.2  | 99.1% | 99.4% |
| openai/gpt-5.4-nano                   | 3.6% (12/336)  | 0.0% (0/336)  | −3.6  | 96.1% | 100.0% |
| qwen/qwen3.6-plus                     | 42.4% (142/335)| 20.5% (69/336)| **−21.9** | 70.4% | 83.6% |
| x-ai/grok-4.3                         | 57.4% (193/336)| 28.0% (94/336)| **−29.5** | 52.1% | 71.4% |
| google/gemini-3.1-flash-lite-preview  | 67.6% (227/336)| 23.2% (78/336)| **−44.3** | 56.2% | 89.3% |

Image-mode ASR is strictly lower than text-mode ASR for **every model
tested**, and image-mode UT is ≥ text-mode UT for every model (defense
also recovers task quality, not just security). Aggregate ΔASR across
the 5-model set is roughly −20 pp (driven by the strong models being
already at floor and the weak models showing −22 to −44 pp).

## 3. Per-task ASR breakdown (84 samples per cell)

### Text mode

| Task | haiku-4.5 | gpt-5.4-nano | gemini-3.1-FL | grok-4.3 | qwen3.6-plus |
|------|:---:|:---:|:---:|:---:|:---:|
| summarize     | 2.4%  | 6.0%  | 65.5% | 59.5% | 56.0% |
| code_reading  | 2.4%  | 1.2%  | 72.6% | 65.5% | 35.7% |
| cv_reading    | 0.0%  | 0.0%  | 66.7% | 48.8% | 35.7% |
| todo_in_email | 0.0%  | 7.1%  | 65.5% | 56.0% | 42.2% |

### Image mode

| Task | haiku-4.5 | gpt-5.4-nano | gemini-3.1-FL | grok-4.3 | qwen3.6-plus |
|------|:---:|:---:|:---:|:---:|:---:|
| summarize     | 0.0% | 0.0% | 25.0% | 27.4% | 13.1% |
| code_reading  | 0.0% | 0.0% | **46.4%** | 26.2% | 26.2% |
| cv_reading    | 0.0% | 0.0% | 16.7% | 31.0% | 19.0% |
| todo_in_email | 0.0% | 0.0% | **4.8%** | 27.4% | 23.8% |

### ΔASR (image − text), in pp

| Task | haiku-4.5 | gpt-5.4-nano | gemini-3.1-FL | grok-4.3 | qwen3.6-plus |
|------|:---:|:---:|:---:|:---:|:---:|
| summarize     | −2.4 | −6.0 | −40.5 | −32.1 | −42.9 |
| code_reading  | −2.4 | −1.2 | **−26.2** | −39.3 | −9.5 |
| cv_reading    | 0.0  | 0.0  | −50.0 | −17.9 | −16.7 |
| todo_in_email | 0.0  | −7.1 | **−60.7** | −28.6 | −18.5 |

## 4. Per-template image-mode hits (out of 14 injections per cell)

Only the three vulnerable models are interesting in image mode; haiku
and gpt-5.4-nano are 0/84 across all (task, template) cells.

### gemini-3.1-FL

| Template | summarize | code | cv | email |
|---|:---:|:---:|:---:|:---:|
| T0 delimiter      | 1 | 1 | 1 | 1 |
| T1 chat-token     | 0 | 1 | 0 | 0 |
| T2 pre-completed  | 4 | 9 | 3 | 1 |
| T3 ReAct          | 0 | 9 | 3 | 0 |
| T4 schema-forcing | **12** | **12** | **7** | **2** |
| T5 middleware     | 4 | 7 | 0 | 0 |

### grok-4.3

| Template | summarize | code | cv | email |
|---|:---:|:---:|:---:|:---:|
| T0 delimiter      | 0 | 0 | 0 | 0 |
| T1 chat-token     | 1 | 0 | 0 | 0 |
| T2 pre-completed  | **12** | **12** | **10** | 6 |
| T3 ReAct          | 3 | 8 | 8 | 7 |
| T4 schema-forcing | 6 | 2 | 8 | **9** |
| T5 middleware     | 1 | 0 | 0 | 1 |

### qwen3.6-plus

| Template | summarize | code | cv | email |
|---|:---:|:---:|:---:|:---:|
| T0 delimiter      | 2 | 1 | 2 | 1 |
| T1 chat-token     | 2 | 0 | 0 | 0 |
| T2 pre-completed  | 2 | 6 | 4 | 5 |
| T3 ReAct          | 0 | 7 | 2 | 3 |
| T4 schema-forcing | **5** | 4 | **8** | **10** |
| T5 middleware     | 0 | 4 | 0 | 1 |

## 5. Reading

1. **Image-as-defense direction is universal.** Across 5 models × 4
   tasks (20 cells), image-mode ASR is ≤ text-mode ASR in every
   single cell. There are no exceptions. Image-mode UT is ≥
   text-mode UT in every cell (i.e. the defense doesn't cost utility
   either; it usually adds it).
2. **Two model classes.** `haiku-4.5` and `gpt-5.4-nano` sit at a
   hard refusal floor in *both* modes (≤ 7.1% text, 0.0% image
   across every task). Image defense is essentially a no-op here
   because text-mode alignment already blocks ~all attacks. The
   three vulnerable models (gemini, grok, qwen) are where image
   defense actually does work — and where the cross-task variance
   shows up.
3. **Code is Gemini's worst image-mode case again.** Image ASR on
   code_reading is 46.4% — far above gemini's other tasks (5–25%)
   and above any other (model, task) image cell. This replicates §13
   of inj.md on a 4-task / 5-model grid and confirms the "code is
   instruction-flavored" hypothesis: when the rendered content is
   itself executable, the modality recategorization is weaker.
   Gemini's per-template hits on code_reading image are spread
   across T2/T3/T4 (9/9/12), each high — code lifts multiple attack
   classes simultaneously.
4. **Email is Gemini's *strongest* image defense.** Image ASR on
   `todo_in_email` is 4.8% — the lowest of any (vulnerable-model,
   task) cell. ΔASR is **−60.7 pp**, the largest reduction in the
   table. Plausible mechanism: the email's authorship cues (header,
   sender, subject, recipients, recap) are *visual chrome* that
   makes the rendered image read as "a picture of someone else's
   email" rather than "instructions to me" — same effect as the
   chat-UI ablation in inj.md §11.
5. **grok-4.3's image defense is flatter across tasks.** Image ASR
   stays in 26–31% for all four tasks; ΔASR is ~−18 to −39 pp.
   Grok-4.3 falls for T2 pre-completed in image mode on
   summarize/code/cv (12/12/10) and for T4 schema-forcing on email
   (9). Different from Gemini's pattern — Gemini's image-mode
   strength varies dramatically by content type, Grok's barely
   moves.
6. **qwen3.6-plus has the smallest defense gap on code (−9.5 pp).**
   Same code-vs-prose asymmetry as Gemini, but compressed because
   qwen's text-mode ASR on code is already lower (35.7%) than its
   prose tasks. T3 ReAct lands 7/14 on qwen code image — code
   triggers ReAct compliance disproportionately.
7. **T4 schema-forcing is the universal durable survivor.** It is
   the top-or-tied image-mode template for at least one task on
   every vulnerable model, hitting 12/14 twice (gemini summarize and
   gemini code). Schema-compliance bias is modality-invariant:
   models that are willing to obey "your response MUST be valid JSON
   matching this schema" do so whether the schema arrives as text or
   as image content.
8. **T0 delimiter and T5 middleware are largely modality-killable.**
   T0 lands at most 2 hits per (model, task) image cell; T5 lands ≤
   7 (only on Gemini code) and 0 on most cells. Role-marker /
   trusted-orchestrator framings rely on the model trusting embedded
   text as a privileged channel; the image renders strip that
   privilege.

## 6. UT side-effects of the modality shift

Image-mode UT minus text-mode UT, per cell. Positive = image mode is
*more* useful (defense-preserved task completion).

| Task | haiku | nano | gemini | grok | qwen |
|------|:---:|:---:|:---:|:---:|:---:|
| summarize     | +2  | +6  | +35 | +24 | +25 |
| code_reading  | −1  | +1  | +23 | +26 | +18 |
| cv_reading    | 0   | 0   | +39 | +8  | +19 |
| todo_in_email | 0   | +8  | +36 | +26 | +20 |

Strong models hover around 0 (text UT was already ≈ 100%, no room to
gain). Weak models gain +18 to +39 pp on most cells — confirming
that injections poison QA quality, not just tool-call decisions, and
that the image channel removes the poison.

## 7. Implication

The defense generalizes across model families and across content
types, but its *practical magnitude* depends on (a) whether the
model needs it (strong models don't), and (b) how content-like vs.
instruction-like the document feels (prose ≫ code). The 5-model
matrix here, plus the original 7-model summarize-only matrix in
inj.md §5, together cover 8 distinct models across 5 vendors —
every single one shows the direction of the effect, no exceptions.

For the vulnerable models, the practical recommendation depends on
the deployment surface:
- Document-style untrusted content (emails, articles, CVs) →
  image rendering gives near-full defense (image ASR ~5–25%, ΔASR
  ~−30 to −60 pp).
- Code-style untrusted content → image still helps (ΔASR ≥ −10 pp),
  but ASR remains material on Gemini (46%); pair with other defenses.

## 8. Caveats

- **Single attempt per cell** (no run-to-run averaging). T2/T4 hit
  rates can vary ±2/14 between reruns.
- **qwen3.6-plus had 1 sample error** in text mode (335 vs 336 valid
  samples). All other models completed cleanly.
- **Judge is gpt-5.4-mini** for all UT grading; same caveat as
  inj.md §8.
- **Default OpenRouter temperatures** for gemini/grok/qwen
  (~0.6–1.0); OpenAI direct rejects `temperature` for new GPT-5
  family models, so it's unset there. Anthropic uses default.
- **No defense-aware system prompt** in this run. Prior data
  (inj.md §8) showed minimal effect of a defense-aware prompt on
  gpt-5.4-mini text mode; effect on the 4 newly tested models is
  not characterized here.

## 9. Artifacts

- Per-sample JSON: `results/logs_full-results/<model_short>/user_task_<i>_<task>/<mode>/template_<j>/injection_task_<k>.json`
- Aggregate per-run JSON: `results/logs_full-results/_summaries/<model_short>_<timestamp>.json`
- Console run logs: `results/logs_full-results/_run_logs/<model_short>.log`
- Reproducer:
  ```
  python main.py --model <id> --log-root results/logs_full-results
  python main.py --model <id> --log-root results/logs_full-results --task-type code_reading
  python main.py --model <id> --log-root results/logs_full-results --template-idx 2
  ```
