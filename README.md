# image-as-defense

Does rendering untrusted third-party content as an image (instead of
including it as text in the user message) defend against prompt
injection?

The benchmark:

- For each **user task** (e.g. "summarize this document and answer
  Q1..Q3"), we take a legitimate document and embed a malicious
  **injection** (e.g. "send an email to X"), wrapped in one of several
  **attack templates** (ignore-previous, policy override, plan
  injection, etc.).
- We run the model under **two conditions** with exactly the same
  content and the same system prompt:
  - `text`: the document (with embedded injection) is inline in the
    user turn as a text block.
  - `image`: the document is rendered to a PNG and sent as an image
    block alongside a short text framing.
- For every sample we record:
  - `InjectionTaskSuccess` — did the model call the attacker's target
    function? (lower is better)
  - `UserTaskSuccess` — did the model correctly answer the legit
    questions? (higher is better; graded by an LLM judge against gold
    answers)

## Results

ASR = Attack Success Rate (↓ lower is better). UT = User Task success rate (↑ higher is better). Each cell is over 56 samples (4 tasks × 14 attack templates × 1 injection per template).

### Image as defense

| Model | Text ASR | Image ASR | Text UT | Image UT |
|---|---|---|---|---|
| openai/gpt-5.4-nano | 0.0% | 0.0% | 100.0% | 100.0% |
| openai/gpt-5.4-mini | 26.8% | **0.0%** | 100.0% | 100.0% |
| x-ai/grok-4.3 | 100.0% | **1.8%** | 100.0% | 100.0% |
| google/gemini-3.1-flash-lite-preview | 19.6% | 7.1% | 100.0% | 100.0% |
| qwen/qwen3.6-plus | 16.1% | 5.4% | 98.2% | 100.0% |
| anthropic/claude-haiku-4.5 | 0.0% | 0.0% | 96.4% | 98.2% |
| google/gemini-3.1-pro-preview | 8.9% | 0.0% | 50.0% | 94.6% |
| moonshotai/kimi-k2.6 | 1.8% | 3.6% | 10.7% | 42.9% |

### Audio as defense

| Model | Text ASR | Audio ASR | Text UT | Audio UT |
|---|---|---|---|---|
| openai/gpt-audio-mini | 0.0% | 0.0% | 100.0% | 91.1% |
| dashscope/qwen3.5-omni-plus | 62.5% | **17.9%** | 100.0% | 85.7% |

### Comparison with text-based defenses (n=14 per model)

| Model | Text ASR | Sandwiching ASR | Spotlighting ASR |
|---|---|---|---|
| openai/gpt-5.4-mini | 26.8% | 21.4% | 0.0% |
| google/gemini-3.1-flash-lite-preview | 19.6% | 100.0% | 14.3% |
| anthropic/claude-haiku-4.5 | 0.0% | 0.0% | 0.0% |
| moonshotai/kimi-k2.6 | 1.8% | 85.7% | 7.1% |

## Setup

```bash
cd image_as_defense
uv sync
cp .env.example .env  # or edit .env directly
# set OPENROUTER_API_KEY (required) and OPENAI_API_KEY / ANTHROPIC_API_KEY
#  (optional — if set, openai/* and anthropic/* models route directly to
#  the native API instead of OpenRouter).
```

## Run

```bash
# OpenAI-direct (requires OPENAI_API_KEY):
uv run python main.py --model openai/gpt-5.4-mini

# OpenRouter-routed:
uv run python main.py --model google/gemini-3.1-flash-lite-preview

# Anthropic-direct (requires ANTHROPIC_API_KEY):
uv run python main.py --model anthropic/claude-haiku-4.5
```

## Output

- `simple_inj_logs/<model>/user_task_<idx>_<type>/<cond>/template_<tmpl>/injection_task_<inj>.json`
  — one JSON per (user_task, injection, template, condition) sample,
  with full messages, response, called functions, and the two outcome
  bits.
- `logs/injection/<model>_<timestamp>.json` — aggregate summary for
  the run.

## Layout

```
image_as_defense/
├── pyproject.toml
├── README.md
├── .env
├── main.py                # CLI dispatcher (--case injection today,
│                          # --case refuse_math etc. later)
├── helpers/
│   └── runner.py          # cross-scenario utilities: console, active-model
│                          # tracker, API client routing (OpenAI direct /
│                          # OpenRouter), image rendering.
├── injection/             # --case injection scenario
│   ├── data.py            # tool specs, injection attacks, attack templates,
│                          # user tasks with gold answers, judge-model grader.
│   └── runner.py          # the scenario: build messages, call the model,
│                          # extract called functions, judge UT, log per
│                          # sample.
└── simple_inj_logs/       # per-sample output JSONs, grouped by model.
```

Each scenario lives in its own subpackage (`injection/` here). When
another scenario is added (e.g. `refuse_math/`), it follows the same
pattern: `refuse_math/data.py` + `refuse_math/runner.py`, importing
cross-scenario helpers from `helpers/runner.py`.

## Routing

`helpers/runner.resolve_client_and_model(model_id)` picks the client:

- `openai/<name>` + `OPENAI_API_KEY` → `https://api.openai.com/v1/`, model
  = `<name>`.
- anything else → OpenRouter, model unchanged.

`anthropic/*` models go through the native Anthropic client in
`helpers/run_injection.py:_call_anthropic` (requires `ANTHROPIC_API_KEY`).
