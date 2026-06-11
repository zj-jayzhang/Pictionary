# Pictionary

This is the code for testing text vs. image ASR on DirectInject with 7 attack methods.

Does rendering untrusted third-party content as an image (or audio) defend against prompt injection?

The benchmark tests three delivery modalities for the same document content:

- `text` — document inline as a text block (baseline)
- `image` — document rendered to a PNG and sent as an image block
- `audio` — document converted to speech and sent as an audio block (requires audio-capable model)

For each **user task** × **injection** × **attack template**, the model is run under the selected conditions with an identical system prompt. Two outcomes are recorded per sample:

- `InjectionTaskSuccess` — did the model execute the attacker's injected function call? (↓ lower is better)
- `UserTaskSuccess` — did the model correctly complete the legitimate task? (↑ higher is better; graded by an LLM judge)

## Setup

```bash
uv sync
# copy .env and fill in your API keys
cp .env .env.local   # or edit .env directly
# OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY
```

## Run

```bash
# default: text vs image, no defense
uv run python main.py --model openai/gpt-5.4-mini

# audio modality (requires audio-capable model)
uv run python main.py --model openai/gpt-audio-mini --conditions text,audio

# image render style variants
uv run python main.py --model openai/gpt-5.4-mini --render-style blackboard
# choices: plain (default) | chat | google | blackboard

# text-based defense baselines (text condition only)
uv run python main.py --model openai/gpt-5.4-mini --defense spotlighting
# choices: none (default) | spotlighting | sandwiching | secalign

# run a single attack template (0-indexed)
uv run python main.py --model openai/gpt-5.4-mini --template-idx 2
```

## Output

- `exp_runs/simple_inj_logs/<model>/user_task_<idx>_<type>/<cond>/template_<tmpl>/injection_task_<inj>.json`
  — one JSON per (user_task, injection, template, condition) sample, with full messages, model response, called functions, and the two outcome bits.
- `exp_runs/logs/injection/<model>_<timestamp>.json` — aggregate summary for the run.

Pass `--log-root <dir>` to redirect both logs under a custom directory.

## Layout

```
Pictionary/
├── pyproject.toml
├── README.md
├── .env
├── main.py                # CLI entry point
├── helpers/
│   └── runner.py          # shared utilities: console, API client routing
│                          # (OpenAI / Anthropic / OpenRouter), image rendering
└── injection/             # injection scenario
    ├── data.py            # tool specs, attack templates, user tasks,
    │                      # gold answers, LLM judge
    └── runner.py          # runs the matrix, logs per-sample JSON,
                           # writes aggregate summary
```

## Routing

- `openai/<name>` + `OPENAI_API_KEY` → OpenAI API directly
- `anthropic/<name>` + `ANTHROPIC_API_KEY` → Anthropic API directly
- `dashscope/<name>` → DashScope OpenAI-compatible endpoint (Alibaba/Qwen)
- anything else → OpenRouter (requires `OPENROUTER_API_KEY`)
