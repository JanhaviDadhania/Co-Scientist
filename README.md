# AI co-scientist

An open source re-implementation of Google's **AI co-scientist** ([Gottweis et al., *Nature*, 2026](https://www.nature.com/articles/s41586-026-10644-y); [research blog, 2025](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/)) — a multi-agent system that takes a natural-language research goal and produces a tournament-ranked **research overview** of novel hypotheses.

The agent roster, prompts, and control flow follow the paper. Source materials that were used to instruct the coding agent (Claude Code) is included with the repo:

- [`reference/8 Pseudocode of Co-Scientist agents`](reference/) — the supplementary pseudocode for Supervisor, Generation, Reflection, Ranking, Evolution, Proximity, Meta-review.
- [`reference/9 Prompts for the specialized agents in .md`](reference/) — the per-agent prompts from the paper's supplement, used verbatim (modulo Jinja interpolation) in [`config/prompts/`](config/prompts/).
- [`reference/AICoScientist-*.png`](reference/) — the architecture and component diagrams from the paper.

The agents:

- **Generation** — proposes hypotheses via literature review and simulated scientific debate.
- **Reflection** — reviews hypotheses for novelty, correctness, and testability; deep-verifies the underlying assumptions.
- **Ranking** — runs an Elo tournament with simulated debates between hypotheses.
- **Evolution** — combines, simplifies, makes more feasible, or out-of-box-reimagines top-ranked hypotheses.
- **Proximity** — embeds and clusters hypotheses to drive dedup and informative tournament pairings.
- **Meta-review** — synthesizes system-wide feedback and the final research overview.

A **Supervisor** parses the goal into a research plan and schedules agent tasks through a durable SQLite-backed queue with bounded concurrency.

This is an independent re-implementation in Python on top of pluggable LLM provider SDKs — not affiliated with Google or the paper's authors.

> [`docs/BENCH_RESULTS.md`](docs/BENCH_RESULTS.md) — every cross-model bench ever run on this code, with per-candidate Elo, every hypothesis produced, gold-set hits, and direct file pointers. Auto-generated from the bench DB.

## Contents

- [Architecture](#architecture)
- [Install](#install)
- [Initialize](#initialize)
- [Run a research session](#run-a-research-session)
- [LLM provider](#llm-provider)
- [Configuration](#configuration)
- [Bench: compare models head-to-head](#bench-compare-models-head-to-head)
- [Repository layout](#repository-layout)

## Architecture

```
                       co-scientist run "<goal>"
                                  │
                                  ▼
            ┌──────────────────────────────────────┐
            │            Supervisor                │  durable task queue (SQLite)
            │  • parse_goal → ResearchPlan         │  bounded concurrency
            │  • enqueue initial Generation tasks  │  lease + dead-letter + resume
            │  • main loop: claim → run → follow-up│  termination: BUDGET / WALL_CLOCK
            │  • decide_next_steps when idle       │              / ELO_STABLE / IDLE / EXTERNAL
            │  • finalize: meta-review overview    │
            └──────────────────────────────────────┘
                                  │  tasks
            ┌─────────────────────┼─────────────────────────────┐
            ▼                     ▼                             ▼
   ┌──────────────┐      ┌──────────────┐              ┌──────────────┐
   │  Generation  │ hyp  │  Reflection  │ review       │   Ranking    │
   │  literature  │─────►│  full +      │─────────────►│ pairwise vs  │──► Elo
   │  + debate    │      │  verification│              │   debate     │
   └──────────────┘      └──────────────┘              └──────────────┘
            ▲                     ▲                             │
            │                     │ informative pairings        ▼
   ┌──────────────┐      ┌──────────────┐              ┌──────────────┐
   │  Evolution   │◄─────│ Meta-review  │              │  Proximity   │
   │ combine /    │ feed │ system fdbk  │              │ FAISS embed  │
   │ simplify /   │ back │ + final      │              │ + cluster /  │
   │ feasibility /│      │ overview     │              │ dedup        │
   │ out_of_box   │      └──────────────┘              └──────────────┘
   └──────────────┘
            │
            ▼
       new hypotheses re-enter the cycle


  Shared infrastructure
  ─────────────────────
  • LLMProvider  ─ anthropic / openai / openrouter / gemini / groq /
                   together / mistral / ollama / openai_compatible
  • ToolRegistry ─ web_fetch + pubmed_search / arxiv_search / europe_pmc_search;
                   web_search auto-registered iff TAVILY/BRAVE key set;
                   science-skills discovered via SKILL.md frontmatter
  • TokenBudget  ─ per-agent shares + global cap; reservation released on retry
  • EventBus     ─ in-memory fan-out to SSE for the live web UI
  • FaissStore   ─ IndexFlatIP per session, asyncio-locked, atomic save/load;
                   Voyage → OpenAI → hash-fallback embedder chain
  • SQLite       ─ sessions / hypotheses / reviews / tournament_matches /
                   elo_journal / tasks / transcripts / system_feedback /
                   embeddings_meta / spans / events / bench_* (15 tables;
                   WAL, busy_timeout, idempotent migration runner)
```

## Install

```bash
# Recommended: Python 3.11–3.13 (FAISS wheel availability)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# fill in the API key for whichever LLM provider you'll use (see below).
```

## Initialize

```bash
co-scientist init
co-scientist list
```

`init` creates `data/` (artifacts, vectors, logs) and applies migrations to `data/co_scientist.db`. The output prints which LLM provider it sees configured and whether its API key is set.

## Run a research session

```bash
co-scientist run "Identify hypotheses about microbiome-driven inflammation" \
  --n 3 --budget-usd 2.0 --wall-clock 600
```

This kicks off Generation → Reflection → Ranking → Evolution → Meta-review under the configured LLM provider. The Supervisor schedules tasks, the Elo tournament refines a leaderboard, and the final research overview is written to `data/artifacts/<session_id>/final/overview.md`.

### Seeding Generation with a pre-curated field survey (`--field-context`)

When you already have a literature survey for the field (e.g. a STORM-generated wiki page), point Generation at it so the agent isn't restarting from cold search every time:

```bash
co-scientist run "test SAEs as a way to interpret neural networks" \
  --field-context path/to/sae_survey.md \
  --n 3 --budget-usd 2.0 --wall-clock 900
```

What this does:
- Loads the file once at session start and stores its text on `cfg.run.field_context`.
- Prepends its contents to the user_blocks of every Generation call, wrapped in instructions that mark the survey as **required reading** and frame the literature search tools (`pubmed_search`, `arxiv_search`, `europe_pmc_search`, `web_search`, `web_fetch`) as **fallback-only** — they should be used to fill specific gaps the survey leaves open, not to duplicate work the survey already did.
- Does not affect Reflection / Ranking / Evolution / Metareview — they operate on the hypotheses Generation produces.

Caveats: the survey is re-sent on every tool-loop iteration of every Generation call (no cross-call caching on `claude_code`). Long surveys (>10k tokens) noticeably slow each call. If your survey is very long, trim to abstract + section TOC + key references before passing it in.

```bash
co-scientist serve            # FastAPI + htmx + SSE dashboard at localhost:7878
co-scientist report <id>      # print the final overview
co-scientist status <id>      # session metadata + counts
co-scientist pause <id> | resume <id> | abort <id>
co-scientist feedback <id> --kind directive --text "focus on metabolic pathways"
co-scientist estimate         # pre-flight cost estimate; warns if > 1.2× budget
co-scientist eval [agent]     # run the rubric eval bundle (offline mode optional)
co-scientist tools list       # show every registered tool the agents can call
```

## LLM provider

The agents are provider-agnostic — every agent talks to one LLM provider per session, picked in [`config/default.toml`](config/default.toml) (override with your own `co-scientist.toml`). Any of the providers below works; pick whichever you have a key for.

Config is **deep-merged** over [`config/default.toml`](config/default.toml), whose `[models]` defaults are Claude model ids. So if you switch `provider` away from `anthropic`, override **every** key in `[models]` — any key you leave out keeps its Claude default and will be sent to your new provider, which will reject it. Fill in model ids your chosen provider exposes (see the provider table below for examples per vendor):

```toml
[llm]
# Pick one. See the provider table below.
provider = "openai"   # anthropic | openai | openrouter | gemini | google | groq | together | mistral | ollama | openai_compatible

[models]
# Override ALL of these with model ids from your chosen provider.
# (Two tiers are enough: a stronger model for reasoning-heavy agents and a
#  cheaper one for the rest — set them to whatever your provider offers.)
parse_goal          = "<cheap-model>"
generation          = "<strong-model>"
reflection          = "<strong-model>"
evolution           = "<strong-model>"
ranking_pairwise    = "<cheap-model>"
ranking_debate      = "<strong-model>"
ranking_priority    = "<strong-model>"
metareview_feedback = "<cheap-model>"
metareview_final    = "<strong-model>"
classifier          = "<cheap-model>"
judge               = "<cheap-model>"
```

Providers are listed alphabetically — none is preferred; pick whichever you have a key for.

| provider              | Endpoint                                                | API-key env var         | Example models                                            |
| --------------------- | ------------------------------------------------------- | ----------------------- | --------------------------------------------------------- |
| `anthropic`           | api.anthropic.com                                       | `ANTHROPIC_API_KEY`     | `claude-opus-4-7`, `claude-sonnet-4-6`                    |
| `claude_code`         | local `claude -p` subprocess (Claude Code CLI)          | *(none — uses your Claude Code subscription)* | `sonnet`, `opus`, `haiku` (aliases) or full ids |
| `gemini` / `google`   | generativelanguage.googleapis.com (OpenAI-compat)       | `GEMINI_API_KEY`        | `gemini-2.5-pro`, `gemini-2.5-flash`                      |
| `groq`                | api.groq.com                                            | `GROQ_API_KEY`          | `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`           |
| `mistral`             | api.mistral.ai                                          | `MISTRAL_API_KEY`       | `mistral-large-latest`, `codestral-latest`                |
| `ollama`              | localhost:11434 — local models                          | *(none)*                | `llama3.3:70b`, `qwen2.5:32b`                             |
| `openai`              | api.openai.com                                          | `OPENAI_API_KEY`        | `gpt-5`, `gpt-4o`, `o3-mini`                              |
| `openai_compatible`   | Anything else; set `[llm.openai] base_url` explicitly   | `OPENAI_API_KEY`        | depends                                                   |
| `openrouter`          | openrouter.ai — 200+ models from every major vendor     | `OPENROUTER_API_KEY`    | `openai/gpt-5`, `google/gemini-2.5-pro`, `anthropic/claude-3.5-sonnet`, `meta-llama/llama-3.3-70b-instruct` |
| `together`            | api.together.xyz                                        | `TOGETHER_API_KEY`      | `meta-llama/Llama-3.3-70B-Instruct-Turbo`                 |

> Key precedence: for every OpenAI-compatible preset (`openrouter`, `gemini`, `groq`, `together`, `mistral`, `ollama`), `OPENAI_API_KEY` is used **first** if it's set, and the provider-specific var above is only the fallback. So if you have a stray `OPENAI_API_KEY` in your environment it will be sent to the preset's endpoint (and rejected) — unset it, or set only the provider's own key, when using a preset.

Mixing vendors per session requires picking the provider once; for multi-vendor routing in a single session, use `provider = "openrouter"` and let OpenRouter dispatch upstream per model:

```toml
[llm]
provider = "openrouter"
[llm.openrouter]
referer = "https://your-app.example.com"   # optional, for catalog attribution
title   = "My Co-Scientist"

[models]
generation         = "openai/gpt-5"
reflection         = "anthropic/claude-3.5-sonnet"
ranking_pairwise   = "google/gemini-2.5-flash"
metareview_final   = "meta-llama/llama-3.3-70b-instruct"
```

Any per-agent model can point at any vendor — the example above just mixes four. Use whatever combination you prefer.

Cost is estimated via `co_scientist/llm/routing.py`'s `PRICE_TABLE`; unknown models match a family-hint (flash / mini / opus / sonnet / gemini / llama / mistral) so brand-new previews price sensibly. Tighten `[run] budget_usd` if running on a new model you haven't sanity-checked.

**Provider feature support.** Tool / function calling is **required** — the agent pipeline is built on it, so a provider (or `openai_compatible` endpoint) that can't do function calling won't work. The other three rows are optional vendor-specific accelerators: when a provider doesn't support one, it's transparently skipped, never an error.

| Feature                     | `anthropic` | everything else (OpenAI + all OpenAI-compatible providers) |
| --------------------------- | ----------- | ---------------------------------------------------------- |
| Tool / function call *(required)* | ✅    | ✅ native OpenAI; on other endpoints it must be supported or the run fails |
| Extended reasoning          | ✅ via `thinking` budgets | ✅ via `reasoning_effort`, **only for reasoning models** — the model id must start with `o1`/`o3`/`o4` or contain `reasoning`; for any other model (e.g. `gpt-4o`) the thinking budget is dropped |
| Prompt-cache breakpoints    | ✅          | ❌ (stripped before sending)                               |
| Batch API (50%-off ranking) | ✅          | ❌ (Anthropic-only; other providers run all matches synchronously) |

> Note: the reasoning-model check is a name heuristic ([`openai_client.py`](co_scientist/llm/openai_client.py) `_is_reasoning_model`). Newer reasoning-capable models whose ids don't match the pattern (e.g. `gpt-5`) won't get `reasoning_effort` until the heuristic is updated — they still work, just without an explicit reasoning budget.

### Using the local Claude Code CLI (`claude_code` provider)

`provider = "claude_code"` routes every LLM call through `claude -p` instead of an API. The benefit is that you pay zero per-call API cost — calls go through your Claude Code subscription's quota. The cost is that you give up the things only the Anthropic API exposes (real tool-use mechanism, prompt caching, batch API, extended-thinking budgets).

```toml
[llm]
provider = "claude_code"

[run]
concurrency = 1            # serialize subprocesses to avoid Claude Code throttling
budget_usd = 100.0         # effectively disabled — wall-clock terminates instead

[models]                   # Claude Code aliases or full ids
parse_goal          = "sonnet"
generation          = "sonnet"
reflection          = "sonnet"
evolution           = "sonnet"
ranking_pairwise    = "sonnet"
ranking_debate      = "sonnet"
ranking_priority    = "sonnet"
metareview_feedback = "sonnet"
metareview_final    = "sonnet"
classifier          = "haiku"
judge               = "sonnet"

[thinking]                 # claude -p has no `thinking` budget knob — zero them
generation_literature   = 0
generation_debate       = 0
reflection_full         = 0
reflection_verification = 0
reflection_observation  = 0
ranking_pairwise        = 0
ranking_debate          = 0
evolution_combine       = 0
evolution_out_of_box    = 0
evolution_feasibility   = 0
evolution_simplify      = 0
metareview_feedback     = 0
metareview_final        = 0
```

Notes on the trade-offs:

- **Tool calls are simulated, not enforced.** The Anthropic API guarantees structured `tool_use` output when you pass `tool_choice`. `claude -p` has no such mechanism — the shim prompts the model to emit `{"tool_calls": [...]}` JSON in its stdout and parses it. Most of the time this works; sometimes the model returns prose and the supervisor falls back gracefully (you'll see warnings like `parse_goal_no_record`).
- **Subprocess cold-start is ~3–5s per call.** Concurrent `claude -p` invocations also appear to throttle against each other on the same Claude Code account, so concurrency=1 is the safe default. Per-call timeout defaults to 400s (`CLAUDE_CODE_TIMEOUT_S` env var to override).
- **Cost accounting still runs.** `cost_usd` in transcripts is the rough API-equivalent estimate using the existing PRICE_TABLE — useful as a proxy for "what this would have cost on the API", not what your subscription was actually billed.

#### The Write-tool safety net for forced tool calls

For *forced* tool calls (`tool_choice = {"type": "tool", "name": X}` — used for `parse_goal`'s `record_research_plan`, and for the last-iteration forced terminal tool in the Generation / Reflection / Evolution loops), the shim opens a per-call tempfile at `/tmp/co_scientist_call_<uuid>.json` and instructs the model in the system prompt to *use the Claude Code Write tool* to drop the JSON object there. Write is a first-class Claude Code action — much more reliably triggered than coercing the model into emitting raw JSON on stdout.

To make Write usable in non-interactive mode, the shim invokes `claude -p` with `--tools Write` and `--permission-mode bypassPermissions`. After the subprocess exits, the shim reads the tempfile, parses it as JSON, and synthesizes an Anthropic-shaped `tool_use` block from the contents. If the file is missing or invalid, the shim falls back to the original stdout-regex path; if that also fails, you get the usual graceful fallback (warning + bare structure for `parse_goal`, or a task retry for terminal tools in Generation / Reflection / Evolution).

Implications:
- The subprocess can write anywhere on disk during a forced call (limited to the Write tool, no Bash / Edit / Read). The model has only been observed to write to the exact tempfile path it's instructed to use, but the permission is broad.
- Auto-tool calls (intermediate iterations where the model picks between search tools and `record_*`) still use the stdout-JSON path — those don't engage the Write fallback.

#### What to expect operationally

From empirical runs on `claude_code` with `--field-context` and `concurrency=1`:

| Run shape | Hypotheses produced | parse_goal | Metareview overview |
| --- | --- | --- | --- |
| No survey, n=3 initial | 0–1 of 3 | falls back to bare plan | usually runs |
| `--field-context` survey only | 2–3 of 3 | falls back to bare plan | usually runs |
| `--field-context` + Write-tool safety net + `--wall-clock 1200` | **3 of 3** | **engages cleanly** | **runs cleanly** |

The dominant failure mode without the Write-tool path is `"Generation did not call record_hypothesis"` — the model exhausts all 8 tool-loop iterations on literature search and never commits. `--field-context` is the single most effective mitigation (the curated survey removes the cold-start search incentive). The Write-tool safety net then catches the remaining forced-tool calls (notably `parse_goal`) that prompt-only stdout forcing misses.

Give the run enough wall-clock for the metareview to fire after the main loop exits. `--wall-clock 900` was tight; `1200`+ is safer.

#### Sample run — fully clean end-to-end

A reference run with all three knobs (`claude_code`, `--field-context`, Write-tool path, `--wall-clock 1200`, `concurrency = 1`) on the goal:

> *"Use sparse autoencoders to detect and reduce hallucinations in large language models"*

with the STORM-generated SAE survey as `--field-context`. Outcomes:

- 3 of 3 initial Generation tasks committed a `record_hypothesis`.
- `parse_goal` produced a structured `record_research_plan` via the Write-tool tempfile path on the first try — no fallback warning.
- Metareview `final` ran inside the wall-clock budget.
- Reflection started (and timed out — first time the system reached that stage).
- No subprocess crashes; clean exit.

Artifacts committed to the repo for reference:

- [`docs/sample_runs/hallucination_detection/overview.md`](docs/sample_runs/hallucination_detection/overview.md) — the metareview's research overview. Identifies two non-mutually-exclusive directions (targeted SAE feature suppression vs. epistemic-superposition cross-layer consistency), gives falsification criteria for each, and proposes running them in parallel as an adjudication design.
- [`docs/sample_runs/hallucination_detection/hyp_fb0242ebb9fccae6.json`](docs/sample_runs/hallucination_detection/hyp_fb0242ebb9fccae6.json) — Hallucination-Subspace Suppression hypothesis.
- [`docs/sample_runs/hallucination_detection/hyp_77724168529242d6.json`](docs/sample_runs/hallucination_detection/hyp_77724168529242d6.json) — Epistemic Superposition Monitoring hypothesis.
- [`docs/sample_runs/hallucination_detection/hyp_6d18246faaab02b3.json`](docs/sample_runs/hallucination_detection/hyp_6d18246faaab02b3.json) — Epistemic-Mismatch SAE Feature Suppression hypothesis.

Reproduce with:

```bash
co-scientist run "Use sparse autoencoders to detect and reduce hallucinations in large language models" \
  --field-context path/to/sae_survey.md \
  --n 3 --budget-usd 100 --wall-clock 1200
```

### Embeddings provider

`[embeddings]` accepts `voyage` (default, requires `VOYAGE_API_KEY`), `openai` (requires `OPENAI_API_KEY`), `gemini` (requires `GEMINI_API_KEY` — free tier on AI Studio works), or `hash` (deterministic local fallback; works offline but only catches near-duplicate hypotheses, not semantic similarity). The auto-fallback chain inside `make_embedder()` is Voyage → OpenAI → hash for `provider = "voyage"`, and Gemini → OpenAI → hash for `provider = "gemini"`.

```toml
[embeddings]
provider = "gemini"
model = "gemini-embedding-001"   # 3072-dim native; truncate via dim below (Matryoshka)
dim = 768
```

> Note: `text-embedding-004` is *only* exposed on Gemini's native API, not on the OpenAI-compatible endpoint co-scientist uses. Use `gemini-embedding-001` here. The client passes `dimensions = dim` for gemini-embedding-001 to get a Matryoshka-truncated vector.

## Configuration

Layered: [`config/default.toml`](config/default.toml) → `~/.co-scientist/config.toml` → `./co-scientist.toml` → `--config <path>`. Secrets come from environment only (see [`.env.example`](.env.example)).

## Bench: compare models head-to-head

`co-scientist bench` runs the same goal under N different `(provider, model)` configurations and ranks them via a single shared Elo tournament. Each candidate independently generates hypotheses; then every candidate-pair plays `--matches` head-to-head debates, judged by ONE fixed judge model (picked separately so no candidate scores its own work).

> **For live numbers** — per-candidate Elo, the actual hypotheses each model proposed, gold-set hits, and what the data showed — see [`docs/BENCH_RESULTS.md`](docs/BENCH_RESULTS.md). It includes a headline-findings section at the top so you don't have to scroll through every bench.

### Presets

| `--preset`               | What it does |
| ---                      | --- |
| `paper`                  | Co-Scientist paper baselines (Gemini 2 Flash Thinking, Gemini 2 Pro, OpenAI o1, Claude Haiku) via OpenRouter, head-to-head Elo only |
| `paper-aml`              | Same candidates + the paper's AML drug-repurposing goal + gold-set recall scoring (defaults to the strict top-3 set: Nanvuranlat / KIRA6 / Leflunomide) |
| `paper-aml-vs-raw`       | `paper-aml` but each model runs **both** in the full pipeline AND as a single raw LM call — isolates the multi-agent harness's value-add |
| `frontier-aml-vs-raw`    | Same pipeline-vs-raw setup but with current frontier models (Claude Opus 4.7, GPT-5, Gemini 3 Pro / Flash) |

```bash
# Reproduce the paper preference-ranking comparison:
co-scientist bench --preset paper --budget-per-candidate 1.5

# Score against the paper's AML drug picks:
co-scientist bench --preset paper-aml --n 3 --matches 2

# Compare multi-agent pipeline vs raw model call on the same goal
# (--budget-per-candidate defaults to 3.0; frontier models need it):
co-scientist bench --preset paper-aml-vs-raw --n 1

# Current frontier models, pipeline vs raw:
co-scientist bench --preset frontier-aml-vs-raw --n 1
```

### Pipeline vs raw LM (one model, isolated)

The `--preset *-vs-raw` presets pit each model's **full co-scientist Generation pipeline** (literature tools + tool loop + dedup + `record_hypothesis`) against a **single raw LM call** with the same model + a forced `record_hypothesis` function call (no tools). Lets you measure how much of the system's output quality comes from the multi-agent harness vs the underlying model. → live numbers in [`docs/BENCH_RESULTS.md`](docs/BENCH_RESULTS.md#headline-findings).

### Gold-set scoring (AML drug repurposing)

`paper-aml*` presets score **recall** against a curated answer key from the Co-Scientist paper. Two gold sets ship; both stay registered so historical bench artifacts remain interpretable.

| label                                                   | size | what it is |
| ---                                                     | --- | --- |
| `aml-repurposing-paper-top3` *(default for `paper-aml*`)* | 3 | Top-3 of the original paper's list: candidates with no prior published AML repurposing, no prior preclinical evidence in AML, and no external inputs (no DepMap scores, no expert curation). → **Nanvuranlat (JPH-203 / KYT-0353), KIRA6, Leflunomide (Arava / HWA-486 / Teriflunomide / Aubagio)** |
| `aml-repurposing-paper-5`                               | 5 | Broader 5-drug list referenced in the paper's main text: **Binimetinib (MEK162), Pacritinib (SB1518 / Vonjo), Cerivastatin (Baycol), Pravastatin (Pravachol), Dimethyl fumarate (DMF / BG-12 / Tecfidera)** |

Swap with `--goldset`:

```bash
co-scientist bench --preset paper-aml --goldset aml-repurposing-paper-5   # broader list
co-scientist bench --preset paper-aml --goldset none                       # head-to-head only
```

The matcher is whole-token, case-insensitive, and looks at every searched field of every hypothesis (title / summary / full_text / `entities` / citation excerpts). Drug **class** mentions (e.g. "DHODH inhibitor") do **not** count — the candidate has to name the actual compound (or one of its registered aliases).

### Custom candidates

`label=provider:model[@mode]`. `mode` is `pipeline` (default) or `direct`. Pipeline goes through the full Generation agent stack; direct is a single forced-tool LM call with no literature tools.

```bash
co-scientist bench "Identify hypotheses about X" \
  -c flash3=openrouter:google/gemini-3-flash-preview \
  -c flash3-raw=openrouter:google/gemini-3-flash-preview@direct \
  -c gpt5=openai:gpt-5 \
  -c opus=anthropic:claude-opus-4.7 \
  --judge anthropic:claude-sonnet-4-6
```

### Where results live

Every bench writes to SQLite + JSON on disk:

```
data/co_scientist.db                          ← SQLite, all metadata
  bench_runs                                  one row per bench
  bench_candidates                            one row per (bench × candidate × mode)
  bench_matches                               one row per head-to-head

data/artifacts/<session_id>/                  ← JSON on disk
  bench/<bench_id>.json                       run summary + per-entity gold_hit_detail
  hypotheses/<hyp_id>.json                    every hypothesis the bench produced
  transcripts/generation/<trn_id>.json        every LLM call
```

The auto-generated [`docs/BENCH_RESULTS.md`](docs/BENCH_RESULTS.md) (rebuild with `python scripts/build_bench_report.py`) walks every recorded bench and renders the per-candidate result table, every hypothesis attributed to the model that produced it, and a post-hoc rescore against every registered gold set.

### Mechanics

- **Generation runs in parallel** per candidate under a deep-copied Config (`cfg.llm.provider`, `cfg.models.*`, thinking budgets zeroed for non-Anthropic).
- **Round-robin pairings**: every pair plays `--matches` head-to-heads (one random hypothesis from each side per match).
- **Structured verdict** via a forced `record_verdict` function call — no fragile `better idea: <N>` text parsing across providers.
- Bench runs are **isolated from regular sessions** — they don't write to `tournament_matches` or affect any session's leaderboard.

## Repository layout

```
co_scientist/
  agents/       # supervisor + 6 specialized agents (base, generation, reflection,
                # ranking, evolution, proximity, metareview)
  bench/        # cross-model bench runner (Elo tournament + gold-set scoring)
  llm/          # provider abstraction (anthropic/openai/openrouter/gemini/...),
                # tool loop, token budgets, model routing, retry, batch, estimator
  storage/      # SQLite schema + migrations, db connection, 10 repos
  tools/        # tool registry; web_fetch, web_search, pubmed/arxiv/europe_pmc,
                # science-skills bridge
  vectors/      # embeddings (Voyage/OpenAI/hash-fallback) + FAISS IndexFlatIP
  orchestrator/ # task scheduling, Elo updates, termination, event bus
  safety/       # injection quoting, classifier, citation verifier
  obs/          # metrics (tokens, cost, cache hit ratio, latency)
  web/          # FastAPI + htmx + SSE UI + sanitized markdown renderer
  evals/        # per-agent + e2e + regression evals
  tests/        # 213 unit tests + fixtures + smoke
config/
  default.toml
  prompts/      # 14 Jinja2 templates (one per agent.mode), derived from
                # the paper's supplementary prompts
docs/
  BENCH_RESULTS.md   # every bench ever run (auto-generated)
scripts/
  build_bench_report.py
reference/      # paper source materials (pseudocode, prompts, diagrams)
data/           # gitignored; runtime artifacts (SQLite, FAISS, transcripts)
vendor/         # gitignored; pinned clone of google-deepmind/science-skills
```

## License

Apache-2.0.
