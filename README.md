# Sentry Project

A Red Team & Blue Team project tested on a mock agentic withdrawal chatbot. The objective is to explore the vulnerabilities of agentic architectures through an automated red-teaming app, and to implement multilayer ML + LLM defences against prompt injection, policy circumvention, and scope drift.

The repository ships two cooperating apps: an attacker (the **Red Teaming App**) and a defender (the **Withdrawal Chatbot** wrapped in **Sentry Blue Team** layers). Running them against each other produces a measurable penetration rate that you can iterate on as you tighten guardrails.

---

## Applications

### Red Teaming App

A Streamlit dashboard that runs single-turn and multi-turn adversarial scenarios against any chatbot endpoint, mutates prompts through a library of obfuscation tools (homoglyph, char-swap, payload-mask, etc.), and scores each turn with an LLM judge. Results are written to `attacks/reports/` for diffing across runs.

![Sentry Red Teamer](pictures/sentry_red_teamer.png)

### Withdrawal Chatbot with Sentry Blue Team Layers

A Flask + LangGraph chatbot grounded on official SGBank withdrawal-policy documents. The Blue Team wraps it in three defensive layers: an external **Sentinel** input guardrail, a tool-driven QA agent constrained to approved policy excerpts, and an LLM **output checker** that can rewrite, block, or send the answer back for one retry.

![Withdrawal Chatbot](pictures/withdrawal_chatbot.png)

---

## Project Directory

```
.
├── app.py                          # Flask entry point — withdrawal chatbot UI
├── main.py                         # CLI entry point for the chatbot
├── ingest.py                       # Loads SGBank policy PDFs into the vector store
├── requirements.txt
│
├── src/
│   ├── chatbot/
│   │   ├── withdrawal_chatbot.py   # LangGraph agent, output checker, retry edge
│   │   └── sentinel_guard.py       # External Sentinel input-guardrail client
│   ├── db/
│   │   └── supabase_client.py      # Supabase + pgvector client
│   └── documents/                  # SGBank policy PDFs (RAG source of truth)
│
├── frontend/
│   ├── templates/                  # Flask Jinja templates (index, login, register)
│   └── static/                     # JS + CSS for the chat UI
│
├── attacks/
│   ├── streamlit_app.py            # Red Team dashboard
│   ├── red_teaming.py              # Single-turn attack runner
│   ├── generative_red_team.py      # Multi-turn attacker LLM + scenario walker
│   ├── evaluator.py                # LLM judge for jailbreak success
│   ├── prompt_attacks.py           # Static attack catalogue
│   ├── modules/                    # Mutation tools (homoglyph, char_swap, …)
│   ├── datasets/                   # Bias / toxicity CSVs + multi-turn scenarios
│   └── reports/                    # Penetration test results
│
└── pictures/                       # Architecture diagrams used in this README
```

---

## Run Project Instructions

### Common setup (do this once)

```bash
# 1. Clone + virtualenv
git clone <repo-url> Joint-Intern-Project
cd Joint-Intern-Project
python -m venv venv
source venv/bin/activate                 # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env — required keys:
#   OPENAI_API_KEY=...
#   SENTINEL_API_KEY=...
#   SENTINEL_API_URL=https://sentinel.stg.aiguardian.gov.sg/api/v1/validate
#   SUPABASE_URL=...
#   SUPABASE_KEY=...
```

### Withdrawal Bot

```bash
# 1. One-off: ingest the SGBank policy PDFs into Supabase pgvector
python ingest.py

# 2. Launch the Flask UI
python app.py
# → http://localhost:5000
```

For a headless CLI session: `python main.py`.

### Red Team App

The Red Team dashboard runs single-turn and multi-turn adversarial scenarios against any chatbot that exposes an HTTP endpoint, mutates prompts through obfuscation tools, and scores each response with an LLM evaluator.

> **Prerequisite — FastAPI endpoint.** The dashboard speaks HTTP, so any chatbot under test must be wrapped in a thin FastAPI shim. To wrap your own chatbot, copy [`FASTAPI_TEMPLATE.py`](FASTAPI_TEMPLATE.py) and follow [`FASTAPI_SETUP.md`](FASTAPI_SETUP.md). The included withdrawal bot is already wrapped in `api.py` at the repo root.

```bash
# 1. Start the chatbot endpoint (the included withdrawal bot, or your own
#    bot wrapped via FASTAPI_TEMPLATE.py)
uvicorn api:app --host 0.0.0.0 --port 8000
# → http://localhost:8000   (POST /chat, POST /reset, GET /health)

# 2. In a second terminal, launch the Streamlit dashboard
streamlit run attacks/streamlit_app.py
```

The dashboard loads `attacks/.env` if present, otherwise it falls back to the repo-root `.env`. Reports are written to `attacks/reports/` and can be replayed or diffed between defence iterations.

#### Dashboard walkthrough

![Dashboard overview](pictures/red_team_dashboard_overview.png)

##### 1. Connect to your chatbot API

Enter the **Base URL** and **Chat path** of your running endpoint. If your chatbot uses non-standard JSON keys (e.g. `query`/`answer` instead of `message`/`response`), override **Request prompt field** and **Response field** to match — no code changes needed on your side.

![Connect API](pictures/connect_api_dashboard.png)

##### 2. Pick an attack mode

Three modes can be combined in one run:

- **General (static datasets)** — runs curated CSVs of red-team prompts in single-turn mode.
- **Custom Single-turn Prompts** — bring your own CSV (`input`,`target` columns), edit inline, or upload.
- **Red Team LLM (multi-turn scenarios)** — JSON scenarios driven by an attacker LLM that adapts based on the bot's previous responses.

###### Built-in datasets (`attacks/datasets/`)

| File | Description |
| --- | --- |
| `moonshot_jailbreak_prompts.csv` | Moonshot's jailbreak corpus — DAN, instruction overrides, role-play bypasses. |
| `age-bias.csv` | BBQ-style scenarios probing age-based stereotyping. |
| `gender-bias.csv` | BBQ-style scenarios probing gender bias. |
| `nationality-bias.csv` | BBQ-style scenarios probing nationality bias. |
| `physical-appearance-bias.csv` | BBQ-style scenarios probing appearance bias. |
| `race-ethnicity-bias.csv` | BBQ-style scenarios probing race / ethnicity bias. |
| `religion-bias.csv` | BBQ-style scenarios probing religion-based bias. |
| `toxicity-prompts.csv` | Toxic input continuations / completions. |
| `singapore-pofma-statements-2024.csv` | Singapore POFMA misinformation truth-claim probes. |
| `custom_single_turn_prompts.csv` | Curated single-turn set targeting the SGBank withdrawal bot (instruction override, off-topic, prompt injection, etc.). |
| `custom_multi_turn_scenarios.json` | Scripted multi-turn scenarios — reconnaissance, scope drift, system / tool leak. |

![Select dataset](pictures/select_dataset.png)
![Custom prompt template](pictures/custom_prompt_template.png)
![Multi-turn scenario template](pictures/multiturn_scenario_template.png)

###### Upload your own

Single-turn CSVs require `input` and `target` columns. Multi-turn JSON files follow the schema in `attacks/datasets/custom_multi_turn_scenarios.json` (`scenario_id`, `attack_type`, `objective`, `description`, `turns[]`).

![Upload custom prompt](pictures/upload_custom_prompt_template.png)

##### 3. (Optional) Apply prompt-mutation tools

Each tool transforms prompts before they hit the bot — useful for testing whether a defence relies on surface-level pattern matching rather than semantic understanding.

| Tool | Description |
| --- | --- |
| `char_swap` | Swaps two adjacent characters in ~20% of words. Tests robustness to typos. |
| `homoglyph` | Replaces letters with Unicode look-alikes (e.g. Latin `a` → Cyrillic `а`). Probes keyword/regex filters. |
| `insert_punctuation` | Injects random punctuation into words. Disrupts tokenization-based filters. |
| `payload_mask` | Masks nouns/verbs by replacing them with their NLTK WordNet definitions. Hides intent semantically while preserving meaning. |
| `text_bugger` | TextAttack character-level adversarial augmenter (mix of swaps, deletions, homoglyphs, embedding swaps). |
| `text_fooler` | TextAttack semantic word-level augmenter using embedding-distance synonyms with POS-tag constraints. |

Selected tools chain in order, so combining `homoglyph` + `payload_mask` produces a prompt that is both lexically and semantically obfuscated.

![Prompt mutation tools](pictures/prompt_mutation_tools.png)

##### 4. Configure the LLM evaluator

The single-turn judge looks for refusal-style language — tweak the **Refusal suggestion** to match your bot's safe-fallback wording so legitimate refusals aren't misclassified. The multi-turn judge uses a free-form **evaluator condition**; describe what counts as a successful jailbreak. The default flags off-topic drift, leaked system internals, and policy-circumventing detail.

![Evaluator settings](pictures/evaluator_settings.png)

##### 5. Configure the attacker (multi-turn only)

For multi-turn scenarios, the attacker is itself an LLM. Edit its system prompt to describe your domain so it generates plausible adversarial follow-ups instead of generic ones. The attacker also receives the scenario `objective`, the next scripted turn as a soft `prompt_reference`, and the running conversation history.

![Attacker prompt](pictures/attacker_prompt.png)

##### 6. Run

Click **Run Red Teaming**. The sidebar locks while the run is in flight; click **Clear Results** afterwards to reconfigure.

![Run](pictures/run_red_teaming.png)

#### Reading the results

All output is rendered on the right side of the dashboard.

##### Overall jailbreak rate

The headline number — successful jailbreaks ÷ total attempts — alongside per-mode breakdowns. You can click download here to obtain the collated dataset of all prompt injections and responses in this test run.

![Overall results](pictures/Overall_dashboard_results.png)

##### Breakdown by category

Per-dataset / per-attack-type pivot showing which categories your bot is weakest against.

![Breakdown by category](<pictures/breakdown of results by cat.png>)

##### Per-case inspection

Drill down into individual prompts and bot responses, including the LLM evaluator's rationale for each verdict — useful for diagnosing why a defence layer let something through.

![Inspect dataset](<pictures/inspect dataset.png>)

Reports persist to `attacks/reports/run_YYYYMMDD_hhmmss.csv` and `attacks/reports/generative_attack_evaluation_results.json` so you can drill down on the successful prompt injections as you tighten the Blue Team layers.

---

## Red Team

![Red Team Workflow](pictures/red_teamer_workflow.png)

The attacker LLM is briefed with a scenario `objective`, a soft `prompt_reference` for the next scripted turn, and the running conversation history. Each generated prompt may be passed through a chain of mutation tools before it hits the bot, and every bot response is scored by an LLM evaluator that flags jailbreak success when the response leaks internal state, drifts off-topic, or surrenders policy-circumventing detail.

## Blue Team

![Blue Team Workflow](pictures/blue_team_workflow.png)

Four layered defences sit in front of the QA agent:

1. **Language query filter** — a per-character allowlist guard (`block_non_english` in `withdrawal_chatbot.py`) that rejects any input containing CJK characters, Cyrillic, Arabic, or other non-Latin scripts before it reaches the LLM. This blocks homoglyph and foreign-script prompt-injection attacks (e.g. Chinese-character payloads, Cyrillic look-alikes used to spell ASCII tokens).
2. **Sentinel input guardrail** — external safety classifier that blocks prompt-injection and unsafe input before any tool is reached.
3. **Tool-driven QA agent** — answers are grounded on `get_account_balance`, `get_withdrawal_limit`, and a RAG `policy_checker` that retrieves only from approved SGBank policy documents.
4. **LLM output checker** — validates every draft for safety, scope, and compliance. It can return the draft as-is, rewrite it, or route it back to the QA agent for one retry with feedback.

## Withdrawal Chatbot

![Withdrawal Chatbot Workflow](pictures/withdrawal_bot_workflow.png)

The chatbot is implemented as a LangGraph state machine: `load_history → sentinel_input → qa_agent → output_check`, with a single conditional edge from `output_check` back to `qa_agent` when a retry is requested. Conversation history and per-user account snapshots live in Supabase; policy excerpts live in pgvector and are retrieved through a keyword-rewrite + cosine-similarity cache pipeline.

---

## Environment Variables

```env
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=384

SENTINEL_API_KEY=
SENTINEL_API_URL=https://sentinel.stg.aiguardian.gov.sg/api/v1/validate

SUPABASE_URL=
SUPABASE_KEY=

RED_TEAM_EVAL_MODEL=gpt-4o-mini
```
