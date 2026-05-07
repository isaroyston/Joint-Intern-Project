# FastAPI Setup Guide for the Sentry Red Team App

This guide is for chatbot developers who want to test their own chatbot against the Sentry Red Team dashboard. The dashboard speaks HTTP, so any chatbot under test must expose three endpoints. The companion file [`FASTAPI_TEMPLATE.py`](FASTAPI_TEMPLATE.py) is a runnable starting point — copy it, plug your chatbot class in, and run.

---

## What the dashboard expects

| Method | Path      | Purpose                                       |
| ------ | --------- | --------------------------------------------- |
| POST   | `/chat`   | Send one user message, receive a bot reply    |
| POST   | `/reset`  | Clear server-side conversation history        |
| GET    | `/health` | Liveness check                                |

### Default request / response shape

```jsonc
// POST /chat
// Request body
{
  "message": "What is the withdrawal limit?",
  "debug": false                 // optional
}

// Response body
{
  "response": "Your daily withdrawal limit is $X."
}
```

```jsonc
// POST /reset
// Response body
{ "status": "ok" }
```

If your existing API already uses different field names (e.g. `query`/`answer` instead of `message`/`response`), you do **not** need to change your API — the dashboard sidebar lets you remap them per run:

| Sidebar field          | What it controls               |
| ---------------------- | ------------------------------ |
| Base URL               | Scheme + host + port           |
| Chat path              | Path of the chat endpoint      |
| Reset path             | Path of the reset endpoint     |
| Request prompt field   | JSON key for the user message  |
| Response field         | JSON key for the bot reply     |
| Debug field (optional) | JSON key for a debug flag      |

---

## Quick start (5 steps)

The template assumes you already have a chatbot class in its own script (e.g. `my_chatbot.py`), and optionally a database / vector-store client in another script (e.g. `my_db.py`). You do not need to refactor anything — just import them.

### 1. Copy the template

```bash
cp FASTAPI_TEMPLATE.py my_bot_api.py
```

The file has three clearly-marked `TODO` blocks. Edit only those.

### 2. `TODO 1` — import your chatbot (and optional DB)

At the top of `my_bot_api.py`, replace the commented-out example imports with the real module paths in your project:

```python
# Stateless bot, no DB:
from my_chatbot import MyChatbot

# Stateful bot with a DB / vector store:
from my_chatbot import MyChatbot
from my_db import MyDB
```

The minimum interface your chatbot class must expose is:

```python
class MyChatbot:
    def chat(self, message: str) -> str:
        # call your LLM / agent / RAG pipeline
        return "..."

    def clear_history(self) -> None:   # optional but recommended
        ...
```

If your real `chat()` method takes extra arguments (user id, conversation id, debug flag, etc.), you'll wire them in at `TODO 3` further down — your class itself does **not** need to change.

### 3. `TODO 2` — instantiate the chatbot at startup

Inside the `@app.on_event("startup")` hook, build your bot once and stash it on `app.state`. Two common patterns:

```python
# Stateless / no DB
app.state.bot = MyChatbot()

# Stateful with a DB / vector store
db = MyDB()
app.state.db = db
app.state.bot = MyChatbot(db=db)
```

Then **delete** the `raise NotImplementedError(...)` line at the bottom of the hook — it's a guardrail to fail fast if you forget to wire this up.

### 4. `TODO 3` — call your chatbot from `/chat`

Inside the `chat()` route, replace `bot.chat(req.message)` with whatever signature your method needs:

```python
# Default — single-arg chat method
answer = bot.chat(req.message)

# With debug flag
answer = bot.chat(req.message, debug=req.debug)

# Full identity-aware signature (matches the included withdrawal bot)
answer = bot.chat(
    req.message,
    user_id="redteam-user",
    conversation_id=getattr(app.state, "conversation_id", None),
    debug=req.debug,
)
```

### 5. Install, run, and verify

```bash
pip install fastapi uvicorn pydantic

uvicorn my_bot_api:app --host 0.0.0.0 --port 8000

# in another terminal:
curl -s http://127.0.0.1:8000/health
# {"status":"ok"}

curl -s -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "hello"}'
# {"response":"...your bot's reply..."}
```

---

## Pointing the dashboard at your endpoint

Launch the dashboard in a second terminal:

```bash
streamlit run attacks/streamlit_app.py
```

In the sidebar under **1) Connect Chatbot API**, set:

- **Base URL**: `http://127.0.0.1:8000`
- **Chat path**: `/chat`
- **Reset path**: `/reset`
- **Request prompt field**: `message`
- **Response field**: `response`

Leave the rest at their defaults unless you renamed JSON keys in your own API.

---

## Stateful vs stateless chatbots

The dashboard calls `/reset` between scenarios so each multi-turn run starts with a clean conversation.

- **Stateless bot** — `/reset` is a no-op. Leave `clear_history` off your class entirely; the template detects its absence and just returns `{"status": "ok"}`.
- **Stateful bot, in-memory** — implement `clear_history()` on your chatbot class. The template wires it into `/reset` automatically.
- **Stateful bot, DB-backed** — your `clear_history()` should create a fresh conversation row in the DB and return its id. Stash the id on `app.state.conversation_id` so subsequent `/chat` calls in `TODO 3` can pass it through. See `api.py` at the repo root for a worked example.

If your conversation state is keyed by `user_id` / `session_id`, hard-code a single red-team identity near the top of `my_bot_api.py` so every red-team request hits the same conversation slot:

```python
DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000000"  # any valid id
```

---

## Reference: the included withdrawal bot

`api.py` at the repo root is a real-world example of this pattern. It exposes `/chat`, `/reset`, `/health`, plus a `/search` endpoint specific to the withdrawal use case. Read it alongside `FASTAPI_TEMPLATE.py` to see what a production-ish wrapper around a stateful, multi-turn agent looks like.
