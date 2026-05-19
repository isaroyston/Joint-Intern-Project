"""
FASTAPI_TEMPLATE.py
-------------------
Minimal FastAPI wrapper that exposes an existing Python chatbot to the
Sentry Red Team dashboard.

Assumes you already have:
    - a chatbot class living in its own module (e.g. `my_chatbot.py`)
    - (optional) a database / vector-store client in its own module
      (e.g. `my_db.py`) that the chatbot needs at construction time

The wrapper exposes three endpoints that the dashboard expects:

    POST /chat    request  : { "message": "..." }
                  response : { "response": "..." }
    POST /reset   clears server-side conversation history
    GET  /health  liveness check

To adapt for your own bot, edit only the three TODO blocks below.
Run with:

    pip install fastapi uvicorn pydantic
    uvicorn FASTAPI_TEMPLATE:app --host 0.0.0.0 --port 8000

See FASTAPI_SETUP.md for the full integration guide.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# TODO 1 — IMPORT YOUR CHATBOT (and optionally your DB client)
# ---------------------------------------------------------------------------
# Replace these imports with the real module paths in your project.
# If your bot does not need a separate DB, delete the db import + usage.
#
# from my_chatbot import MyChatbot
# from my_db import MyDB
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Chatbot Red-Team Endpoint")


class ChatRequest(BaseModel):
    message: str
    debug: bool = False


class ChatResponse(BaseModel):
    response: str


class ResetResponse(BaseModel):
    status: str


@app.on_event("startup")
def _startup() -> None:
    """Build the chatbot once at startup so init cost isn't paid per request."""

    # ---------------------------------------------------------------------
    # TODO 2 — INSTANTIATE YOUR CHATBOT
    # ---------------------------------------------------------------------
    # Wire your real chatbot + dependencies here. Two common patterns:
    #
    #   Stateless / no DB:
    #       app.state.bot = MyChatbot()
    #
    #   Stateful with a DB / vector store:
    #       db = MyDB()
    #       app.state.db = db
    #       app.state.bot = MyChatbot(db=db)
    #
    # Anything you stash on `app.state` is accessible from the routes below.
    # ---------------------------------------------------------------------
    raise NotImplementedError(
        "FASTAPI_TEMPLATE: edit the @app.on_event('startup') hook to "
        "instantiate your chatbot, then delete this NotImplementedError."
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    bot = getattr(app.state, "bot", None)
    if bot is None:
        raise HTTPException(status_code=503, detail="Bot not initialized")

    try:
        # -----------------------------------------------------------------
        # TODO 3 — CALL YOUR CHATBOT
        # -----------------------------------------------------------------
        # Default: bot.chat(message) -> str
        #
        # If your chat() method needs extra args (user_id, conversation_id,
        # debug flag, etc.), pass them here. Examples:
        #
        #   answer = bot.chat(req.message)
        #   answer = bot.chat(req.message, debug=req.debug)
        #   answer = bot.chat(
        #       req.message,
        #       user_id="redteam-user",
        #       conversation_id=getattr(app.state, "conversation_id", None),
        #       debug=req.debug,
        #   )
        # -----------------------------------------------------------------
        answer = bot.chat(req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(response=str(answer))


@app.post("/reset", response_model=ResetResponse)
def reset() -> ResetResponse:
    """Clear server-side conversation state so each red-team scenario starts fresh."""
    bot = getattr(app.state, "bot", None)
    if bot is None:
        raise HTTPException(status_code=503, detail="Bot not initialized")

    clear = getattr(bot, "clear_history", None)
    if callable(clear):
        try:
            clear()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return ResetResponse(status="ok")
