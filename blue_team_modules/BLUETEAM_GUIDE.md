# blue_team_modules

Reusable blue-team modules extracted for direct use in other Python projects.

## Contents

- `sentinel.py`: Sentinel API guardrails client
SETUP BLUE TEAM

This folder contains reusable blue-team safety modules that can be copied into other Python projects.

Contents
- `sentinel.py` — Sentinel API guardrails client
- `landetect_guard.py` — language / non-English input gate
- `output_guard.py` — output-checker prompt + parser utilities

## Installation

Install runtime dependencies in your project:

```bash
pip install httpx
```

Optional packages (only required for additional features):

```bash
pip install langdetect langchain-core
```

## Sentinel — Setup & Use

Purpose
- Server-side guardrail validation using Sentinel's policy engine. Detects off-topic, prompt-injection, system-prompt leakage, and other defined rules.

How to obtain a Sentinel API key
1. Contact your Sentinel platform administrator or security team and request a key for your application.
    - Connect with Sentinel team for onboarding: https://form.gov.sg/665978af9040a6be24d3978b?665982626cae0284c464a22a=Sentinel
2. Provide: project/application name, environment (`stg` or `prod`), intended use/guardrails.
3. Store the key in a secrets manager or environment variable.


Configure key in environment (example PowerShell):

```powershell
$env:SENTINEL_API_KEY = "your-api-key"
```

Quick connectivity test (PowerShell):

```powershell
$headers = @{ "x-api-key" = $env:SENTINEL_API_KEY; "Content-Type" = "application/json" }
$body = @{ text = "hello"; guardrails = @{ "lionguard-2-binary" = @{} } } | ConvertTo-Json -Depth 6
Invoke-RestMethod -Uri "https://sentinel.stg.aiguardian.gov.sg/api/v1/validate" -Method Post -Headers $headers -Body $body
```

Basic (Exaample) usage (async):

```python
import asyncio
from blue_team_modules.sentinel import SentinelGuard

async def main():
    guard = SentinelGuard(timeout=5, threshold=0.90)
    result = await guard.validate(text="User message")
    if result.error:
        print("Sentinel error:", result.error)
    print("blocked:", result.blocked)
    print("triggered:", result.triggering_guardrails)

asyncio.run(main())
```

## LangDetect Guard — Setup & Use

Purpose
- Fast character-based rejection of non-ASCII/mixed-script input, with an optional `langdetect`-based second stage (if installed) for higher confidence.

Install `langdetect` (optional, improves detection):

```bash
pip install langdetect
```

Usage examples:

1) One-line helper (convenience):

```python
from blue_team_modules.landetect_guard import block_non_english

if block_non_english(user_text):
    # ask user to rephrase in English or reject
    pass
```

2) Custom guard instance (tunable):

```python
from blue_team_modules.landetect_guard import LanguageDetectGuard

guard = LanguageDetectGuard(min_chars_for_lang_check=40, min_english_prob=0.9)
if guard.block_non_english(user_text, enforce_langdetect=True):
    # handle non-English input
    pass
```

## Output Guard — Setup & Use

Purpose
- Drop-in output safety checker. Sits after your LLM call, before you return to the user. Validates and can request regeneration if the output violates safety rules.

Usage options
- Configure rules, topics, and model provider/API key at startup. 
- Use `guard.check(user_message, draft_answer, generate_fn)` to evaluate the draft. Use `generate_fn` to allow the guard to instruct your own LLM to regenerate on failure, or omit it to have the guard attempt a rewrite.

Example:

```python
from blue_team_modules.output_guard import OutputGuard

guard = OutputGuard(
    domain_name="CPF Board internal usage",
    domain_scope="withdrawal policies, account information",
    off_topic_examples=["How's the weather today?"],
    safety_rules=["Toxic content: Do not engage with hostile messages."],
    trusted_tools=[],
    blocked_message="I'm sorry, I cannot assist with that.",
    api_key="YOUR_GUARD_KEY",
    provider="openai",
    model="gpt-4o",
)

# Example generating function (to pass optional hint)
def generate(user_msg, hint=None):
    # Your LLM call logic
    return "Some draft answer"
    
result = guard.check(
    user_message="What is my withdrawal limit?",
    draft_answer=generate("What is my withdrawal limit?"),
    generate_fn=generate,
)

print(result["answer"])     # Safe answer
print(result["blocked"])    # True if blocked/filtered
```

5) Plugging everything into your app

Example integration sketch (sync entrypoint calling async sentinel):

```python
import asyncio
from blue_team_modules.landetect_guard import block_non_english
from blue_team_modules.sentinel import SentinelGuard
from blue_team_modules.output_guard import OutputGuard

def handle_user_message(user_text):
    if block_non_english(user_text):
        return "Please rephrase in English."

    # Sentinel check (async -> sync wrapper)
    guard = SentinelGuard()
    result = asyncio.run(guard.validate(text=user_text))
    if result.error:
        # decide whether to fail open/closed
        print("Sentinel error:", result.error)
    if result.blocked:
        return "Request blocked by safety filters."

    # generate draft answer with your LLM, then run output guard
    # draft_answer = llm_generate(user_text)
    # guard = OutputGuard(domain_name="...", api_key="...", ...) # typically configured globally
    # verdict = guard.check(user_message=user_text, draft_answer=draft_answer, generate_fn=llm_generate)

    return "OK to proceed"
```

6) Notes & Best Practices

- Keep the Sentinel API key in a secrets manager or env var; never commit keys.
- Use `fail_closed=True` on `SentinelGuard` only if you want API failures to block requests.
- The provided helpers are intentionally minimal; adapt them to your app's async model and error handling policies.

Questions or want example scripts added? I can add a small `examples/` folder with runnable scripts for each module.
