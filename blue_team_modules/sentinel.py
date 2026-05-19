"""Standalone Sentinel guard module for reuse across projects.

Usage requirements:
- Set `SENTINEL_API_KEY` in your environment.
- Create `SentinelGuard(...)` and call `await guard.validate(...)`.
- User Guide: https://www.aiguardian.gov.sg/docs/wiki/Sentinel-APIs-%E2%80%90-User-Guide
- Connect with Sentinel team for onboarding: https://form.gov.sg/665978af9040a6be24d3978b?665982626cae0284c464a22a=Sentinel
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


DEFAULT_SENTINEL_GUARDRAILS: Dict[str, Dict[str, Any]] = {
    # Binary safety classifier for harmful/disallowed content.
    "lionguard-2-binary": {},
    # Detects content that drifts outside the assistant's supported scope.
    "off-topic": {},
    # Detects attempts to reveal hidden instructions/system prompts.
    "system-prompt-leakage": {},
    # Detects prompt-injection / jailbreak attack patterns.
    "aws/prompt_attack": {},
}


@dataclass
class SentinelResult:
    blocked: bool
    status_code: Optional[int] = None
    response_json: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    triggering_guardrails: Optional[List[str]] = None


class SentinelGuard:
    """Validates text/messages against Sentinel guardrails.

    Args:
        api_key: Sentinel API key. Falls back to `SENTINEL_API_KEY` env var.
        url: Sentinel endpoint URL.
        guardrails: Guardrail config sent in request payload.
        timeout: HTTP timeout in seconds.
        threshold: Score above this value is treated as triggered.
        fail_closed: If True, API failures will block requests.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        guardrails: Optional[Dict[str, Dict[str, Any]]] = None,
        timeout: int = 5,
        threshold: float = 0.90,
        fail_closed: bool = False,
    ):
        self.api_key = api_key or os.getenv("SENTINEL_API_KEY")
        self.url = url or "https://sentinel.stg.aiguardian.gov.sg/api/v1/validate"
        self.guardrails = guardrails or DEFAULT_SENTINEL_GUARDRAILS
        self.timeout = timeout
        self.threshold = threshold
        self.fail_closed = fail_closed

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def validate(
        self,
        text: str,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> SentinelResult:
        """Run Sentinel validation and return normalized result data.

        `messages` can include chat history to improve contextual detection.
        """
        if not self.enabled:
            return SentinelResult(blocked=False, error="SENTINEL_API_KEY missing")

        payload: Dict[str, Any] = {
            "text": text,
            "guardrails": self.guardrails,
        }
        if messages:
            payload["messages"] = messages

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.url, headers=headers, json=payload)
                response.raise_for_status()
                response_json = response.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            return SentinelResult(
                blocked=self.fail_closed,
                error=f"Sentinel API request failed: {exc}",
            )

        triggered = []
        results_dict = response_json.get("results", {})
        for guardrail_name, data in results_dict.items():
            score = data.get("score", 0.0)
            if score > self.threshold:
                triggered.append(f"{guardrail_name} ({score})")

        return SentinelResult(
            blocked=len(triggered) > 0,
            status_code=response.status_code,
            response_json=response_json,
            triggering_guardrails=triggered,
        )
