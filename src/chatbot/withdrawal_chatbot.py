"""
SGBank Withdrawal Assistant
RAG-powered chatbot restricted to official withdrawal policy documentation
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_store.vector_store import VectorStore


class WithdrawalChatbot: 
    """SGBank Withdrawal Policy Assistant (RAG-enabled)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 400,
        vector_store: Optional[VectorStore] = None,
    ):
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=self.api_key)

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.vector_store = vector_store

        self.conversation_history: List[Dict[str, str]] = []

        self.system_prompt = """
You are SGBank's official Withdrawal Policy Assistant.

You must:
- Answer strictly using SGBank withdrawal policy documentation.
- Use only provided context when available.
- If the information is not present in the policy excerpts, state:
  "This information is not available in the SGBank Withdrawal Policy."
- Never fabricate withdrawal limits, fees, or rules.
- Never disclose internal fraud detection logic.
- Provide clear, professional, and neutral banking responses.
"""

    def clear_history(self):
        self.conversation_history = []

    # ---------------------------
    # Deterministic Rejection Layer
    # ---------------------------
    def _should_reject(self, user_message: str) -> bool:
        risky_keywords = [
            "bypass",
            "avoid aml",
            "scam",
            "trick elderly",
            "fraud",
            "circumvent",
            "exploit",
            "hack",
            "override limit",
            "without detection",
            "illegal"
        ]

        message_lower = user_message.lower()
        return any(keyword in message_lower for keyword in risky_keywords)

    # ---------------------------
    # Context Retrieval
    # ---------------------------
    def _retrieve_context(self, query: str, n_results: int = 3) -> str:
        if not self.vector_store:
            return ""

        results = self.vector_store.search(query, n_results=n_results)

        if not results or "documents" not in results:
            return ""

        documents = results["documents"]
        if not documents or not documents[0]:
            return ""

        context_chunks = documents[0]
        return "\n\n".join(context_chunks)

    # ---------------------------
    # Main Chat Method
    # ---------------------------
    def chat(self, user_message: str, use_rag: bool = True) -> str:

        # Hard rejection before RAG or LLM
        if self._should_reject(user_message):
            return "Sorry I am unable to assist with that. Please feel free to ask other questions regarding withdrawal"

        context = ""
        if use_rag and self.vector_store:
            context = self._retrieve_context(user_message)

        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.append({
                "role": "system",
                "content": f"Official SGBank Withdrawal Policy Excerpts:\n\n{context}"
            })

        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"System error: {str(e)}"
