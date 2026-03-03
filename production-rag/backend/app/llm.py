"""Thin async wrapper around the Groq Chat Completions API (OpenAI-compatible)."""

from groq import APIError, AsyncGroq


class LLMError(RuntimeError):
    pass


class GroqLLM:
    def __init__(self, api_key: str, default_model: str) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._default_model = default_model

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=model or self._default_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=max_tokens,  # reasoning models use this, not max_tokens
                temperature=temperature,
            )
        except APIError as exc:
            raise LLMError(f"Groq API error: {exc}") from exc
        return response.choices[0].message.content or ""