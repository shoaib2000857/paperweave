from __future__ import annotations

import logging

from app.core.config import Settings
from app.services.providers import LLMProvider
from app.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, settings: Settings, provider: LLMProvider):
        self.settings = settings
        self.provider = provider

    async def complete(self, prompt: str) -> tuple[str, int, int]:
        prompt_tokens = count_tokens(prompt)
        logger.info("Generating answer with provider=%s model=%s", self.settings.providers.llm.provider, self.settings.providers.llm.model)
        answer = await self.provider.generate(prompt)
        completion_tokens = count_tokens(answer)
        return answer, prompt_tokens, completion_tokens
