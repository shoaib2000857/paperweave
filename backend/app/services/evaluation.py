from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.models.api import EvaluationResult
from app.services.llm import LLMClient


class EvaluationService:
    def __init__(self, settings: Settings, llm_client: LLMClient):
        self.settings = settings
        self.llm_client = llm_client

    async def evaluate(self, question: str, answer: str, reference_answer: str | None) -> EvaluationResult | None:
        if not reference_answer:
            return None

        from bert_score import score as bert_score

        precision, recall, f1 = bert_score(
            [answer],
            [reference_answer],
            lang="en",
            model_type=self.settings.evaluation.bertscore_model,
            verbose=False,
        )
        judge_prompt = (
            "You are grading an answer for factual alignment.\n"
            f"Question: {question}\n"
            f"Reference answer: {reference_answer}\n"
            f"Candidate answer: {answer}\n"
            "Respond with PASS or FAIL on the first line, then a short reason."
        )
        judge_text, _, _ = await self.llm_client.complete(judge_prompt)
        first_line = judge_text.strip().splitlines()[0].upper() if judge_text.strip() else "FAIL"
        return EvaluationResult(
            bertscore_f1=float(f1[0].item()),
            judge_pass=first_line.startswith("PASS"),
            judge_reasoning=judge_text.strip(),
        )

    def summarize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        return {"count": len(results)}
