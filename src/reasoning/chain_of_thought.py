"""
Chain-of-Thought (CoT) prompting implementation.

Supports:
- Zero-shot CoT ("Let's think step by step")
- Few-shot CoT (with examples)
- Self-consistency CoT (majority vote over multiple samples)
- Structured CoT (with explicit reasoning format)
"""

import asyncio
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from src.models.base_model import BaseModel, GenerationConfig, ModelResponse


# ── Prompt templates ──────────────────────────────────────────────────────────

ZERO_SHOT_COT_SUFFIX = (
    "\n\nLet's think through this carefully, step by step, "
    "before giving the final answer."
)

STRUCTURED_COT_SYSTEM = """\
You are an expert analytical reasoner. For every question:

1. UNDERSTAND: Restate what is being asked in your own words.
2. PLAN: Outline your approach before executing it.
3. REASON: Work through each step explicitly, showing your work.
4. VERIFY: Check your reasoning for errors or gaps.
5. CONCLUDE: State your final answer clearly.

Always separate your reasoning from your final answer using:
<reasoning>
... your step-by-step thinking ...
</reasoning>
<answer>
... your final answer ...
</answer>
"""

FEW_SHOT_COT_EXAMPLES = [
    {
        "question": (
            "If a train travels 120 miles in 2 hours, "
            "then stops for 30 minutes, then travels another "
            "90 miles in 1.5 hours, what is the average speed "
            "for the entire journey?"
        ),
        "reasoning": (
            "Step 1: Calculate total distance.\n"
            "  - First leg: 120 miles\n"
            "  - Second leg: 90 miles\n"
            "  - Total distance: 120 + 90 = 210 miles\n\n"
            "Step 2: Calculate total time.\n"
            "  - First leg: 2 hours\n"
            "  - Stop: 0.5 hours\n"
            "  - Second leg: 1.5 hours\n"
            "  - Total time: 2 + 0.5 + 1.5 = 4 hours\n\n"
            "Step 3: Calculate average speed.\n"
            "  - Average speed = Total distance / Total time\n"
            "  - Average speed = 210 / 4 = 52.5 mph"
        ),
        "answer": "The average speed for the entire journey is 52.5 mph.",
    }
]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CoTResult:
    """Result from a Chain-of-Thought generation."""
    answer: str
    reasoning: str
    raw_response: str
    confidence: float = 1.0          # 1.0 = single sample, <1.0 from voting
    supporting_votes: int = 1
    total_votes: int = 1
    model_response: Optional[ModelResponse] = None
    all_samples: list[str] = field(default_factory=list)


# ── Parser ────────────────────────────────────────────────────────────────────

class CoTParser:
    """Extract structured reasoning from model output."""

    @staticmethod
    def extract_reasoning_and_answer(text: str) -> tuple[str, str]:
        """
        Parse <reasoning> / <answer> tags.
        Falls back to heuristic split if tags not found.
        """
        reasoning_match = re.search(
            r'<reasoning>(.*?)</reasoning>',
            text, re.DOTALL | re.IGNORECASE
        )
        answer_match = re.search(
            r'<answer>(.*?)</answer>',
            text, re.DOTALL | re.IGNORECASE
        )

        if reasoning_match and answer_match:
            return (
                reasoning_match.group(1).strip(),
                answer_match.group(1).strip(),
            )

        # Fallback: split on common final-answer markers
        for marker in [
            "Therefore,", "Thus,", "In conclusion,",
            "Final answer:", "Answer:", "**Answer**",
        ]:
            if marker.lower() in text.lower():
                idx = text.lower().rfind(marker.lower())
                return text[:idx].strip(), text[idx:].strip()

        # Last resort: treat everything as reasoning, last paragraph as answer
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            return "\n\n".join(paragraphs[:-1]), paragraphs[-1]
        return "", text.strip()

    @staticmethod
    def extract_final_answer(text: str) -> str:
        """Extract just the answer portion."""
        _, answer = CoTParser.extract_reasoning_and_answer(text)
        return answer

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """
        Normalize answer for comparison in self-consistency voting.
        Lowercases, strips punctuation, collapses whitespace.
        """
        answer = answer.lower().strip()
        answer = re.sub(r'[^\w\s]', '', answer)
        answer = re.sub(r'\s+', ' ', answer)
        return answer


# ── Main CoT class ────────────────────────────────────────────────────────────

class ChainOfThought:
    """
    Chain-of-Thought prompting engine.

    Usage:
        cot = ChainOfThought(model)
        result = await cot.generate(
            question="What causes inflation?",
            mode="structured",
        )
        print(result.answer)
        print(result.reasoning)
    """

    def __init__(self, model: BaseModel):
        self.model = model
        self.parser = CoTParser()

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate(
        self,
        question: str,
        context: Optional[str] = None,
        mode: str = "structured",          # "zero_shot" | "few_shot" | "structured"
        config: Optional[GenerationConfig] = None,
    ) -> CoTResult:
        """
        Single CoT generation.

        Args:
            question:  The question or task to reason about.
            context:   Optional background context (e.g., from web search).
            mode:      CoT strategy to use.
            config:    Generation configuration.
        """
        messages = self._build_messages(question, context, mode)
        config = config or GenerationConfig(
            max_tokens=4096,
            temperature=0.7,
        )

        logger.debug(f"CoT generate | mode={mode} | model={self.model.model_name}")
        response = await self.model.generate(messages, config)

        reasoning, answer = self.parser.extract_reasoning_and_answer(
            response.content
        )

        # If model exposes its own reasoning trace, prefer that
        if response.reasoning_content:
            reasoning = response.reasoning_content

        return CoTResult(
            answer=answer or response.content,
            reasoning=reasoning,
            raw_response=response.content,
            model_response=response,
        )

    async def self_consistency(
        self,
        question: str,
        context: Optional[str] = None,
        n_samples: int = 5,
        mode: str = "structured",
        config: Optional[GenerationConfig] = None,
        aggregation: str = "majority_vote",   # "majority_vote" | "weighted"
    ) -> CoTResult:
        """
        Self-consistency: sample N independent reasoning paths,
        aggregate answers by majority vote.

        Wang et al. (2022): https://arxiv.org/abs/2203.11171
        Empirically improves accuracy 10-20% on reasoning benchmarks.
        """
        config = config or GenerationConfig(
            max_tokens=4096,
            temperature=0.8,      # Higher temperature for diversity
        )
        messages = self._build_messages(question, context, mode)

        logger.info(
            f"Self-consistency: sampling {n_samples} paths | "
            f"model={self.model.model_name}"
        )

        # Sample N paths concurrently
        tasks = [
            self.model.generate(messages, config)
            for _ in range(n_samples)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter failed responses
        valid_responses: list[ModelResponse] = []
        for r in responses:
            if isinstance(r, Exception):
                logger.warning(f"Sample failed: {r}")
            else:
                valid_responses.append(r)

        if not valid_responses:
            raise RuntimeError("All self-consistency samples failed")

        # Extract answers
        parsed_answers = []
        for resp in valid_responses:
            _, answer = self.parser.extract_reasoning_and_answer(resp.content)
            parsed_answers.append(answer or resp.content)

        # Aggregate
        best_answer, confidence, votes = self._aggregate_answers(
            parsed_answers, aggregation
        )

        # Pick the response whose answer matches the best
        best_reasoning = ""
        for resp, ans in zip(valid_responses, parsed_answers):
            if self.parser.normalize_answer(ans) == self.parser.normalize_answer(
                best_answer
            ):
                reasoning, _ = self.parser.extract_reasoning_and_answer(
                    resp.content
                )
                best_reasoning = resp.reasoning_content or reasoning
                break

        logger.info(
            f"Self-consistency result: confidence={confidence:.2f} "
            f"({votes}/{len(valid_responses)} votes)"
        )

        return CoTResult(
            answer=best_answer,
            reasoning=best_reasoning,
            raw_response=valid_responses[0].content,
            confidence=confidence,
            supporting_votes=votes,
            total_votes=len(valid_responses),
            all_samples=parsed_answers,
        )

    # ── Message builders ──────────────────────────────────────────────────────

    def _build_messages(
        self,
        question: str,
        context: Optional[str],
        mode: str,
    ) -> list[dict]:
        """Build messages list for chosen CoT mode."""
        if mode == "zero_shot":
            return self._zero_shot_messages(question, context)
        elif mode == "few_shot":
            return self._few_shot_messages(question, context)
        elif mode == "structured":
            return self._structured_messages(question, context)
        else:
            raise ValueError(
                f"Unknown CoT mode: {mode}. "
                f"Choose: zero_shot, few_shot, structured"
            )

    def _zero_shot_messages(
        self,
        question: str,
        context: Optional[str],
    ) -> list[dict]:
        content = ""
        if context:
            content += f"Context:\n{context}\n\n"
        content += f"Question: {question}{ZERO_SHOT_COT_SUFFIX}"
        return [{"role": "user", "content": content}]

    def _few_shot_messages(
        self,
        question: str,
        context: Optional[str],
    ) -> list[dict]:
        messages = []
        # Add examples as alternating user/assistant turns
        for ex in FEW_SHOT_COT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": f"Question: {ex['question']}"
            })
            messages.append({
                "role": "assistant",
                "content": (
                    f"<reasoning>\n{ex['reasoning']}\n</reasoning>\n"
                    f"<answer>\n{ex['answer']}\n</answer>"
                )
            })

        # Add the actual question
        content = ""
        if context:
            content += f"Context:\n{context}\n\n"
        content += f"Question: {question}"
        messages.append({"role": "user", "content": content})
        return messages

    def _structured_messages(
        self,
        question: str,
        context: Optional[str],
    ) -> list[dict]:
        content = ""
        if context:
            content += f"Context information:\n{context}\n\n"
        content += f"Question: {question}"
        return [
            {"role": "system", "content": STRUCTURED_COT_SYSTEM},
            {"role": "user", "content": content},
        ]

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate_answers(
        self,
        answers: list[str],
        method: str,
    ) -> tuple[str, float, int]:
        """
        Aggregate multiple answers into one.

        Returns: (best_answer, confidence, vote_count)
        """
        normalized = [self.parser.normalize_answer(a) for a in answers]
        counts = Counter(normalized)
        most_common_normalized, vote_count = counts.most_common(1)[0]
        confidence = vote_count / len(answers)

        # Return original (non-normalized) version of best answer
        for original, norm in zip(answers, normalized):
            if norm == most_common_normalized:
                return original, confidence, vote_count

        return answers[0], confidence, vote_count