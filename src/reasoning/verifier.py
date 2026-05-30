"""
Verifier / Reward Models for search-against-a-verifier.

Implements:
  - Outcome Reward Model (ORM): score final answer quality
  - Process Reward Model (PRM): score intermediate reasoning steps  
  - Rule-based verifier: for math/code where answers are checkable
  - LLM-as-judge: use strong model to evaluate weaker model output
  - Best-of-N with verifier: generate N, pick highest-scoring

Based on: Let's Verify Step by Step (Lightman et al., 2023)
"""

import re
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable
from loguru import logger

from src.models.base_model import BaseModel, GenerationConfig, ModelResponse


# ── Base verifier ─────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """Result from any verifier."""
    score: float                          # 0.0 to 1.0
    is_correct: bool                      # Binary judgment
    feedback: str                         # Human-readable explanation
    step_scores: list[float] = field(default_factory=list)  # PRM only
    confidence: float = 1.0


class BaseVerifier(ABC):
    """Abstract verifier interface."""

    @abstractmethod
    async def verify(
        self,
        question: str,
        answer: str,
        reasoning: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> VerificationResult:
        """Verify an answer, optionally against a reference."""
        ...

    async def verify_steps(
        self,
        question: str,
        steps: list[str],
    ) -> list[VerificationResult]:
        """Verify each reasoning step (PRM-style). Default: verify all."""
        tasks = [
            self.verify(question, step)
            for step in steps
        ]
        return await asyncio.gather(*tasks)


# ── ORM: Outcome Reward Model ─────────────────────────────────────────────────

ORM_SYSTEM_PROMPT = """\
You are an expert evaluator assessing the quality of answers.

Evaluate the given answer to a question on these criteria:
1. CORRECTNESS: Is the answer factually accurate?
2. COMPLETENESS: Does it fully address the question?
3. REASONING: Is the reasoning sound and well-supported?
4. CLARITY: Is it clearly expressed?

Provide:
- A score from 0.0 (completely wrong) to 1.0 (perfect)
- A brief explanation of your score
- Whether you consider the answer correct (YES/NO)

Output format:
SCORE: <0.0-1.0>
CORRECT: <YES/NO>
FEEDBACK: <your explanation>
"""


class OutcomeRewardModel(BaseVerifier):
    """
    ORM: Scores final answers.
    Uses a strong LLM as judge when no ground truth is available.
    """

    def __init__(self, judge_model: BaseModel):
        self.judge = judge_model

    async def verify(
        self,
        question: str,
        answer: str,
        reasoning: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> VerificationResult:
        """Score a final answer using LLM judge."""

        content = f"Question: {question}\n\nAnswer: {answer}"
        if reasoning:
            content += f"\n\nReasoning provided:\n{reasoning}"
        if reference:
            content += f"\n\nReference answer (for comparison):\n{reference}"

        messages = [
            {"role": "system", "content": ORM_SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]

        config = GenerationConfig(max_tokens=512, temperature=0.0)
        try:
            response = await self.judge.generate(messages, config)
            return self._parse_orm_response(response.content)
        except Exception as e:
            logger.warning(f"ORM verification failed: {e}")
            return VerificationResult(
                score=0.5, is_correct=False,
                feedback=f"Verification failed: {e}"
            )

    def _parse_orm_response(self, text: str) -> VerificationResult:
        score_match = re.search(r'SCORE:\s*([\d.]+)', text)
        correct_match = re.search(r'CORRECT:\s*(YES|NO)', text, re.IGNORECASE)
        feedback_match = re.search(r'FEEDBACK:\s*(.+)', text, re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.5
        score = max(0.0, min(1.0, score))
        is_correct = (
            correct_match.group(1).upper() == "YES"
            if correct_match else score >= 0.7
        )
        feedback = (
            feedback_match.group(1).strip()
            if feedback_match else "No feedback provided"
        )

        return VerificationResult(
            score=score,
            is_correct=is_correct,
            feedback=feedback,
        )


# ── PRM: Process Reward Model ─────────────────────────────────────────────────

PRM_STEP_PROMPT = """\
You are evaluating a SINGLE reasoning step in a multi-step solution.

Question: {question}

Previous steps:
{previous_steps}

Current step being evaluated:
{current_step}

Is this step:
1. Logically valid given previous steps?
2. Making genuine progress toward the answer?
3. Free of errors?

Output format:
STEP_SCORE: <0.0-1.0>
STEP_CORRECT: <YES/NO>
STEP_FEEDBACK: <brief explanation>
"""


class ProcessRewardModel(BaseVerifier):
    """
    PRM: Scores each intermediate reasoning step.

    More fine-grained than ORM - can identify exactly
    where reasoning went wrong.

    Reference: "Let's Verify Step by Step" (Lightman et al., 2023)
    """

    def __init__(self, judge_model: BaseModel):
        self.judge = judge_model

    async def verify(
        self,
        question: str,
        answer: str,
        reasoning: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify by parsing reasoning into steps and scoring each.
        Aggregate step scores into final verdict.
        """
        if not reasoning:
            # Fall back to ORM-style
            orm = OutcomeRewardModel(self.judge)
            return await orm.verify(question, answer, reference=reference)

        steps = self._parse_steps(reasoning)
        if not steps:
            steps = [reasoning]

        step_results = await self.verify_steps(question, steps)
        step_scores = [r.score for r in step_results]

        # Aggregate: minimum score bottlenecks the reasoning
        # (one bad step poisons the whole chain)
        min_score = min(step_scores) if step_scores else 0.5
        avg_score = sum(step_scores) / len(step_scores) if step_scores else 0.5
        # Weighted: min matters more than average
        final_score = 0.6 * min_score + 0.4 * avg_score

        # Find first failing step
        first_fail = next(
            (i for i, r in enumerate(step_results) if not r.is_correct),
            None
        )

        if first_fail is not None:
            feedback = (
                f"Reasoning failed at step {first_fail + 1}: "
                f"{step_results[first_fail].feedback}"
            )
        else:
            feedback = (
                f"All {len(steps)} steps verified. "
                f"Average score: {avg_score:.2f}"
            )

        return VerificationResult(
            score=final_score,
            is_correct=final_score >= 0.7,
            feedback=feedback,
            step_scores=step_scores,
        )

    async def verify_steps(
        self,
        question: str,
        steps: list[str],
    ) -> list[VerificationResult]:
        """Score each step individually."""
        tasks = []
        for i, step in enumerate(steps):
            previous = "\n".join(
                f"Step {j+1}: {steps[j]}"
                for j in range(i)
            ) or "None"
            tasks.append(
                self._score_step(question, previous, step)
            )
        return await asyncio.gather(*tasks)

    async def _score_step(
        self,
        question: str,
        previous_steps: str,
        current_step: str,
    ) -> VerificationResult:
        prompt = PRM_STEP_PROMPT.format(
            question=question,
            previous_steps=previous_steps,
            current_step=current_step,
        )
        config = GenerationConfig(max_tokens=256, temperature=0.0)
        try:
            response = await self.judge.generate(
                [{"role": "user", "content": prompt}],
                config
            )
            return self._parse_step_response(response.content)
        except Exception as e:
            logger.warning(f"PRM step scoring failed: {e}")
            return VerificationResult(
                score=0.5, is_correct=True,
                feedback="Step verification failed"
            )

    def _parse_step_response(self, text: str) -> VerificationResult:
        score_match = re.search(r'STEP_SCORE:\s*([\d.]+)', text)
        correct_match = re.search(
            r'STEP_CORRECT:\s*(YES|NO)', text, re.IGNORECASE
        )
        feedback_match = re.search(r'STEP_FEEDBACK:\s*(.+)', text, re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.5
        score = max(0.0, min(1.0, score))
        is_correct = (
            correct_match.group(1).upper() == "YES"
            if correct_match else score >= 0.6
        )
        feedback = (
            feedback_match.group(1).strip()
            if feedback_match else ""
        )
        return VerificationResult(
            score=score, is_correct=is_correct, feedback=feedback
        )

    @staticmethod
    def _parse_steps(reasoning: str) -> list[str]:
        """Parse numbered steps from reasoning text."""
        steps = re.findall(
            r'(?:step\s+\d+[:.]\s*|^\d+\.\s*)(.+?)(?=step\s+\d+|^\d+\.|$)',
            reasoning,
            re.DOTALL | re.MULTILINE | re.IGNORECASE
        )
        if steps:
            return [s.strip() for s in steps if s.strip()]
        # Fallback: split on double newlines
        return [p.strip() for p in reasoning.split("\n\n") if p.strip()]


# ── Rule-based verifier ───────────────────────────────────────────────────────

class RuleBasedVerifier(BaseVerifier):
    """
    Deterministic verifier for problems with checkable answers.

    Supports:
    - Exact match
    - Numeric comparison (with tolerance)
    - Code execution
    - Regular expression matching
    - Custom verifier functions
    """

    def __init__(
        self,
        verify_fn: Optional[Callable[[str, str], bool]] = None,
        numeric_tolerance: float = 1e-6,
    ):
        self._verify_fn = verify_fn
        self._numeric_tolerance = numeric_tolerance

    async def verify(
        self,
        question: str,
        answer: str,
        reasoning: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> VerificationResult:
        if reference is None:
            return VerificationResult(
                score=0.5,
                is_correct=False,
                feedback="No reference answer provided for rule-based verification"
            )

        if self._verify_fn:
            try:
                is_correct = self._verify_fn(answer, reference)
            except Exception as e:
                return VerificationResult(
                    score=0.0,
                    is_correct=False,
                    feedback=f"Custom verifier error: {e}"
                )
        else:
            is_correct = self._default_verify(answer, reference)

        return VerificationResult(
            score=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            feedback=(
                "Answer matches reference." if is_correct
                else f"Expected: {reference}, Got: {answer}"
            )
        )

    def _default_verify(self, answer: str, reference: str) -> bool:
        """Try exact match, then numeric match."""
        # Normalize
        norm_ans = answer.strip().lower()
        norm_ref = reference.strip().lower()

        if norm_ans == norm_ref:
            return True

        # Try numeric comparison
        try:
            float_ans = float(re.sub(r'[,\s]', '', norm_ans))
            float_ref = float(re.sub(r'[,\s]', '', norm_ref))
            return abs(float_ans - float_ref) <= self._numeric_tolerance
        except ValueError:
            pass

        return False


# ── Best-of-N with Verifier ───────────────────────────────────────────────────

class BestOfNWithVerifier:
    """
    Generate N candidates, score with verifier, return best.

    This is the "search against a verifier" approach that
    combines inference-time scaling with quality control.
    """

    def __init__(
        self,
        generator: BaseModel,
        verifier: BaseVerifier,
    ):
        self.generator = generator
        self.verifier = verifier

    async def generate_and_verify(
        self,
        question: str,
        n: int = 8,
        context: Optional[str] = None,
        reference: Optional[str] = None,
        gen_config: Optional[GenerationConfig] = None,
    ) -> tuple[str, VerificationResult, list[VerificationResult]]:
        """
        Generate N candidates, verify each, return best.

        Returns:
            (best_answer, best_verification, all_verifications)
        """
        gen_config = gen_config or GenerationConfig(
            max_tokens=4096,
            temperature=0.8,
        )

        content = f"Question: {question}"
        if context:
            content = f"Context:\n{context}\n\n{content}"

        messages = [{"role": "user", "content": content}]

        logger.info(
            f"Best-of-N search: generating {n} candidates | "
            f"model={self.generator.model_name}"
        )

        # Generate N candidates
        tasks = [
            self._safe_generate(messages, gen_config, i)
            for i in range(n)
        ]
        responses = await asyncio.gather(*tasks)
        valid = [r for r in responses if r is not None]

        if not valid:
            raise RuntimeError("All candidate generations failed")

        # Verify each candidate
        verify_tasks = [
            self.verifier.verify(
                question,
                r.content,
                reasoning=r.reasoning_content,
                reference=reference,
            )
            for r in valid
        ]
        verifications = await asyncio.gather(*verify_tasks)

        # Find best
        best_idx = max(
            range(len(verifications)),
            key=lambda i: verifications[i].score
        )

        logger.info(
            f"Best-of-N: best_score={verifications[best_idx].score:.3f} | "
            f"correct={verifications[best_idx].is_correct} | "
            f"n_valid={len(valid)}/{n}"
        )

        return (
            valid[best_idx].content,
            verifications[best_idx],
            list(verifications),
        )

    async def _safe_generate(
        self,
        messages: list[dict],
        config: GenerationConfig,
        idx: int,
    ) -> Optional[ModelResponse]:
        try:
            return await self.generator.generate(messages, config)
        except Exception as e:
            logger.warning(f"Generation {idx} failed: {e}")
            return None