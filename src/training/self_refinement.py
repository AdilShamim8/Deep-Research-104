"""
Self-Refinement: model improves its own outputs iteratively
without external feedback.

Based on: "Self-Refine: Iterative Refinement with Self-Feedback"
(Madaan et al., 2023)

The model:
1. Generates initial response
2. Critiques its own response
3. Refines based on critique
4. Repeats until satisfied or max steps
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from src.models.base_model import BaseModel, GenerationConfig


@dataclass
class SelfRefinementConfig:
    """Configuration for self-refinement."""
    max_refinements: int = 3
    min_improvement_threshold: float = 0.1
    generation_temperature: float = 0.7
    critique_temperature: float = 0.3
    refinement_temperature: float = 0.6
    max_tokens: int = 4096
    critique_max_tokens: int = 1024
    stop_on_satisfied: bool = True


@dataclass
class RefinementTrace:
    """Full trace of self-refinement process."""
    question: str
    drafts: list[str] = field(default_factory=list)
    critiques: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    final_answer: str = ""
    n_refinements: int = 0
    converged: bool = False
    total_tokens: int = 0

    @property
    def improvement(self) -> float:
        """Score improvement from first to last draft."""
        if len(self.scores) < 2:
            return 0.0
        return self.scores[-1] - self.scores[0]


SELF_CRITIQUE_PROMPT = """\
Review your response below and provide a detailed self-critique.

Question asked: {question}

Your response:
{response}

Evaluate your response on:
1. ACCURACY: Are all facts correct?
2. COMPLETENESS: Have you fully answered the question?
3. CLARITY: Is it clear and well-organized?
4. DEPTH: Is the analysis sufficiently deep?
5. GAPS: What important aspects did you miss?

Then give an overall quality score 0.0-1.0.

Format:
<critique>
[Your detailed critique]
</critique>
<improvements>
- [Specific improvement 1]
- [Specific improvement 2]
</improvements>
<score>[0.0-1.0]</score>
<satisfied>[YES/NO - whether the response is good enough as-is]</satisfied>
"""

SELF_REFINE_PROMPT = """\
Improve your response based on your self-critique.

Question: {question}

Previous response:
{previous_response}

Self-critique:
{critique}

Specific improvements needed:
{improvements}

Provide an improved response that addresses all critique points.
Maintain what was good, fix what was wrong, and add what was missing.
"""


class SelfRefinementEngine:
    """
    Self-refinement engine.

    The model acts as both generator and critic,
    iteratively improving its own outputs.

    Advantages over external feedback:
    - No human or separate model required
    - Can run entirely locally
    - Fast iteration

    Limitations:
    - Limited by model's self-awareness
    - Can reinforce errors it can't detect
    - Benefits diminish after 2-3 refinements
    """

    def __init__(
        self,
        model: BaseModel,
        config: Optional[SelfRefinementConfig] = None,
    ):
        self.model = model
        self.config = config or SelfRefinementConfig()

    async def refine(
        self,
        question: str,
        initial_response: Optional[str] = None,
        context: Optional[str] = None,
    ) -> RefinementTrace:
        """
        Run self-refinement loop.

        Args:
            question:           The question to answer.
            initial_response:   Optional pre-generated response.
                                If None, generates from scratch.
            context:            Optional search context.
        """
        trace = RefinementTrace(question=question)
        total_tokens = 0

        logger.info(
            f"Self-refinement | max_steps={self.config.max_refinements} | "
            f"model={self.model.model_name}"
        )

        # Step 1: Initial generation
        if initial_response:
            current_response = initial_response
        else:
            content = question
            if context:
                content = f"Context:\n{context}\n\nQuestion: {question}"
            gen_config = GenerationConfig(
                max_tokens=self.config.max_tokens,
                temperature=self.config.generation_temperature,
            )
            response = await self.model.generate(
                [{"role": "user", "content": content}],
                gen_config,
            )
            current_response = response.content
            total_tokens += response.total_tokens

        trace.drafts.append(current_response)

        # Refinement loop
        for step in range(self.config.max_refinements):
            # Step 2: Self-critique
            critique_result = await self._self_critique(
                question, current_response
            )
            total_tokens += critique_result["tokens"]

            trace.critiques.append(critique_result["critique"])
            trace.scores.append(critique_result["score"])

            logger.debug(
                f"Refinement step {step+1} | "
                f"score={critique_result['score']:.2f} | "
                f"satisfied={critique_result['satisfied']}"
            )

            # Stop if model is satisfied with current response
            if (
                self.config.stop_on_satisfied
                and critique_result["satisfied"]
                and critique_result["score"] >= 0.8
            ):
                logger.info(
                    f"Self-refinement converged at step {step+1} "
                    f"(score={critique_result['score']:.2f})"
                )
                trace.converged = True
                break

            # Stop if last refinement
            if step >= self.config.max_refinements - 1:
                break

            # Step 3: Refine
            refined_result = await self._refine_once(
                question=question,
                previous_response=current_response,
                critique=critique_result["critique"],
                improvements=critique_result["improvements"],
            )
            total_tokens += refined_result["tokens"]
            current_response = refined_result["response"]
            trace.drafts.append(current_response)

        trace.final_answer = current_response
        trace.n_refinements = len(trace.drafts) - 1
        trace.total_tokens = total_tokens

        logger.info(
            f"Self-refinement complete | "
            f"steps={trace.n_refinements} | "
            f"improvement={trace.improvement:.2f} | "
            f"final_score={trace.scores[-1] if trace.scores else 0:.2f}"
        )

        return trace

    async def batch_refine(
        self,
        questions: list[str],
        max_concurrent: int = 3,
    ) -> list[RefinementTrace]:
        """Refine multiple questions concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_refine(q: str) -> RefinementTrace:
            async with semaphore:
                return await self.refine(q)

        tasks = [bounded_refine(q) for q in questions]
        return await asyncio.gather(*tasks, return_exceptions=False)

    # ── Private methods ───────────────────────────────────────────────────────

    async def _self_critique(
        self,
        question: str,
        response: str,
    ) -> dict:
        """Generate self-critique of a response."""
        prompt = SELF_CRITIQUE_PROMPT.format(
            question=question,
            response=response[:3000],  # Limit to avoid token overflow
        )
        config = GenerationConfig(
            max_tokens=self.config.critique_max_tokens,
            temperature=self.config.critique_temperature,
        )
        result = await self.model.generate(
            [{"role": "user", "content": prompt}],
            config,
        )

        # Parse results
        critique = ""
        improvements = ""
        score = 0.7
        satisfied = False

        critique_match = re.search(
            r'<critique>(.*?)</critique>',
            result.content, re.DOTALL
        )
        if critique_match:
            critique = critique_match.group(1).strip()

        improvements_match = re.search(
            r'<improvements>(.*?)</improvements>',
            result.content, re.DOTALL
        )
        if improvements_match:
            improvements = improvements_match.group(1).strip()

        score_match = re.search(r'<score>([\d.]+)</score>', result.content)
        if score_match:
            score = float(score_match.group(1))
            score = max(0.0, min(1.0, score))

        satisfied_match = re.search(
            r'<satisfied>(YES|NO)</satisfied>',
            result.content, re.IGNORECASE
        )
        if satisfied_match:
            satisfied = satisfied_match.group(1).upper() == "YES"

        return {
            "critique": critique,
            "improvements": improvements,
            "score": score,
            "satisfied": satisfied,
            "tokens": result.total_tokens,
        }

    async def _refine_once(
        self,
        question: str,
        previous_response: str,
        critique: str,
        improvements: str,
    ) -> dict:
        """Generate one refinement pass."""
        prompt = SELF_REFINE_PROMPT.format(
            question=question,
            previous_response=previous_response[:2000],
            critique=critique,
            improvements=improvements,
        )
        config = GenerationConfig(
            max_tokens=self.config.max_tokens,
            temperature=self.config.refinement_temperature,
        )
        result = await self.model.generate(
            [{"role": "user", "content": prompt}],
            config,
        )
        return {
            "response": result.content,
            "tokens": result.total_tokens,
        }