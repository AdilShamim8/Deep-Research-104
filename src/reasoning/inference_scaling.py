"""
Inference-Time Scaling — dedicated module.

The core idea: spend more compute at inference time
(not training time) to improve answer quality.

Techniques unified here:
  1. Scaling by samples   — generate more, pick best
  2. Scaling by steps     — longer reasoning chains
  3. Scaling by search    — wider/deeper tree search
  4. Scaling by revision  — iterative self-improvement
  5. Budget forcing       — control compute explicitly

Reference:
  - "Scaling LLM Test-Time Compute" (Snell et al., 2024)
  - OpenAI o1 system card
  - DeepSeek-R1 paper
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from loguru import logger

from src.models.base_model import BaseModel, GenerationConfig, ModelResponse
from src.reasoning.parallel_sampling import ParallelSampler
from src.reasoning.sequential_sampling import SequentialSampler
from src.reasoning.chain_of_thought import ChainOfThought
from src.reasoning.tree_of_thoughts import TreeOfThoughts, SearchStrategy
from src.reasoning.verifier import BaseVerifier, OutcomeRewardModel


class ScalingStrategy(Enum):
    """
    How to spend the compute budget.

    SAMPLES:   More parallel samples, pick best (width)
    STEPS:     Longer sequential reasoning (depth)
    SEARCH:    Wider tree search (branching)
    REVISION:  More self-refinement iterations
    ADAPTIVE:  Auto-select best strategy per budget
    """
    SAMPLES  = "samples"
    STEPS    = "steps"
    SEARCH   = "search"
    REVISION = "revision"
    ADAPTIVE = "adaptive"


@dataclass
class ScalingBudget:
    """
    Explicit compute budget for inference-time scaling.

    token_budget:    Maximum tokens to spend total.
    time_budget_s:   Maximum wall-clock seconds.
    max_samples:     Hard cap on parallel samples.
    max_steps:       Hard cap on sequential steps.
    max_depth:       Hard cap on tree depth.
    """
    token_budget: int = 16_000
    time_budget_s: float = 60.0
    max_samples: int = 8
    max_steps: int = 4
    max_depth: int = 4

    def scale_from_difficulty(self, difficulty: str) -> "ScalingBudget":
        """
        Auto-scale budget based on estimated problem difficulty.
        Difficulty: "easy" | "medium" | "hard" | "very_hard"
        """
        multipliers = {
            "easy":      0.25,
            "medium":    0.5,
            "hard":      1.0,
            "very_hard": 2.0,
        }
        m = multipliers.get(difficulty, 1.0)
        return ScalingBudget(
            token_budget=int(self.token_budget * m),
            time_budget_s=self.time_budget_s * m,
            max_samples=max(1, int(self.max_samples * m)),
            max_steps=max(1, int(self.max_steps * m)),
            max_depth=max(1, int(self.max_depth * m)),
        )


@dataclass
class ScalingResult:
    """
    Result from inference-time scaling.
    Records what strategy was used and how much was spent.
    """
    answer: str
    reasoning: Optional[str]
    strategy_used: str
    tokens_spent: int
    time_spent_s: float
    quality_score: float
    n_candidates_generated: int
    metadata: dict = field(default_factory=dict)

    @property
    def tokens_per_quality_point(self) -> float:
        """Efficiency metric: tokens spent per unit of quality."""
        if self.quality_score <= 0:
            return float('inf')
        return self.tokens_spent / self.quality_score


class InferenceTimeScaler:
    """
    Unified inference-time scaling controller.

    Given a question and a compute budget, decides HOW to spend
    that budget for maximum answer quality.

    Usage:
        scaler = InferenceTimeScaler(model, verifier)
        result = await scaler.scale(
            question="Prove that sqrt(2) is irrational",
            budget=ScalingBudget(token_budget=8000, max_samples=4),
            strategy=ScalingStrategy.ADAPTIVE,
        )
    """

    def __init__(
        self,
        model: BaseModel,
        verifier: Optional[BaseVerifier] = None,
    ):
        self.model = model
        self.verifier = verifier or OutcomeRewardModel(model)

        # Sub-engines
        self._parallel  = ParallelSampler(model)
        self._sequential = SequentialSampler(model)
        self._cot        = ChainOfThought(model)

    # ── Public API ────────────────────────────────────────────────────────────

    async def scale(
        self,
        question: str,
        context: Optional[str] = None,
        budget: Optional[ScalingBudget] = None,
        strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE,
        reference_answer: Optional[str] = None,
    ) -> ScalingResult:
        """
        Scale inference compute for a question.

        Args:
            question:           Question or problem to solve.
            context:            Optional background context.
            budget:             Compute budget (tokens, time, samples).
            strategy:           How to spend the budget.
            reference_answer:   Ground truth for verification (if known).

        Returns:
            ScalingResult with best answer and spend statistics.
        """
        budget = budget or ScalingBudget()
        start  = time.time()

        logger.info(
            f"Inference scaling | strategy={strategy.value} | "
            f"token_budget={budget.token_budget} | "
            f"model={self.model.model_name}"
        )

        if strategy == ScalingStrategy.ADAPTIVE:
            strategy = await self._select_strategy(question, budget)
            logger.info(f"Adaptive strategy selected: {strategy.value}")

        if strategy == ScalingStrategy.SAMPLES:
            result = await self._scale_by_samples(
                question, context, budget, reference_answer
            )
        elif strategy == ScalingStrategy.STEPS:
            result = await self._scale_by_steps(
                question, context, budget
            )
        elif strategy == ScalingStrategy.SEARCH:
            result = await self._scale_by_search(
                question, context, budget
            )
        elif strategy == ScalingStrategy.REVISION:
            result = await self._scale_by_revision(
                question, context, budget
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        result.time_spent_s = time.time() - start
        logger.info(
            f"Scaling complete | "
            f"strategy={result.strategy_used} | "
            f"tokens={result.tokens_spent} | "
            f"quality={result.quality_score:.3f} | "
            f"time={result.time_spent_s:.1f}s"
        )
        return result

    async def compute_scaling_curve(
        self,
        question: str,
        context: Optional[str] = None,
        token_budgets: Optional[list[int]] = None,
        strategy: ScalingStrategy = ScalingStrategy.SAMPLES,
        reference_answer: Optional[str] = None,
    ) -> list[ScalingResult]:
        """
        Compute scaling curve: quality vs. compute budget.

        Runs the same question at multiple budget levels
        to show how quality scales with compute.

        Returns results sorted by token_budget ascending.
        """
        token_budgets = token_budgets or [
            1_000, 2_000, 4_000, 8_000, 16_000, 32_000
        ]

        logger.info(
            f"Computing scaling curve | "
            f"n_points={len(token_budgets)} | "
            f"strategy={strategy.value}"
        )

        results = []
        for tb in token_budgets:
            budget = ScalingBudget(
                token_budget=tb,
                max_samples=max(1, tb // 2000),
                max_steps=max(1, tb // 4000),
                max_depth=max(1, tb // 4000),
            )
            result = await self.scale(
                question=question,
                context=context,
                budget=budget,
                strategy=strategy,
                reference_answer=reference_answer,
            )
            results.append(result)
            logger.info(
                f"Budget {tb:>6} tokens | "
                f"quality={result.quality_score:.3f} | "
                f"efficiency={result.tokens_per_quality_point:.0f}"
            )

        return results

    # ── Strategy implementations ──────────────────────────────────────────────

    async def _scale_by_samples(
        self,
        question: str,
        context: Optional[str],
        budget: ScalingBudget,
        reference_answer: Optional[str],
    ) -> ScalingResult:
        """
        Scaling strategy 1: More parallel samples.

        Generate N samples, score each with verifier,
        return the best. Quality improves logarithmically with N.

        pass@k formula: P(at least one correct) = 1 - (1-p)^k
        """
        messages = self._build_messages(question, context)
        tokens_per_sample = budget.token_budget // max(budget.max_samples, 1)
        n_samples = min(
            budget.max_samples,
            max(1, budget.token_budget // 1500),
        )

        config = GenerationConfig(
            max_tokens=tokens_per_sample,
            temperature=0.8,
        )

        logger.debug(f"Scaling by samples: n={n_samples}")

        result = await self._parallel.sample_best_of_n(
            messages=messages,
            n=n_samples,
            config=config,
            selection="tournament" if n_samples >= 4 else "score",
        )

        # Score the best answer
        verification = await self.verifier.verify(
            question=question,
            answer=result.best_content,
            reasoning=result.best_reasoning,
            reference=reference_answer,
        )

        # Estimate tokens spent
        tokens_spent = n_samples * tokens_per_sample

        return ScalingResult(
            answer=result.best_content,
            reasoning=result.best_reasoning,
            strategy_used="samples",
            tokens_spent=min(tokens_spent, budget.token_budget),
            time_spent_s=0.0,
            quality_score=verification.score,
            n_candidates_generated=n_samples,
            metadata={
                "n_samples": n_samples,
                "best_sample_score": result.best_score,
                "selection": "tournament" if n_samples >= 4 else "score",
            },
        )

    async def _scale_by_steps(
        self,
        question: str,
        context: Optional[str],
        budget: ScalingBudget,
    ) -> ScalingResult:
        """
        Scaling strategy 2: Longer reasoning chains.

        Use more sequential refinement steps.
        Quality improves as model catches its own errors.
        """
        tokens_per_step = budget.token_budget // max(budget.max_steps, 1)
        n_steps = min(
            budget.max_steps,
            max(1, budget.token_budget // 2000),
        )

        config = GenerationConfig(
            max_tokens=tokens_per_step,
            temperature=0.6,
        )

        logger.debug(f"Scaling by steps: n_steps={n_steps}")

        result = await self._sequential.refine(
            question=question,
            context=context,
            max_steps=n_steps,
            config=config,
        )

        verification = await self.verifier.verify(
            question=question,
            answer=result.final_answer,
        )

        return ScalingResult(
            answer=result.final_answer,
            reasoning="\n\n".join(
                s.critique or "" for s in result.steps
            ),
            strategy_used="steps",
            tokens_spent=result.total_tokens,
            time_spent_s=0.0,
            quality_score=verification.score,
            n_candidates_generated=len(result.steps),
            metadata={
                "n_steps": len(result.steps),
                "converged": result.converged,
            },
        )

    async def _scale_by_search(
        self,
        question: str,
        context: Optional[str],
        budget: ScalingBudget,
    ) -> ScalingResult:
        """
        Scaling strategy 3: Wider/deeper tree search.

        Tree of Thoughts with budget-controlled branching.
        Quality improves by exploring more reasoning paths.
        """
        # Scale tree parameters with budget
        branching = min(
            3,
            max(2, budget.token_budget // 4000)
        )
        depth = min(
            budget.max_depth,
            max(2, budget.token_budget // 3000)
        )
        beam_w = min(3, max(1, budget.token_budget // 6000))

        logger.debug(
            f"Scaling by search: "
            f"branching={branching} depth={depth} beam={beam_w}"
        )

        tot = TreeOfThoughts(
            model=self.model,
            branching_factor=branching,
            max_depth=depth,
            beam_width=beam_w,
            search_strategy=SearchStrategy.BEAM,
        )

        config = GenerationConfig(
            max_tokens=min(1024, budget.token_budget // 4),
            temperature=0.8,
        )

        result = await tot.solve(
            problem=question,
            context=context,
            config=config,
        )

        verification = await self.verifier.verify(
            question=question,
            answer=result.answer,
        )

        # Estimate tokens spent
        tokens_spent = (
            result.total_nodes_explored
            * config.max_tokens
            * 2  # Account for scoring calls
        )

        return ScalingResult(
            answer=result.answer,
            reasoning="\n\n".join(
                f"Step {i+1}: {n.thought}"
                for i, n in enumerate(result.best_path)
            ),
            strategy_used="search",
            tokens_spent=min(tokens_spent, budget.token_budget),
            time_spent_s=0.0,
            quality_score=verification.score,
            n_candidates_generated=result.total_nodes_explored,
            metadata={
                "nodes_explored": result.total_nodes_explored,
                "best_path_length": len(result.best_path),
                "best_score": result.best_score,
                "branching": branching,
                "depth": depth,
            },
        )

    async def _scale_by_revision(
        self,
        question: str,
        context: Optional[str],
        budget: ScalingBudget,
    ) -> ScalingResult:
        """
        Scaling strategy 4: More self-revision passes.

        Iterative self-refinement with increasing depth.
        """
        from src.training.self_refinement import (
            SelfRefinementEngine,
            SelfRefinementConfig,
        )

        n_revisions = min(
            budget.max_steps,
            max(1, budget.token_budget // 3000),
        )
        tokens_per_pass = budget.token_budget // (n_revisions + 1)

        config = SelfRefinementConfig(
            max_refinements=n_revisions,
            max_tokens=tokens_per_pass,
            generation_temperature=0.7,
            critique_temperature=0.2,
        )

        logger.debug(f"Scaling by revision: n_revisions={n_revisions}")

        engine = SelfRefinementEngine(self.model, config)
        trace = await engine.refine(question, context=context)

        verification = await self.verifier.verify(
            question=question,
            answer=trace.final_answer,
        )

        return ScalingResult(
            answer=trace.final_answer,
            reasoning="\n\n".join(
                f"Draft {i}: {d[:200]}..."
                for i, d in enumerate(trace.drafts)
            ),
            strategy_used="revision",
            tokens_spent=trace.total_tokens,
            time_spent_s=0.0,
            quality_score=verification.score,
            n_candidates_generated=len(trace.drafts),
            metadata={
                "n_revisions": trace.n_refinements,
                "converged": trace.converged,
                "score_improvement": trace.improvement,
                "final_score": (
                    trace.scores[-1] if trace.scores else 0
                ),
            },
        )

    # ── Adaptive strategy selection ───────────────────────────────────────────

    async def _select_strategy(
        self,
        question: str,
        budget: ScalingBudget,
    ) -> ScalingStrategy:
        """
        Automatically select best scaling strategy.

        Rules:
          - Very small budget (<2k tokens) → SAMPLES (fast, parallel)
          - Math/logic/step-by-step keywords → SEARCH (ToT)
          - "Explain"/"Why"/"How" questions → STEPS (long reasoning)
          - Factual questions → SAMPLES (self-consistency)
          - Large budget + complex → REVISION (deep polish)
        """
        q_lower = question.lower()

        # Budget-based selection
        if budget.token_budget < 2_000:
            return ScalingStrategy.SAMPLES

        # Keyword heuristics
        math_keywords = {
            "prove", "solve", "calculate", "compute",
            "derive", "equation", "formula", "theorem",
            "step", "algorithm",
        }
        explanation_keywords = {
            "explain", "why", "how does", "describe",
            "what is", "summarize", "elaborate",
        }
        creative_keywords = {
            "write", "create", "generate", "design",
            "compose", "draft",
        }

        words = set(q_lower.split())

        if words & math_keywords:
            return ScalingStrategy.SEARCH

        if words & explanation_keywords:
            if budget.token_budget >= 8_000:
                return ScalingStrategy.STEPS
            return ScalingStrategy.SAMPLES

        if words & creative_keywords:
            return ScalingStrategy.REVISION

        # Default: samples (most reliable baseline)
        return ScalingStrategy.SAMPLES

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_messages(
        self,
        question: str,
        context: Optional[str],
    ) -> list[dict]:
        content = ""
        if context:
            content += f"Context:\n{context}\n\n"
        content += f"Question: {question}"
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert analytical reasoner. "
                    "Think carefully and provide thorough, accurate answers."
                ),
            },
            {"role": "user", "content": content},
        ]