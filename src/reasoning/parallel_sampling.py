"""
Parallel Sampling: generate N independent completions simultaneously,
then select the best using a scoring strategy.

Different from self-consistency in that we use a quality scorer
(not just majority vote) to pick the best sample.

Strategies:
  - best_of_n:      Score each sample, return highest score
  - diverse_beam:   Maximize quality AND diversity
  - tournament:     Pairwise comparison to find winner
"""

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Optional
from loguru import logger

from src.models.base_model import BaseModel, GenerationConfig, ModelResponse


@dataclass
class SampleScore:
    """A scored candidate sample."""
    content: str
    reasoning: Optional[str]
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class ParallelSamplingResult:
    """Result from parallel sampling."""
    best_content: str
    best_reasoning: Optional[str]
    best_score: float
    all_samples: list[SampleScore]
    n_generated: int
    n_successful: int
    selection_method: str


class ParallelSampler:
    """
    Parallel sampling engine.

    Fires N generation requests simultaneously, scores each,
    and returns the best result.

    Useful when:
    - Model is non-deterministic (temperature > 0)
    - Quality varies across samples
    - You want to maximize answer quality per wall-clock second
    """

    def __init__(
        self,
        model: BaseModel,
        scorer: Optional[Callable[[str], float]] = None,
    ):
        self.model = model
        # Default scorer: length-calibrated heuristic
        # Replace with a trained reward model for production
        self._scorer = scorer or self._default_scorer

    # ── Public API ────────────────────────────────────────────────────────────

    async def sample_best_of_n(
        self,
        messages: list[dict],
        n: int = 5,
        config: Optional[GenerationConfig] = None,
        selection: str = "score",       # "score" | "tournament" | "diverse"
    ) -> ParallelSamplingResult:
        """
        Generate N samples in parallel, return best.

        Args:
            messages:   Input messages.
            n:          Number of parallel samples.
            config:     Generation config (temperature should be > 0).
            selection:  Selection strategy.
        """
        config = config or GenerationConfig(
            max_tokens=4096,
            temperature=0.8,
            n=1,                    # We parallelize manually for control
        )

        logger.info(
            f"Parallel sampling: n={n} | "
            f"model={self.model.model_name} | "
            f"strategy={selection}"
        )

        # Launch all samples concurrently
        tasks = [
            self._safe_generate(messages, config, idx=i)
            for i in range(n)
        ]
        results = await asyncio.gather(*tasks)

        # Separate successes from failures
        successful = [r for r in results if r is not None]
        n_failed = n - len(successful)

        if n_failed > 0:
            logger.warning(f"{n_failed}/{n} samples failed")

        if not successful:
            raise RuntimeError("All parallel samples failed")

        # Score each sample
        scored = await self._score_all(successful)

        # Select best
        if selection == "tournament":
            best = await self._tournament_select(scored)
        elif selection == "diverse":
            best = self._diverse_select(scored)
        else:
            best = max(scored, key=lambda s: s.score)

        logger.info(
            f"Best sample score={best.score:.3f} | "
            f"successful={len(successful)}/{n}"
        )

        return ParallelSamplingResult(
            best_content=best.content,
            best_reasoning=best.reasoning,
            best_score=best.score,
            all_samples=scored,
            n_generated=n,
            n_successful=len(successful),
            selection_method=selection,
        )

    async def sample_with_native_n(
        self,
        messages: list[dict],
        n: int = 5,
        config: Optional[GenerationConfig] = None,
    ) -> ParallelSamplingResult:
        """
        Use model's native n parameter (one API call, N completions).
        More efficient for models that support it natively.
        """
        config = config or GenerationConfig(
            max_tokens=4096,
            temperature=0.8,
            n=n,
        )

        logger.info(
            f"Native parallel sampling: n={n} | "
            f"model={self.model.model_name}"
        )

        response = await self.model.generate(messages, config)

        # Collect all completions
        all_contents = [response.content] + response.alternatives
        samples = [
            ModelResponse(
                content=c,
                model=response.model,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=0,
            )
            for c in all_contents
        ]

        scored = await self._score_all(samples)
        best = max(scored, key=lambda s: s.score)

        return ParallelSamplingResult(
            best_content=best.content,
            best_reasoning=best.reasoning,
            best_score=best.score,
            all_samples=scored,
            n_generated=n,
            n_successful=len(scored),
            selection_method="score",
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _safe_generate(
        self,
        messages: list[dict],
        config: GenerationConfig,
        idx: int = 0,
    ) -> Optional[ModelResponse]:
        """Generate with exception handling."""
        try:
            return await self.model.generate(messages, config)
        except Exception as e:
            logger.warning(f"Sample {idx} failed: {e}")
            return None

    async def _score_all(
        self,
        responses: list[ModelResponse],
    ) -> list[SampleScore]:
        """Score all samples concurrently."""
        tasks = [self._score_one(r) for r in responses]
        return await asyncio.gather(*tasks)

    async def _score_one(self, response: ModelResponse) -> SampleScore:
        """Score a single sample."""
        try:
            score = self._scorer(response.content)
        except Exception as e:
            logger.warning(f"Scoring failed: {e}, defaulting to 0.5")
            score = 0.5

        return SampleScore(
            content=response.content,
            reasoning=response.reasoning_content,
            score=score,
            metadata={
                "model": response.model,
                "tokens": response.completion_tokens,
                "finish_reason": response.finish_reason,
            }
        )

    async def _tournament_select(
        self,
        samples: list[SampleScore],
    ) -> SampleScore:
        """
        Pairwise tournament: compare pairs of responses,
        ask model to pick winner. Winner advances.

        More expensive but more accurate than simple scoring.
        """
        if len(samples) == 1:
            return samples[0]

        competitors = list(samples)

        while len(competitors) > 1:
            next_round = []
            for i in range(0, len(competitors) - 1, 2):
                winner = await self._compare_pair(
                    competitors[i], competitors[i + 1]
                )
                next_round.append(winner)
            # Handle odd one out
            if len(competitors) % 2 == 1:
                next_round.append(competitors[-1])
            competitors = next_round

        return competitors[0]

    async def _compare_pair(
        self,
        a: SampleScore,
        b: SampleScore,
    ) -> SampleScore:
        """Ask the model which of two responses is better."""
        comparison_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator. Compare the two responses "
                    "below and pick the better one. Reply with only 'A' or 'B'."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Response A:\n{a.content[:2000]}\n\n"
                    f"Response B:\n{b.content[:2000]}\n\n"
                    "Which response is better? Reply A or B only."
                )
            }
        ]
        try:
            from src.models.base_model import GenerationConfig
            resp = await self.model.generate(
                comparison_messages,
                GenerationConfig(max_tokens=5, temperature=0.0)
            )
            choice = resp.content.strip().upper()
            return a if choice.startswith("A") else b
        except Exception:
            # On failure, fall back to score comparison
            return a if a.score >= b.score else b

    def _diverse_select(self, samples: list[SampleScore]) -> SampleScore:
        """
        Select sample that balances quality (score) and diversity.
        Penalizes samples that are too similar to higher-scoring ones.
        """
        if len(samples) == 1:
            return samples[0]

        # Sort by score descending
        sorted_samples = sorted(samples, key=lambda s: s.score, reverse=True)
        selected = [sorted_samples[0]]

        for candidate in sorted_samples[1:]:
            # Check similarity to already-selected samples
            is_diverse = all(
                self._text_similarity(candidate.content, sel.content) < 0.7
                for sel in selected
            )
            if is_diverse:
                selected.append(candidate)

        # Return highest-scoring diverse sample
        return max(selected, key=lambda s: s.score)

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Simple Jaccard similarity on word bigrams."""
        def bigrams(text: str) -> set:
            words = text.lower().split()
            return {(words[i], words[i+1]) for i in range(len(words)-1)}

        bg_a = bigrams(a)
        bg_b = bigrams(b)
        if not bg_a or not bg_b:
            return 0.0
        intersection = bg_a & bg_b
        union = bg_a | bg_b
        return len(intersection) / len(union)

    @staticmethod
    def _default_scorer(content: str) -> float:
        """
        Heuristic quality scorer.
        Replace with a trained Process Reward Model for production.

        Signals:
        - Length (longer generally more thorough, up to a point)
        - Presence of structured reasoning markers
        - Absence of uncertainty/refusal markers
        """
        score = 0.5

        # Length score (normalized, peaks at ~500 words)
        word_count = len(content.split())
        length_score = min(word_count / 500, 1.0) * 0.3
        score += length_score

        # Reasoning structure markers
        structure_markers = [
            "step", "first", "second", "third", "therefore",
            "because", "thus", "since", "given that", "conclude",
        ]
        structure_count = sum(
            1 for m in structure_markers
            if m in content.lower()
        )
        score += min(structure_count / 5, 1.0) * 0.2

        # Penalize refusal/uncertainty
        refusal_markers = [
            "i cannot", "i'm not able", "i don't know",
            "i'm uncertain", "i can't determine"
        ]
        if any(m in content.lower() for m in refusal_markers):
            score -= 0.3

        return max(0.0, min(1.0, score))