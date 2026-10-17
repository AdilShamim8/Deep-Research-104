"""
Reasoning engines for Deep Research 104.

Public surface:
    - ChainOfThought, CoTResult         — CoT + self-consistency
    - ParallelSampler, ParallelSamplingResult
    - SequentialSampler, SequentialResult
    - TreeOfThoughts, ToTResult, SearchStrategy
    - OutcomeRewardModel, ProcessRewardModel, RuleBasedVerifier,
      BestOfNWithVerifier
    - InferenceTimeScaler, ScalingStrategy, ScalingBudget, ScalingResult

Each engine implements a distinct inference-time technique; the
pipeline orchestrator (src.pipeline) composes them into a research
loop. All engines are provider-agnostic and accept a BaseModel.

Date tag: 17 October 2026.
"""

from src.reasoning.chain_of_thought import ChainOfThought, CoTResult
from src.reasoning.inference_scaling import (
    InferenceTimeScaler,
    ScalingBudget,
    ScalingResult,
    ScalingStrategy,
)
from src.reasoning.parallel_sampling import ParallelSampler, ParallelSamplingResult
from src.reasoning.sequential_sampling import SequentialResult, SequentialSampler
from src.reasoning.tree_of_thoughts import SearchStrategy, ToTResult, TreeOfThoughts
from src.reasoning.verifier import (
    BestOfNWithVerifier,
    OutcomeRewardModel,
    ProcessRewardModel,
    RuleBasedVerifier,
)

__all__ = [
    # Chain-of-Thought
    "ChainOfThought",
    "CoTResult",
    # Parallel / sequential sampling
    "ParallelSampler",
    "ParallelSamplingResult",
    "SequentialSampler",
    "SequentialResult",
    # Tree of Thoughts
    "TreeOfThoughts",
    "ToTResult",
    "SearchStrategy",
    # Verifiers
    "OutcomeRewardModel",
    "ProcessRewardModel",
    "RuleBasedVerifier",
    "BestOfNWithVerifier",
    # Inference-time scaling
    "InferenceTimeScaler",
    "ScalingStrategy",
    "ScalingBudget",
    "ScalingResult",
]
