"""
Training-time techniques for reasoning models.

Public surface:
    - STaRTrainer, STaRDataset, SFTTrainer
        SFT on self-generated reasoning traces (STaR recipe)
    - RewardModelTrainer, SyntheticPreferenceGenerator, PreferencePair
        ORM + PRM reward modeling
    - GRPOTrainer, GRPOCollector, RLConfig
        Group-Relative Policy Optimization (RL with verifier)
    - SelfRefinementEngine
        Self-refinement loop (generate -> critique -> revise)

These modules require the optional [training] extras (torch,
transformers, trl, peft, datasets). Import lazily where appropriate.

Date tag: 17 October 2026.
"""

from src.training.reward_models import (
    PreferencePair,
    RewardModelTrainer,
    SyntheticPreferenceGenerator,
)
from src.training.rl_trainer import GRPOCollector, GRPOTrainer, RLConfig
from src.training.self_refinement import SelfRefinementEngine
from src.training.star_trainer import SFTTrainer, STaRDataset, STaRTrainer

__all__ = [
    "GRPOCollector",
    # RL
    "GRPOTrainer",
    "PreferencePair",
    "RLConfig",
    # Reward modeling
    "RewardModelTrainer",
    "SFTTrainer",
    "STaRDataset",
    # STaR
    "STaRTrainer",
    # Self-refinement
    "SelfRefinementEngine",
    "SyntheticPreferenceGenerator",
]
