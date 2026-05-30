"""
Reinforcement Learning Training for Reasoning Models.

Implements GRPO (Group Relative Policy Optimization) -
the algorithm used to train DeepSeek-R1.

GRPO vs PPO:
  - No separate value/critic model needed
  - Uses group of samples to estimate baseline
  - More stable for reasoning tasks
  - Lower memory footprint

Reference: DeepSeek-R1 paper (2025)
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional, Callable
from loguru import logger

from src.models.base_model import BaseModel, GenerationConfig
from src.reasoning.verifier import BaseVerifier


@dataclass
class RLConfig:
    """Configuration for RL training."""
    # GRPO hyperparameters
    group_size: int = 8              # G in GRPO - samples per question
    temperature: float = 0.8        # Sampling temperature
    max_tokens: int = 2048
    clip_epsilon: float = 0.2       # PPO clipping
    kl_coefficient: float = 0.04    # KL penalty weight
    entropy_coefficient: float = 0.01

    # Training
    learning_rate: float = 1e-6
    batch_size: int = 4
    num_epochs: int = 1
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0

    # Reward shaping
    correct_reward: float = 1.0
    format_reward: float = 0.1      # Reward for using correct format tags
    length_penalty: float = 0.0001  # Penalty per token (avoid verbosity)

    # Checkpointing
    save_steps: int = 50
    output_dir: str = "./models/rl_trained"


@dataclass
class RLEpisode:
    """A single RL training episode (question + sampled response)."""
    question: str
    response: str
    reasoning: str
    reward: float
    is_correct: bool
    token_count: int
    format_valid: bool
    group_id: int = 0               # Which GRPO group this belongs to


class GRPOCollector:
    """
    Collect GRPO training data by:
    1. Sampling G responses per question
    2. Scoring each with verifier
    3. Computing group-relative advantages
    """

    def __init__(
        self,
        model: BaseModel,
        verifier: BaseVerifier,
        config: RLConfig,
    ):
        self.model = model
        self.verifier = verifier
        self.config = config

    async def collect_episodes(
        self,
        questions: list[dict],       # [{"question": ..., "answer": ...}]
        max_concurrent: int = 4,
    ) -> tuple[list[RLEpisode], dict]:
        """
        Collect episodes for all questions.

        Returns:
            (episodes, stats)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def collect_one(
            problem: dict, group_id: int
        ) -> list[RLEpisode]:
            async with semaphore:
                return await self._collect_group(problem, group_id)

        tasks = [
            collect_one(p, i) for i, p in enumerate(questions)
        ]
        all_groups = await asyncio.gather(*tasks, return_exceptions=True)

        episodes: list[RLEpisode] = []
        for group in all_groups:
            if isinstance(group, Exception):
                logger.warning(f"Group collection failed: {group}")
            else:
                episodes.extend(group)

        # Compute statistics
        stats = self._compute_stats(episodes)
        logger.info(
            f"Collected {len(episodes)} episodes | "
            f"accuracy={stats['accuracy']:.2%} | "
            f"avg_reward={stats['avg_reward']:.3f}"
        )
        return episodes, stats

    async def _collect_group(
        self,
        problem: dict,
        group_id: int,
    ) -> list[RLEpisode]:
        """Collect G episodes for a single question."""
        question = problem["question"]
        reference = problem.get("answer", "")

        config = GenerationConfig(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        # Sample G responses
        messages = [{"role": "user", "content": question}]
        tasks = [
            self._safe_generate(messages, config)
            for _ in range(self.config.group_size)
        ]
        responses = await asyncio.gather(*tasks)
        valid_responses = [r for r in responses if r is not None]

        if not valid_responses:
            return []

        # Score each response
        episodes = []
        rewards = []

        for response in valid_responses:
            import re
            think_match = re.search(
                r'<think>(.*?)</think>', response, re.DOTALL
            )
            reasoning = think_match.group(1).strip() if think_match else ""
            answer = re.sub(
                r'<think>.*?</think>', '', response, flags=re.DOTALL
            ).strip()

            # Check format (has think tags?)
            format_valid = bool(think_match)

            # Verify correctness
            verification = await self.verifier.verify(
                question=question,
                answer=answer,
                reasoning=reasoning,
                reference=reference,
            )

            # Compute reward
            reward = self._compute_reward(
                is_correct=verification.is_correct,
                score=verification.score,
                format_valid=format_valid,
                token_count=len(response.split()),
            )
            rewards.append(reward)

            episodes.append(RLEpisode(
                question=question,
                response=response,
                reasoning=reasoning,
                reward=reward,
                is_correct=verification.is_correct,
                token_count=len(response.split()),
                format_valid=format_valid,
                group_id=group_id,
            ))

        # Compute GRPO advantages (reward - group mean) / group std
        if rewards:
            import statistics
            mean_reward = sum(rewards) / len(rewards)
            std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 1.0
            std_reward = max(std_reward, 1e-8)  # Avoid division by zero

            for ep, reward in zip(episodes, rewards):
                ep.reward = (reward - mean_reward) / std_reward

        return episodes

    def _compute_reward(
        self,
        is_correct: bool,
        score: float,
        format_valid: bool,
        token_count: int,
    ) -> float:
        """Compute shaped reward for an episode."""
        reward = 0.0

        # Primary: correctness
        if is_correct:
            reward += self.config.correct_reward
        else:
            reward += score * 0.3  # Partial credit for near-correct

        # Format reward (encourage using reasoning tags)
        if format_valid:
            reward += self.config.format_reward

        # Length penalty (discourage verbosity)
        reward -= token_count * self.config.length_penalty

        return reward

    async def _safe_generate(
        self,
        messages: list[dict],
        config: GenerationConfig,
    ) -> Optional[str]:
        try:
            response = await self.model.generate(messages, config)
            return response.content
        except Exception as e:
            logger.warning(f"Generation failed in GRPO: {e}")
            return None

    @staticmethod
    def _compute_stats(episodes: list[RLEpisode]) -> dict:
        if not episodes:
            return {"accuracy": 0, "avg_reward": 0}
        return {
            "accuracy": sum(1 for e in episodes if e.is_correct) / len(episodes),
            "avg_reward": sum(e.reward for e in episodes) / len(episodes),
            "format_rate": sum(1 for e in episodes if e.format_valid) / len(episodes),
            "n_episodes": len(episodes),
        }


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) trainer.

    Trains a language model to reason better using RL
    with a verifier as the reward signal.

    This is the core algorithm behind DeepSeek-R1's training.
    """

    def __init__(
        self,
        policy_model_name: str,
        reference_model_name: str,
        config: RLConfig,
    ):
        self.policy_model_name = policy_model_name
        self.reference_model_name = reference_model_name
        self.config = config

    def train(
        self,
        episodes: list[RLEpisode],
    ) -> dict:
        """
        Run GRPO update on collected episodes.

        Args:
            episodes:   Collected episodes with GRPO advantages.

        Returns:
            Training statistics dict.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("pip install torch transformers")

        logger.info(
            f"GRPO update | episodes={len(episodes)} | "
            f"model={self.policy_model_name}"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.policy_model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        policy_model = AutoModelForCausalLM.from_pretrained(
            self.policy_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.reference_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        ref_model.eval()

        optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=self.config.learning_rate,
        )

        total_loss = 0.0
        steps = 0

        policy_model.train()
        for i in range(
            0, len(episodes), self.config.batch_size
        ):
            batch = episodes[i:i + self.config.batch_size]

            batch_loss = torch.tensor(0.0, requires_grad=True)

            for episode in batch:
                full_text = (
                    f"Question: {episode.question}\n"
                    f"Answer: {episode.response}"
                )
                inputs = tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_tokens,
                ).to(policy_model.device)

                with torch.no_grad():
                    ref_outputs = ref_model(**inputs)
                    ref_log_probs = torch.nn.functional.log_softmax(
                        ref_outputs.logits, dim=-1
                    )

                policy_outputs = policy_model(**inputs)
                policy_log_probs = torch.nn.functional.log_softmax(
                    policy_outputs.logits, dim=-1
                )

                # KL divergence penalty
                kl_div = (
                    torch.exp(policy_log_probs) *
                    (policy_log_probs - ref_log_probs)
                ).sum(-1).mean()

                # Policy gradient loss with GRPO advantage
                advantage = torch.tensor(episode.reward)
                pg_loss = -(advantage * policy_log_probs.mean())

                episode_loss = (
                    pg_loss +
                    self.config.kl_coefficient * kl_div
                )
                batch_loss = batch_loss + episode_loss

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(),
                self.config.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()

            total_loss += batch_loss.item()
            steps += 1

            if steps % self.config.save_steps == 0:
                checkpoint_dir = (
                    f"{self.config.output_dir}/checkpoint-{steps}"
                )
                policy_model.save_pretrained(checkpoint_dir)
                logger.info(f"Checkpoint saved: {checkpoint_dir}")

        policy_model.save_pretrained(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)

        avg_loss = total_loss / max(steps, 1)
        logger.info(
            f"GRPO training complete | "
            f"avg_loss={avg_loss:.4f} | steps={steps}"
        )
        return {"avg_loss": avg_loss, "steps": steps}