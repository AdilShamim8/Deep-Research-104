"""
Reward Model Training.

Implements:
  - Outcome Reward Model (ORM): trained binary classifier on (question, answer)
  - Process Reward Model (PRM): trained step-level scorer
  - Bradley-Terry preference model from human feedback pairs

Both use a pretrained LLM backbone with a value head added.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from loguru import logger


@dataclass
class PreferencePair:
    """
    Human preference pair: chosen (better) vs rejected (worse) response.
    Standard format for reward model training.
    """
    question: str
    chosen: str                  # Better response
    rejected: str                # Worse response
    chosen_reasoning: str = ""
    rejected_reasoning: str = ""
    source: str = "human"        # "human" | "synthetic" | "llm_judge"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_reasoning": self.chosen_reasoning,
            "rejected_reasoning": self.rejected_reasoning,
            "source": self.source,
        }


@dataclass
class StepAnnotation:
    """
    PRM training annotation: correctness label for each reasoning step.
    """
    question: str
    steps: list[str]
    labels: list[int]            # 1 = correct, 0 = incorrect
    final_answer: str
    is_correct: bool
    annotator: str = "llm_judge"

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "steps": self.steps,
            "labels": self.labels,
            "final_answer": self.final_answer,
            "is_correct": self.is_correct,
            "annotator": self.annotator,
        }


class SyntheticPreferenceGenerator:
    """
    Generate synthetic preference pairs for reward model training.

    Uses a strong model (teacher) to judge between responses
    from a weaker model (student), creating preference data
    without human annotation.
    """

    def __init__(
        self,
        teacher_model,      # Strong judge model (e.g., o3)
        student_model,      # Model being evaluated (e.g., local 7B)
        n_student_samples: int = 4,
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.n_samples = n_student_samples

    async def generate_pairs(
        self,
        questions: list[str],
    ) -> list[PreferencePair]:
        """
        For each question:
        1. Generate N responses from student
        2. Score each with teacher
        3. Create chosen/rejected pairs from highest/lowest scored
        """
        import asyncio
        from src.reasoning.parallel_sampling import ParallelSampler
        from src.reasoning.verifier import OutcomeRewardModel

        sampler = ParallelSampler(self.student)
        orm = OutcomeRewardModel(self.teacher)

        pairs: list[PreferencePair] = []

        for question in questions:
            try:
                messages = [{"role": "user", "content": question}]

                # Sample N responses from student
                result = await sampler.sample_best_of_n(
                    messages=messages,
                    n=self.n_samples,
                    selection="score",
                )

                if len(result.all_samples) < 2:
                    continue

                # Sort by score
                sorted_samples = sorted(
                    result.all_samples,
                    key=lambda s: s.score,
                    reverse=True,
                )

                chosen = sorted_samples[0]
                rejected = sorted_samples[-1]

                # Only create pair if there's meaningful quality difference
                if chosen.score - rejected.score > 0.2:
                    pairs.append(PreferencePair(
                        question=question,
                        chosen=chosen.content,
                        rejected=rejected.content,
                        source="synthetic",
                        metadata={
                            "chosen_score": chosen.score,
                            "rejected_score": rejected.score,
                        }
                    ))

            except Exception as e:
                logger.warning(f"Pair generation failed: {e}")
                continue

        logger.info(
            f"Generated {len(pairs)} preference pairs "
            f"from {len(questions)} questions"
        )
        return pairs


class RewardModelTrainer:
    """
    Train an Outcome Reward Model (ORM) from preference pairs.

    Architecture: base LLM + linear value head
    Training: Bradley-Terry model with pairwise ranking loss

    The trained ORM can score any (question, answer) pair
    and return a scalar reward between 0 and 1.
    """

    def __init__(
        self,
        base_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        output_dir: str = "./models/reward_model",
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir

    def train(
        self,
        pairs: list[PreferencePair],
        num_epochs: int = 2,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        max_seq_length: int = 2048,
    ) -> None:
        """Train ORM on preference pairs."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import Dataset, DataLoader
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                get_linear_schedule_with_warmup,
            )
        except ImportError as e:
            raise ImportError(f"Training requires PyTorch/Transformers: {e}")

        logger.info(
            f"ORM training | pairs={len(pairs)} | "
            f"model={self.base_model_name}"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load as sequence classifier (value head = 1 output)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        class PairDataset(Dataset):
            def __init__(self, pairs, tokenizer, max_length):
                self.pairs = pairs
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.pairs)

            def __getitem__(self, idx):
                pair = self.pairs[idx]
                chosen_text = (
                    f"Question: {pair.question}\n"
                    f"Answer: {pair.chosen}"
                )
                rejected_text = (
                    f"Question: {pair.question}\n"
                    f"Answer: {pair.rejected}"
                )
                chosen_enc = self.tokenizer(
                    chosen_text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                rejected_enc = self.tokenizer(
                    rejected_text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                return {
                    "chosen_input_ids": chosen_enc["input_ids"].squeeze(),
                    "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(),
                    "rejected_input_ids": rejected_enc["input_ids"].squeeze(),
                    "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(),
                }

        dataset = PairDataset(pairs, tokenizer, max_seq_length)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate
        )
        total_steps = len(dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

        model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad()

                chosen_rewards = model(
                    input_ids=batch["chosen_input_ids"].to(model.device),
                    attention_mask=batch["chosen_attention_mask"].to(model.device),
                ).logits

                rejected_rewards = model(
                    input_ids=batch["rejected_input_ids"].to(model.device),
                    attention_mask=batch["rejected_attention_mask"].to(model.device),
                ).logits

                # Bradley-Terry pairwise ranking loss
                loss = -torch.nn.functional.logsigmoid(
                    chosen_rewards - rejected_rewards
                ).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs} | loss={avg_loss:.4f}")

        # Save
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        logger.info(f"ORM saved to {self.output_dir}")