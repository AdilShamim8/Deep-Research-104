"""
STaR: Self-Taught Reasoner
Zelikman et al. (2022): https://arxiv.org/abs/2203.14465

Algorithm:
  1. For each problem in dataset:
     a. Try to solve with current model
     b. If correct -> keep (question, reasoning, answer) as training example
     c. If wrong   -> give model the correct answer, ask it to rationalize
                      (generate reasoning that leads to correct answer)
     d. If rationalization correct -> keep as training example
  2. Fine-tune model on collected examples
  3. Repeat until convergence

This bootstraps reasoning capability from a small labeled dataset.
"""

import json
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
from loguru import logger

from src.models.base_model import BaseModel, GenerationConfig
from src.reasoning.chain_of_thought import ChainOfThought, CoTParser
from src.reasoning.verifier import BaseVerifier, RuleBasedVerifier


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TrainingExample:
    """
    A single (question, reasoning, answer) training tuple.
    Collected by STaR algorithm.
    """
    question: str
    reasoning: str
    answer: str
    correct: bool
    source: str = "direct"          # "direct" | "rationalization"
    model_name: str = ""
    iteration: int = 0
    metadata: dict = field(default_factory=dict)

    def to_messages(self) -> list[dict]:
        """Convert to chat-format training messages."""
        return [
            {"role": "user", "content": self.question},
            {
                "role": "assistant",
                "content": (
                    f"<reasoning>\n{self.reasoning}\n</reasoning>\n"
                    f"<answer>\n{self.answer}\n</answer>"
                )
            }
        ]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "reasoning": self.reasoning,
            "answer": self.answer,
            "correct": self.correct,
            "source": self.source,
            "model_name": self.model_name,
            "iteration": self.iteration,
            "metadata": self.metadata,
        }


@dataclass
class STaRDataset:
    """Collection of STaR training examples."""
    examples: list[TrainingExample] = field(default_factory=list)

    @property
    def n_direct(self) -> int:
        return sum(1 for e in self.examples if e.source == "direct")

    @property
    def n_rationalized(self) -> int:
        return sum(1 for e in self.examples if e.source == "rationalization")

    @property
    def accuracy(self) -> float:
        if not self.examples:
            return 0.0
        return sum(1 for e in self.examples if e.correct) / len(self.examples)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in self.examples], f, indent=2)
        logger.info(f"Saved {len(self.examples)} examples to {path}")

    @classmethod
    def load(cls, path: str) -> "STaRDataset":
        with open(path) as f:
            data = json.load(f)
        examples = [TrainingExample(**d) for d in data]
        dataset = cls(examples=examples)
        logger.info(f"Loaded {len(examples)} examples from {path}")
        return dataset

    def to_hf_dataset(self):
        """Convert to HuggingFace Dataset for training."""
        try:
            from datasets import Dataset
            records = []
            for ex in self.examples:
                messages = ex.to_messages()
                records.append({
                    "messages": messages,
                    "question": ex.question,
                    "answer": ex.answer,
                    "source": ex.source,
                    "iteration": ex.iteration,
                })
            return Dataset.from_list(records)
        except ImportError:
            raise ImportError(
                "Install 'datasets' package: pip install datasets"
            )


@dataclass
class STaRConfig:
    """Configuration for STaR training loop."""
    max_iterations: int = 3
    n_samples_per_problem: int = 1
    temperature: float = 0.7
    max_tokens: int = 2048
    rationalization_temperature: float = 0.5
    save_dataset_path: str = "./data/star_dataset.json"
    min_examples_to_finetune: int = 50
    concurrent_requests: int = 5


# ── STaR Trainer ──────────────────────────────────────────────────────────────

class STaRTrainer:
    """
    STaR (Self-Taught Reasoner) implementation.

    Bootstraps reasoning data from a small labeled problem set.
    The collected reasoning data can then be used for SFT.

    Usage:
        problems = [
            {"question": "What is 15% of 80?", "answer": "12"},
            ...
        ]
        trainer = STaRTrainer(model, verifier)
        dataset = await trainer.run(problems, n_iterations=3)
        dataset.save("./data/star_examples.json")
    """

    def __init__(
        self,
        model: BaseModel,
        verifier: BaseVerifier,
        config: Optional[STaRConfig] = None,
    ):
        self.model = model
        self.verifier = verifier
        self.config = config or STaRConfig()
        self._cot = ChainOfThought(model)
        self._parser = CoTParser()

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(
        self,
        problems: list[dict],
        n_iterations: Optional[int] = None,
    ) -> STaRDataset:
        """
        Run STaR algorithm for N iterations.

        Args:
            problems:       List of {"question": ..., "answer": ...} dicts.
            n_iterations:   Override config max_iterations.

        Returns:
            STaRDataset with all collected training examples.
        """
        n_iterations = n_iterations or self.config.max_iterations
        all_examples: list[TrainingExample] = []

        logger.info(
            f"STaR training | problems={len(problems)} | "
            f"iterations={n_iterations} | model={self.model.model_name}"
        )

        for iteration in range(1, n_iterations + 1):
            logger.info(
                f"STaR iteration {iteration}/{n_iterations}"
            )

            # Process problems in batches
            iteration_examples = await self._process_iteration(
                problems, iteration
            )
            all_examples.extend(iteration_examples)

            dataset = STaRDataset(examples=all_examples)
            logger.info(
                f"After iteration {iteration}: "
                f"total={len(all_examples)} | "
                f"direct={dataset.n_direct} | "
                f"rationalized={dataset.n_rationalized} | "
                f"accuracy={dataset.accuracy:.2%}"
            )

            # Save checkpoint
            checkpoint_path = (
                self.config.save_dataset_path
                .replace(".json", f"_iter{iteration}.json")
            )
            dataset.save(checkpoint_path)

            # Early stopping if high accuracy
            if dataset.accuracy >= 0.95:
                logger.info(
                    f"High accuracy {dataset.accuracy:.2%} reached, "
                    f"stopping STaR loop"
                )
                break

        final_dataset = STaRDataset(examples=all_examples)
        final_dataset.save(self.config.save_dataset_path)
        return final_dataset

    async def _process_iteration(
        self,
        problems: list[dict],
        iteration: int,
    ) -> list[TrainingExample]:
        """Process all problems for one STaR iteration."""
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)

        async def process_one(problem: dict) -> Optional[TrainingExample]:
            async with semaphore:
                return await self._process_problem(problem, iteration)

        tasks = [process_one(p) for p in problems]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        examples = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Problem processing error: {result}")
            elif result is not None:
                examples.append(result)

        return examples

    async def _process_problem(
        self,
        problem: dict,
        iteration: int,
    ) -> Optional[TrainingExample]:
        """
        Process a single problem through STaR algorithm.

        1. Generate reasoning + answer
        2. Check if correct
        3. If wrong, try rationalization
        """
        question = problem["question"]
        reference_answer = problem.get("answer", "")

        gen_config = GenerationConfig(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        try:
            # Step 1: Direct generation
            cot_result = await self._cot.generate(
                question=question,
                mode="structured",
                config=gen_config,
            )

            # Step 2: Verify
            verification = await self.verifier.verify(
                question=question,
                answer=cot_result.answer,
                reasoning=cot_result.reasoning,
                reference=reference_answer,
            )

            if verification.is_correct:
                # Keep as direct training example
                return TrainingExample(
                    question=question,
                    reasoning=cot_result.reasoning,
                    answer=cot_result.answer,
                    correct=True,
                    source="direct",
                    model_name=self.model.model_name,
                    iteration=iteration,
                    metadata={"score": verification.score},
                )

            # Step 3: Rationalization (hint with correct answer)
            if reference_answer:
                rationalized = await self._rationalize(
                    question=question,
                    correct_answer=reference_answer,
                    config=gen_config,
                )

                if rationalized:
                    # Verify rationalization makes sense
                    rat_verification = await self.verifier.verify(
                        question=question,
                        answer=rationalized["answer"],
                        reasoning=rationalized["reasoning"],
                        reference=reference_answer,
                    )

                    if rat_verification.is_correct:
                        return TrainingExample(
                            question=question,
                            reasoning=rationalized["reasoning"],
                            answer=rationalized["answer"],
                            correct=True,
                            source="rationalization",
                            model_name=self.model.model_name,
                            iteration=iteration,
                            metadata={
                                "original_answer": cot_result.answer,
                                "score": rat_verification.score,
                            },
                        )

            # Neither direct nor rationalized was correct
            logger.debug(
                f"Problem not solved: {question[:50]}... "
                f"(got: {cot_result.answer[:30]})"
            )
            return None

        except Exception as e:
            logger.warning(f"Problem processing failed: {e}")
            return None

    async def _rationalize(
        self,
        question: str,
        correct_answer: str,
        config: GenerationConfig,
    ) -> Optional[dict]:
        """
        Generate reasoning that leads to the known correct answer.
        Hint: "The answer is X. Show your reasoning."
        """
        rationalization_prompt = (
            f"Question: {question}\n\n"
            f"The correct answer is: {correct_answer}\n\n"
            f"Show the step-by-step reasoning that leads to this answer. "
            f"Work backwards from the answer to construct valid reasoning.\n\n"
            f"<reasoning>\n[Your reasoning here]\n</reasoning>\n"
            f"<answer>\n{correct_answer}\n</answer>"
        )

        rat_config = GenerationConfig(
            max_tokens=config.max_tokens,
            temperature=self.config.rationalization_temperature,
        )

        try:
            response = await self.model.generate(
                [{"role": "user", "content": rationalization_prompt}],
                rat_config,
            )
            reasoning, answer = self._parser.extract_reasoning_and_answer(
                response.content
            )
            return {
                "reasoning": reasoning,
                "answer": answer or correct_answer,
            }
        except Exception as e:
            logger.warning(f"Rationalization failed: {e}")
            return None


# ── SFT Fine-tuning wrapper ───────────────────────────────────────────────────

class SFTTrainer:
    """
    Supervised Fine-Tuning on collected STaR reasoning data.

    Uses HuggingFace TRL's SFTTrainer under the hood.
    Supports LoRA for efficient fine-tuning on consumer hardware.
    """

    def __init__(
        self,
        base_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        output_dir: str = "./models/star_finetuned",
    ):
        self.base_model_name = base_model_name
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.output_dir = output_dir

    def train(
        self,
        dataset: STaRDataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_seq_length: int = 2048,
    ) -> None:
        """Fine-tune model on STaR dataset."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import SFTTrainer as HFSFTTrainer, SFTConfig
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as e:
            raise ImportError(
                f"Training requires additional packages: {e}\n"
                f"pip install transformers trl peft"
            )

        logger.info(
            f"SFT training | model={self.base_model_name} | "
            f"examples={len(dataset.examples)} | "
            f"epochs={num_epochs} | lora={self.use_lora}"
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Apply LoRA if enabled
        if self.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=0.1,
                target_modules=[
                    "q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"
                ],
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        # Convert dataset
        hf_dataset = dataset.to_hf_dataset()

        # Training config
        sft_config = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            bf16=True,
            max_seq_length=max_seq_length,
            dataset_text_field=None,
        )

        trainer = HFSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=hf_dataset,
            tokenizer=tokenizer,
        )

        trainer.train()
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        logger.info(f"SFT training complete. Model saved to {self.output_dir}")