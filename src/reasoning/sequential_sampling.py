"""
Sequential Sampling: iteratively refine a response through
multiple generation passes, each building on the previous.

Techniques implemented:
  1. Sequential refinement   - Each step critiques and improves previous
  2. Stepwise decomposition  - Break task into sub-tasks, solve sequentially
  3. Debate                  - Model argues for/against then synthesizes
  4. Reflection loop         - Detect and correct own errors
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from src.models.base_model import BaseModel, GenerationConfig, ModelResponse
from src.utils.token_counter import TokenCounter


# ── Prompt templates ──────────────────────────────────────────────────────────

CRITIC_SYSTEM = """\
You are a rigorous critic. Given a response, identify:
1. Factual errors or unsupported claims
2. Logical gaps or flawed reasoning  
3. Missing important considerations
4. Areas where the answer is vague or incomplete

Be specific and constructive. Output your critique in this format:
<critique>
- [Issue 1]: <description>
- [Issue 2]: <description>
...
</critique>
<severity>low|medium|high</severity>
"""

REFINEMENT_SYSTEM = """\
You are an expert reviser. You will be given:
1. An original question
2. A draft response
3. A critique of that response

Your job is to produce an improved response that:
- Addresses all critique points
- Maintains what was correct in the original
- Is clearer, more accurate, and more complete

Output only the improved response, nothing else.
"""

DECOMPOSITION_SYSTEM = """\
You are a task decomposition expert. Break the given question into
a sequence of smaller, answerable sub-questions that together
fully address the main question.

Output format:
<subtasks>
1. <sub-question 1>
2. <sub-question 2>
...
</subtasks>
"""

SYNTHESIS_SYSTEM = """\
You are a synthesis expert. Given a question and answers to several
sub-questions, synthesize a comprehensive, coherent final answer.
"""


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RefinementStep:
    """A single step in the sequential refinement process."""
    step_number: int
    content: str
    critique: Optional[str] = None
    critique_severity: str = "none"
    tokens_used: int = 0


@dataclass
class SequentialResult:
    """Result from sequential sampling."""
    final_answer: str
    steps: list[RefinementStep] = field(default_factory=list)
    total_iterations: int = 0
    converged: bool = False
    total_tokens: int = 0
    method: str = "refinement"


# ── Engine ────────────────────────────────────────────────────────────────────

class SequentialSampler:
    """
    Sequential refinement engine.

    Iteratively improves responses using critique-refine loops.
    Stops when critique severity is "low" or max_steps reached.
    """

    def __init__(self, model: BaseModel):
        self.model = model
        self._token_counter = TokenCounter(model.model_name)

    # ── Public API ────────────────────────────────────────────────────────────

    async def refine(
        self,
        question: str,
        context: Optional[str] = None,
        max_steps: int = 3,
        stop_on_low_severity: bool = True,
        config: Optional[GenerationConfig] = None,
    ) -> SequentialResult:
        """
        Iterative critique-and-refine loop.

        Process:
          1. Generate initial response
          2. Critique the response
          3. Refine based on critique
          4. Repeat until convergence or max_steps

        Args:
            question:               The question to answer.
            context:                Optional background context.
            max_steps:              Maximum refinement iterations.
            stop_on_low_severity:   Stop if critique is low severity.
            config:                 Generation config.
        """
        config = config or GenerationConfig(max_tokens=4096, temperature=0.7)
        steps: list[RefinementStep] = []
        total_tokens = 0

        logger.info(
            f"Sequential refinement | max_steps={max_steps} | "
            f"model={self.model.model_name}"
        )

        # Step 1: Initial generation
        initial_messages = self._build_initial_messages(question, context)
        initial_response = await self.model.generate(initial_messages, config)
        total_tokens += initial_response.total_tokens

        current_answer = initial_response.content
        steps.append(RefinementStep(
            step_number=0,
            content=current_answer,
            tokens_used=initial_response.total_tokens,
        ))

        # Refinement loop
        converged = False
        for step_num in range(1, max_steps + 1):
            # Critique current answer
            critique, severity = await self._critique(
                question, current_answer, config
            )
            steps[-1].critique = critique
            steps[-1].critique_severity = severity

            logger.debug(
                f"Step {step_num} critique severity: {severity}"
            )

            if stop_on_low_severity and severity == "low":
                logger.info(f"Converged at step {step_num} (low severity)")
                converged = True
                break

            if step_num >= max_steps:
                break

            # Refine based on critique
            refined_response = await self._refine_once(
                question, current_answer, critique, config
            )
            total_tokens += refined_response.total_tokens
            current_answer = refined_response.content

            steps.append(RefinementStep(
                step_number=step_num,
                content=current_answer,
                tokens_used=refined_response.total_tokens,
            ))

        return SequentialResult(
            final_answer=current_answer,
            steps=steps,
            total_iterations=len(steps),
            converged=converged,
            total_tokens=total_tokens,
            method="refinement",
        )

    async def decompose_and_solve(
        self,
        question: str,
        context: Optional[str] = None,
        max_subtasks: int = 5,
        config: Optional[GenerationConfig] = None,
    ) -> SequentialResult:
        """
        Decompose question into sub-tasks, solve each,
        then synthesize into final answer.

        Effective for complex multi-part questions.
        """
        config = config or GenerationConfig(max_tokens=2048, temperature=0.5)
        steps: list[RefinementStep] = []
        total_tokens = 0

        logger.info(f"Decompose-and-solve | question={question[:80]}...")

        # Step 1: Decompose
        subtasks = await self._decompose(question, context, config)
        subtasks = subtasks[:max_subtasks]

        logger.info(f"Decomposed into {len(subtasks)} sub-tasks")

        # Step 2: Solve each sub-task sequentially
        subtask_answers: list[tuple[str, str]] = []
        for i, subtask in enumerate(subtasks):
            logger.debug(f"Solving sub-task {i+1}/{len(subtasks)}: {subtask}")

            subtask_context = context or ""
            if subtask_answers:
                # Provide previous answers as context
                prev = "\n".join(
                    f"Q{j+1}: {q}\nA{j+1}: {a}"
                    for j, (q, a) in enumerate(subtask_answers)
                )
                subtask_context = (
                    f"{subtask_context}\n\nPrevious answers:\n{prev}"
                )

            messages = self._build_initial_messages(subtask, subtask_context)
            response = await self.model.generate(messages, config)
            total_tokens += response.total_tokens

            subtask_answers.append((subtask, response.content))
            steps.append(RefinementStep(
                step_number=i,
                content=f"Q: {subtask}\nA: {response.content}",
                tokens_used=response.total_tokens,
            ))

        # Step 3: Synthesize
        synthesis = await self._synthesize(
            question, subtask_answers, config
        )
        total_tokens += synthesis.total_tokens
        steps.append(RefinementStep(
            step_number=len(subtasks),
            content=synthesis.content,
            tokens_used=synthesis.total_tokens,
        ))

        return SequentialResult(
            final_answer=synthesis.content,
            steps=steps,
            total_iterations=len(steps),
            converged=True,
            total_tokens=total_tokens,
            method="decompose_and_solve",
        )

    async def debate(
        self,
        question: str,
        context: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> SequentialResult:
        """
        Structured debate: generate pro argument, con argument,
        then synthesize balanced final answer.

        Reduces confirmation bias and improves balanced analysis.
        """
        config = config or GenerationConfig(max_tokens=2048, temperature=0.7)
        steps: list[RefinementStep] = []
        total_tokens = 0

        logger.info(f"Debate mode | question={question[:80]}...")

        context_str = f"Context:\n{context}\n\n" if context else ""

        # Pro argument
        pro_messages = [
            {"role": "system", "content": (
                "You are arguing FOR the most affirmative position on the "
                "given question. Present the strongest possible case."
            )},
            {"role": "user", "content": f"{context_str}Question: {question}"}
        ]
        pro_response = await self.model.generate(pro_messages, config)
        total_tokens += pro_response.total_tokens
        steps.append(RefinementStep(
            step_number=0,
            content=pro_response.content,
            critique="Pro position",
        ))

        # Con argument
        con_messages = [
            {"role": "system", "content": (
                "You are arguing AGAINST or presenting the counterpoint to "
                "the given question. Present the strongest possible counter-case."
            )},
            {"role": "user", "content": f"{context_str}Question: {question}"}
        ]
        con_response = await self.model.generate(con_messages, config)
        total_tokens += con_response.total_tokens
        steps.append(RefinementStep(
            step_number=1,
            content=con_response.content,
            critique="Con position",
        ))

        # Synthesis
        synthesis_messages = [
            {"role": "system", "content": (
                "You are a balanced judge. Given arguments for and against, "
                "synthesize a nuanced, accurate final answer."
            )},
            {"role": "user", "content": (
                f"{context_str}Question: {question}\n\n"
                f"Argument FOR:\n{pro_response.content}\n\n"
                f"Argument AGAINST:\n{con_response.content}\n\n"
                "Synthesize a balanced, accurate final answer:"
            )}
        ]
        synthesis_response = await self.model.generate(synthesis_messages, config)
        total_tokens += synthesis_response.total_tokens
        steps.append(RefinementStep(
            step_number=2,
            content=synthesis_response.content,
            critique="Synthesis",
        ))

        return SequentialResult(
            final_answer=synthesis_response.content,
            steps=steps,
            total_iterations=3,
            converged=True,
            total_tokens=total_tokens,
            method="debate",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _critique(
        self,
        question: str,
        answer: str,
        config: GenerationConfig,
    ) -> tuple[str, str]:
        """Generate critique and extract severity."""
        import re
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Response to critique:\n{answer}"
            )}
        ]
        critique_config = GenerationConfig(
            max_tokens=1024,
            temperature=0.3,
        )
        response = await self.model.generate(messages, critique_config)

        # Extract severity
        severity_match = re.search(
            r'<severity>(low|medium|high)</severity>',
            response.content,
            re.IGNORECASE
        )
        severity = severity_match.group(1).lower() if severity_match else "medium"

        # Extract critique text
        critique_match = re.search(
            r'<critique>(.*?)</critique>',
            response.content, re.DOTALL
        )
        critique_text = (
            critique_match.group(1).strip()
            if critique_match else response.content
        )

        return critique_text, severity

    async def _refine_once(
        self,
        question: str,
        draft: str,
        critique: str,
        config: GenerationConfig,
    ) -> ModelResponse:
        """Produce one refinement pass."""
        messages = [
            {"role": "system", "content": REFINEMENT_SYSTEM},
            {"role": "user", "content": (
                f"Original Question: {question}\n\n"
                f"Draft Response:\n{draft}\n\n"
                f"Critique:\n{critique}\n\n"
                "Please provide an improved response:"
            )}
        ]
        return await self.model.generate(messages, config)

    async def _decompose(
        self,
        question: str,
        context: Optional[str],
        config: GenerationConfig,
    ) -> list[str]:
        """Decompose question into sub-tasks."""
        import re
        context_str = f"Context:\n{context}\n\n" if context else ""
        messages = [
            {"role": "system", "content": DECOMPOSITION_SYSTEM},
            {"role": "user", "content": f"{context_str}Question: {question}"}
        ]
        decomp_config = GenerationConfig(max_tokens=1024, temperature=0.3)
        response = await self.model.generate(messages, decomp_config)

        # Parse numbered list
        subtasks_match = re.search(
            r'<subtasks>(.*?)</subtasks>',
            response.content, re.DOTALL
        )
        if subtasks_match:
            text = subtasks_match.group(1)
        else:
            text = response.content

        lines = text.strip().split("\n")
        subtasks = []
        for line in lines:
            cleaned = re.sub(r'^\d+\.\s*', '', line.strip())
            if cleaned and len(cleaned) > 5:
                subtasks.append(cleaned)

        return subtasks or [question]

    async def _synthesize(
        self,
        question: str,
        subtask_answers: list[tuple[str, str]],
        config: GenerationConfig,
    ) -> ModelResponse:
        """Synthesize sub-task answers into final answer."""
        qa_text = "\n\n".join(
            f"Sub-question {i+1}: {q}\nAnswer: {a}"
            for i, (q, a) in enumerate(subtask_answers)
        )
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user", "content": (
                f"Main question: {question}\n\n"
                f"Sub-question answers:\n{qa_text}\n\n"
                "Provide a comprehensive, synthesized final answer:"
            )}
        ]
        synth_config = GenerationConfig(max_tokens=4096, temperature=0.5)
        return await self.model.generate(messages, synth_config)

    def _build_initial_messages(
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
                    "You are an expert analyst. Provide a thorough, "
                    "accurate, well-reasoned response."
                )
            },
            {"role": "user", "content": content}
        ]