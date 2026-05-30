"""
Tree of Thoughts (ToT) implementation.
Yao et al. (2023): https://arxiv.org/abs/2305.10601

Explores a tree of reasoning steps. At each node:
  1. Generate candidate next thoughts (branching)
  2. Evaluate each thought (scoring)
  3. Prune low-scoring branches (beam search or greedy)
  4. Expand promising branches

Search strategies:
  - BFS (breadth-first): explore all nodes at depth D before D+1
  - DFS (depth-first):   go deep on most promising path first
  - Beam search:         keep top-K candidates at each depth
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from loguru import logger

from src.models.base_model import BaseModel, GenerationConfig


class SearchStrategy(Enum):
    BFS = "bfs"
    DFS = "dfs"
    BEAM = "beam"


# ── Node ──────────────────────────────────────────────────────────────────────

@dataclass
class ThoughtNode:
    """
    A single node in the thought tree.

    Each node represents one reasoning step.
    Children are candidate next steps.
    """
    thought: str                                 # This step's reasoning
    depth: int = 0                               # Distance from root
    score: float = 0.0                           # Evaluator score
    parent: Optional["ThoughtNode"] = None
    children: list["ThoughtNode"] = field(default_factory=list)
    is_terminal: bool = False                    # True if this is a final answer
    node_id: str = ""

    def __post_init__(self):
        if not self.node_id:
            import uuid
            self.node_id = str(uuid.uuid4())[:8]

    @property
    def path(self) -> list["ThoughtNode"]:
        """Return all nodes from root to this node."""
        if self.parent is None:
            return [self]
        return self.parent.path + [self]

    @property
    def path_text(self) -> str:
        """Full reasoning path as text."""
        return "\n\n".join(
            f"Step {i+1}: {node.thought}"
            for i, node in enumerate(self.path)
        )

    @property
    def cumulative_score(self) -> float:
        """Average score along path."""
        path = self.path
        if not path:
            return 0.0
        return sum(n.score for n in path) / len(path)


# ── Prompts ───────────────────────────────────────────────────────────────────

THOUGHT_GENERATION_PROMPT = """\
You are solving a problem step by step.

Problem: {problem}

{context_section}
Current reasoning path:
{current_path}

Generate {n_candidates} distinct, promising NEXT STEPS in the reasoning.
Each step should advance toward a solution in a different way.

Format each step as:
STEP: <your reasoning step here>

Generate exactly {n_candidates} steps, each starting with 'STEP:'.
"""

THOUGHT_EVALUATION_PROMPT = """\
You are evaluating the quality of a reasoning step for solving a problem.

Problem: {problem}

Reasoning path so far:
{current_path}

Next step being evaluated:
{thought}

Rate this step on a scale of 0.0 to 1.0 based on:
- Correctness (is this step logically valid?)
- Progress (does it meaningfully advance toward the solution?)
- Clarity (is it clear and well-reasoned?)
- Promise (does this path look like it will lead to the answer?)

Output ONLY a number between 0.0 and 1.0. Nothing else.
"""

TERMINAL_CHECK_PROMPT = """\
Given this reasoning path, determine if we have reached a complete final answer.

Problem: {problem}

Complete reasoning path:
{path_text}

Has this path produced a complete, satisfactory answer to the problem?
Answer ONLY with 'YES' or 'NO'.
"""

ANSWER_EXTRACTION_PROMPT = """\
Based on the following reasoning path, provide the final answer to the problem.

Problem: {problem}

Reasoning path:
{path_text}

Final answer (be concise and direct):
"""


# ── Tree of Thoughts Engine ───────────────────────────────────────────────────

@dataclass
class ToTResult:
    """Result from Tree of Thoughts search."""
    answer: str
    best_path: list[ThoughtNode]
    best_score: float
    total_nodes_explored: int
    search_strategy: str
    all_terminal_nodes: list[ThoughtNode] = field(default_factory=list)


class TreeOfThoughts:
    """
    Tree of Thoughts reasoning engine.

    Explores multiple reasoning paths simultaneously,
    scoring each step and pruning unpromising branches.

    Best for:
    - Complex multi-step math problems
    - Logic puzzles
    - Strategic planning tasks
    - Problems where early mistakes cascade
    """

    def __init__(
        self,
        model: BaseModel,
        branching_factor: int = 3,
        max_depth: int = 4,
        beam_width: int = 2,
        search_strategy: SearchStrategy = SearchStrategy.BEAM,
    ):
        self.model = model
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.search_strategy = search_strategy
        self._nodes_explored = 0

    # ── Public API ────────────────────────────────────────────────────────────

    async def solve(
        self,
        problem: str,
        context: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> ToTResult:
        """
        Run Tree of Thoughts search to solve a problem.

        Args:
            problem:    The problem statement.
            context:    Optional context (e.g., search results).
            config:     Generation config.

        Returns:
            ToTResult with best answer and exploration stats.
        """
        config = config or GenerationConfig(max_tokens=1024, temperature=0.8)
        self._nodes_explored = 0

        logger.info(
            f"ToT search | strategy={self.search_strategy.value} | "
            f"branching={self.branching_factor} | depth={self.max_depth} | "
            f"beam_width={self.beam_width}"
        )

        # Root node (empty path so far)
        root = ThoughtNode(
            thought=f"Problem: {problem}",
            depth=0,
            score=1.0,
        )

        if self.search_strategy == SearchStrategy.BEAM:
            terminal_nodes = await self._beam_search(
                root, problem, context, config
            )
        elif self.search_strategy == SearchStrategy.BFS:
            terminal_nodes = await self._bfs(root, problem, context, config)
        else:
            terminal_nodes = await self._dfs(root, problem, context, config)

        if not terminal_nodes:
            logger.warning("No terminal nodes found, using deepest node")
            terminal_nodes = [root]

        # Pick best terminal node by cumulative score
        best_node = max(terminal_nodes, key=lambda n: n.cumulative_score)

        # Extract final answer
        answer = await self._extract_answer(
            problem, best_node.path_text, config
        )

        logger.info(
            f"ToT complete | nodes_explored={self._nodes_explored} | "
            f"terminal_nodes={len(terminal_nodes)} | "
            f"best_score={best_node.cumulative_score:.3f}"
        )

        return ToTResult(
            answer=answer,
            best_path=best_node.path,
            best_score=best_node.cumulative_score,
            total_nodes_explored=self._nodes_explored,
            search_strategy=self.search_strategy.value,
            all_terminal_nodes=terminal_nodes,
        )

    # ── Search strategies ─────────────────────────────────────────────────────

    async def _beam_search(
        self,
        root: ThoughtNode,
        problem: str,
        context: Optional[str],
        config: GenerationConfig,
    ) -> list[ThoughtNode]:
        """
        Beam search: keep top beam_width candidates at each depth level.
        Most efficient for large branching factors.
        """
        beam: list[ThoughtNode] = [root]
        terminal_nodes: list[ThoughtNode] = []

        for depth in range(1, self.max_depth + 1):
            logger.debug(
                f"Beam search depth={depth} | beam_size={len(beam)}"
            )

            # Expand all nodes in current beam
            all_children: list[ThoughtNode] = []
            expand_tasks = [
                self._expand_node(node, problem, context, config)
                for node in beam
            ]
            results = await asyncio.gather(*expand_tasks)
            for children in results:
                all_children.extend(children)

            if not all_children:
                break

            # Check for terminal nodes
            terminal_tasks = [
                self._check_terminal(node, problem, config)
                for node in all_children
            ]
            terminal_flags = await asyncio.gather(*terminal_tasks)

            next_beam = []
            for node, is_terminal in zip(all_children, terminal_flags):
                if is_terminal:
                    node.is_terminal = True
                    terminal_nodes.append(node)
                else:
                    next_beam.append(node)

            if not next_beam:
                break

            # Prune to beam_width
            beam = sorted(
                next_beam,
                key=lambda n: n.score,
                reverse=True
            )[:self.beam_width]

        # If no terminal nodes, use deepest beam nodes
        if not terminal_nodes:
            terminal_nodes = beam

        return terminal_nodes

    async def _bfs(
        self,
        root: ThoughtNode,
        problem: str,
        context: Optional[str],
        config: GenerationConfig,
    ) -> list[ThoughtNode]:
        """BFS: explore all nodes level by level."""
        from collections import deque
        queue: deque[ThoughtNode] = deque([root])
        terminal_nodes: list[ThoughtNode] = []

        while queue:
            node = queue.popleft()

            if node.depth >= self.max_depth:
                terminal_nodes.append(node)
                continue

            children = await self._expand_node(node, problem, context, config)
            for child in children:
                is_terminal = await self._check_terminal(
                    child, problem, config
                )
                if is_terminal:
                    child.is_terminal = True
                    terminal_nodes.append(child)
                else:
                    queue.append(child)

        return terminal_nodes or [root]

    async def _dfs(
        self,
        root: ThoughtNode,
        problem: str,
        context: Optional[str],
        config: GenerationConfig,
        _terminal_nodes: Optional[list] = None,
    ) -> list[ThoughtNode]:
        """DFS: go deep on most promising branch first."""
        if _terminal_nodes is None:
            _terminal_nodes = []

        if root.depth >= self.max_depth:
            _terminal_nodes.append(root)
            return _terminal_nodes

        children = await self._expand_node(root, problem, context, config)
        # Sort by score, explore best first
        children.sort(key=lambda n: n.score, reverse=True)

        for child in children:
            is_terminal = await self._check_terminal(child, problem, config)
            if is_terminal:
                child.is_terminal = True
                _terminal_nodes.append(child)
            else:
                await self._dfs(
                    child, problem, context, config, _terminal_nodes
                )

        return _terminal_nodes

    # ── Node operations ───────────────────────────────────────────────────────

    async def _expand_node(
        self,
        node: ThoughtNode,
        problem: str,
        context: Optional[str],
        config: GenerationConfig,
    ) -> list[ThoughtNode]:
        """Generate candidate next thoughts for a node."""
        self._nodes_explored += 1

        context_section = f"Context:\n{context}\n" if context else ""
        current_path = node.path_text if node.depth > 0 else "None yet."

        prompt = THOUGHT_GENERATION_PROMPT.format(
            problem=problem,
            context_section=context_section,
            current_path=current_path,
            n_candidates=self.branching_factor,
        )

        gen_config = GenerationConfig(
            max_tokens=min(config.max_tokens, 2048),
            temperature=config.temperature,
        )

        try:
            response = await self.model.generate(
                [{"role": "user", "content": prompt}],
                gen_config,
            )
        except Exception as e:
            logger.warning(f"Node expansion failed at depth {node.depth}: {e}")
            return []

        # Parse STEP: markers
        thoughts = self._parse_thoughts(response.content)

        if not thoughts:
            logger.warning(
                f"No thoughts parsed at depth {node.depth}. "
                f"Raw: {response.content[:200]}"
            )
            return []

        # Score all thoughts concurrently
        score_tasks = [
            self._score_thought(node, thought, problem, gen_config)
            for thought in thoughts
        ]
        scores = await asyncio.gather(*score_tasks, return_exceptions=True)

        children = []
        for thought, score in zip(thoughts, scores):
            if isinstance(score, Exception):
                score = 0.5  # Default on failure
            child = ThoughtNode(
                thought=thought,
                depth=node.depth + 1,
                score=float(score),
                parent=node,
            )
            node.children.append(child)
            children.append(child)

        logger.debug(
            f"Expanded node at depth {node.depth}: "
            f"{len(children)} children, "
            f"scores={[f'{c.score:.2f}' for c in children]}"
        )

        return children

    async def _score_thought(
        self,
        parent: ThoughtNode,
        thought: str,
        problem: str,
        config: GenerationConfig,
    ) -> float:
        """Score a candidate thought using the model as evaluator."""
        prompt = THOUGHT_EVALUATION_PROMPT.format(
            problem=problem,
            current_path=parent.path_text,
            thought=thought,
        )

        score_config = GenerationConfig(max_tokens=10, temperature=0.0)
        try:
            response = await self.model.generate(
                [{"role": "user", "content": prompt}],
                score_config,
            )
            score_text = response.content.strip()
            # Extract first float found
            import re
            match = re.search(r'(\d+\.?\d*)', score_text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            return 0.5
        except Exception as e:
            logger.warning(f"Scoring error: {e}")
            return 0.5

    async def _check_terminal(
        self,
        node: ThoughtNode,
        problem: str,
        config: GenerationConfig,
    ) -> bool:
        """Check if a node represents a complete solution."""
        if node.depth < 2:  # Need at least 2 steps
            return False

        prompt = TERMINAL_CHECK_PROMPT.format(
            problem=problem,
            path_text=node.path_text,
        )

        check_config = GenerationConfig(max_tokens=5, temperature=0.0)
        try:
            response = await self.model.generate(
                [{"role": "user", "content": prompt}],
                check_config,
            )
            return "YES" in response.content.upper()
        except Exception:
            return node.depth >= self.max_depth

    async def _extract_answer(
        self,
        problem: str,
        path_text: str,
        config: GenerationConfig,
    ) -> str:
        """Extract clean final answer from reasoning path."""
        prompt = ANSWER_EXTRACTION_PROMPT.format(
            problem=problem,
            path_text=path_text,
        )
        extract_config = GenerationConfig(
            max_tokens=min(config.max_tokens * 2, 4096),
            temperature=0.3,
        )
        response = await self.model.generate(
            [{"role": "user", "content": prompt}],
            extract_config,
        )
        return response.content.strip()

    @staticmethod
    def _parse_thoughts(text: str) -> list[str]:
        """Parse STEP: markers from generated text."""
        import re
        steps = re.findall(r'STEP:\s*(.+?)(?=STEP:|$)', text, re.DOTALL)
        return [s.strip() for s in steps if s.strip()]