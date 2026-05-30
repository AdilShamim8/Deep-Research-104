"""
Deep Research Pipeline - the main orchestrator.

Implements iterative research loop:
  1. Analyze question -> generate search queries
  2. Search web -> extract content
  3. Synthesize findings -> identify gaps
  4. Generate follow-up queries if gaps exist
  5. Repeat until confident or max iterations
  6. Produce final comprehensive report

This mirrors how a human researcher works:
  read -> think -> search -> read -> think -> conclude
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator
from loguru import logger

from src.models.base_model import BaseModel, GenerationConfig
from src.reasoning.chain_of_thought import ChainOfThought, CoTResult
from src.reasoning.verifier import OutcomeRewardModel
from src.search.web_search import search_engine, SearchResponse
from src.search.content_extractor import extractor, ExtractedContent
from src.search.search_aggregator import SearchAggregator, AggregatedResult
from src.utils.token_counter import TokenCounter
from config.settings import settings


# ── Prompts ───────────────────────────────────────────────────────────────────

QUERY_GENERATION_PROMPT = """\
You are a research strategist. Given a research question, generate
{n_queries} diverse search queries that will help gather comprehensive
information.

Research question: {question}

{prior_context}

Requirements:
- Make queries specific and varied (don't repeat the same angle)
- Cover different aspects: facts, analysis, recent developments, expert opinions
- Use different phrasings and perspectives
- If prior searches found gaps, generate queries to fill them

Output exactly {n_queries} queries, one per line, numbered:
1. <query>
2. <query>
...
"""

GAP_ANALYSIS_PROMPT = """\
You are analyzing research findings to identify gaps.

Original research question: {question}

Information gathered so far:
{context}

Analyze what's been found and identify:
1. What key aspects are well-covered?
2. What important aspects are missing or unclear?
3. What contradictions exist that need clarification?
4. What follow-up searches would improve the answer?

Output format:
<covered>
- [aspect 1]: <brief description>
</covered>

<gaps>
- [gap 1]: <what's missing>
- [gap 2]: <what's missing>
</gaps>

<follow_up_queries>
1. <specific search query to fill gap>
2. <specific search query to fill gap>
</follow_up_queries>

<confidence>low|medium|high</confidence>
"""

SYNTHESIS_PROMPT = """\
You are a world-class research analyst. Synthesize the following
research findings into a comprehensive, accurate answer.

Research question: {question}

Research findings from {n_sources} sources:
{context}

Requirements:
- Be comprehensive and well-organized
- Cite specific sources when making claims (use [Source: domain.com] format)
- Acknowledge uncertainty where it exists
- Distinguish between facts, analysis, and opinion
- Structure with clear headings for complex topics
- Be accurate: only state what the sources support

Provide a thorough research report:
"""

CONFIDENCE_CHECK_PROMPT = """\
Given this research question and the information gathered,
assess the confidence level of the answer.

Question: {question}
Information quality indicators:
- Number of sources: {n_sources}
- Sources with full content: {n_with_content}
- Average relevance score: {avg_relevance:.2f}
- Gaps identified: {n_gaps}

Reply with only: low, medium, or high
"""


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ResearchIteration:
    """Records one iteration of the research loop."""
    iteration_number: int
    queries: list[str]
    search_responses: list[SearchResponse]
    aggregated_results: list[AggregatedResult]
    gaps_identified: list[str]
    confidence: str
    context_tokens: int = 0


@dataclass
class ResearchReport:
    """Final output of the deep research pipeline."""
    question: str
    answer: str
    sources: list[AggregatedResult]
    iterations: list[ResearchIteration]
    confidence: str
    total_sources_found: int
    total_search_time_ms: float
    total_llm_calls: int
    total_tokens_used: int
    synthesis_model: str
    gaps_remaining: list[str] = field(default_factory=list)

    @property
    def source_urls(self) -> list[str]:
        return [s.url for s in self.sources]

    @property
    def formatted_sources(self) -> str:
        lines = ["## Sources Used\n"]
        for i, source in enumerate(self.sources, 1):
            lines.append(
                f"{i}. [{source.title}]({source.url}) - {source.url}"
            )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Format full report as Markdown."""
        sections = [
            f"# Research Report\n",
            f"**Question:** {self.question}\n",
            f"**Confidence:** {self.confidence} | "
            f"**Sources:** {len(self.sources)} | "
            f"**Iterations:** {len(self.iterations)}\n",
            "---\n",
            "## Answer\n",
            self.answer,
            "\n---\n",
            self.formatted_sources,
        ]
        if self.gaps_remaining:
            sections.append("\n## Remaining Uncertainties\n")
            sections.extend(
                f"- {gap}" for gap in self.gaps_remaining
            )
        return "\n".join(sections)


# ── Main pipeline ─────────────────────────────────────────────────────────────

class DeepResearchPipeline:
    """
    Deep Research Pipeline.

    Orchestrates: search -> extract -> reason -> gap_analyze -> repeat

    Example:
        pipeline = DeepResearchPipeline(model)
        report = await pipeline.research(
            "What are the latest developments in fusion energy?"
        )
        print(report.to_markdown())
    """

    def __init__(
        self,
        model: BaseModel,
        judge_model: Optional[BaseModel] = None,
        max_iterations: int = 3,
        queries_per_iteration: int = 3,
        max_sources_per_iteration: int = 5,
        max_context_tokens: int = 24000,
    ):
        self.model = model
        self.judge = judge_model or model
        self.max_iterations = max_iterations
        self.queries_per_iteration = queries_per_iteration
        self.max_sources_per_iteration = max_sources_per_iteration
        self.max_context_tokens = max_context_tokens

        self._cot = ChainOfThought(model)
        self._aggregator = SearchAggregator(
            extractor=extractor,
            max_results_to_fetch=max_sources_per_iteration,
        )
        self._token_counter = TokenCounter(model.model_name)
        self._orm = OutcomeRewardModel(self.judge)

        # Tracking
        self._total_llm_calls = 0
        self._total_tokens = 0

    # ── Public API ────────────────────────────────────────────────────────────

    async def research(
        self,
        question: str,
        fast_mode: bool = False,
    ) -> ResearchReport:
        """
        Run the full deep research pipeline.

        Args:
            question:   Research question.
            fast_mode:  If True, single iteration, fewer queries.
        """
        start_time = time.time()
        logger.info(
            f"Deep research started: {question[:80]}... | "
            f"max_iter={self.max_iterations} | fast={fast_mode}"
        )

        if fast_mode:
            max_iter = 1
            n_queries = 2
        else:
            max_iter = self.max_iterations
            n_queries = self.queries_per_iteration

        iterations: list[ResearchIteration] = []
        all_sources: list[AggregatedResult] = []
        accumulated_context = ""
        confidence = "low"
        gaps: list[str] = []

        for iteration_num in range(1, max_iter + 1):
            logger.info(
                f"Research iteration {iteration_num}/{max_iter}"
            )

            # Step 1: Generate search queries
            queries = await self._generate_queries(
                question,
                n=n_queries,
                prior_context=accumulated_context[:2000] if accumulated_context else None,
                prior_gaps=gaps,
            )
            logger.info(f"Generated {len(queries)} queries: {queries}")

            # Step 2: Execute searches
            search_responses = await search_engine.multi_search(queries)

            # Step 3: Aggregate results
            new_results = await self._aggregator.aggregate(
                query=question,
                search_responses=search_responses,
                fetch_content=True,
                max_total_results=self.max_sources_per_iteration * 2,
            )

            # Merge with existing sources (avoiding duplicates)
            all_sources = self._merge_sources(all_sources, new_results)

            # Build context from best sources
            top_sources = sorted(
                all_sources,
                key=lambda r: r.relevance_score,
                reverse=True
            )[:self.max_sources_per_iteration * iteration_num]

            accumulated_context = self._aggregator.build_context_string(
                top_sources,
                max_total_tokens=self.max_context_tokens,
                model=self.model.model_name,
            )

            context_tokens = self._token_counter.count_tokens(
                accumulated_context
            )

            # Step 4: Analyze gaps
            gaps, confidence = await self._analyze_gaps(
                question, accumulated_context
            )

            iteration = ResearchIteration(
                iteration_number=iteration_num,
                queries=queries,
                search_responses=search_responses,
                aggregated_results=new_results,
                gaps_identified=gaps,
                confidence=confidence,
                context_tokens=context_tokens,
            )
            iterations.append(iteration)

            logger.info(
                f"Iteration {iteration_num}: "
                f"confidence={confidence} | "
                f"gaps={len(gaps)} | "
                f"sources={len(all_sources)} | "
                f"context_tokens={context_tokens}"
            )

            # Stop if confident
            if confidence == "high":
                logger.info(
                    f"High confidence reached after "
                    f"{iteration_num} iterations"
                )
                break

            # Stop if no gaps to fill
            if not gaps:
                logger.info("No gaps identified, stopping")
                break

        # Step 5: Final synthesis
        logger.info(
            f"Synthesizing from {len(all_sources)} sources..."
        )
        answer = await self._synthesize(
            question,
            accumulated_context,
            len(all_sources),
        )

        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"Deep research complete in {total_time:.0f}ms | "
            f"answer_length={len(answer)} chars | "
            f"llm_calls={self._total_llm_calls} | "
            f"tokens={self._total_tokens}"
        )

        return ResearchReport(
            question=question,
            answer=answer,
            sources=all_sources[:self.max_sources_per_iteration * 3],
            iterations=iterations,
            confidence=confidence,
            total_sources_found=len(all_sources),
            total_search_time_ms=total_time,
            total_llm_calls=self._total_llm_calls,
            total_tokens_used=self._total_tokens,
            synthesis_model=self.model.model_name,
            gaps_remaining=gaps,
        )

    async def stream_research(
        self,
        question: str,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream research progress events for real-time UI updates.

        Yields dicts with: type, content, metadata
        """
        yield {
            "type": "start",
            "content": f"Starting research: {question}",
            "metadata": {}
        }

        start_time = time.time()
        iterations: list[ResearchIteration] = []
        all_sources: list[AggregatedResult] = []
        accumulated_context = ""
        gaps: list[str] = []
        confidence = "low"

        for iteration_num in range(1, self.max_iterations + 1):
            yield {
                "type": "iteration_start",
                "content": f"Research iteration {iteration_num}",
                "metadata": {"iteration": iteration_num}
            }

            # Generate queries
            queries = await self._generate_queries(
                question,
                n=self.queries_per_iteration,
                prior_context=accumulated_context[:2000] if accumulated_context else None,
                prior_gaps=gaps,
            )

            yield {
                "type": "queries",
                "content": f"Searching: {', '.join(queries)}",
                "metadata": {"queries": queries}
            }

            # Search
            search_responses = await search_engine.multi_search(queries)
            total_found = sum(
                len(r.results) for r in search_responses
            )

            yield {
                "type": "search_complete",
                "content": f"Found {total_found} results",
                "metadata": {"total_results": total_found}
            }

            # Aggregate
            new_results = await self._aggregator.aggregate(
                query=question,
                search_responses=search_responses,
                fetch_content=True,
                max_total_results=self.max_sources_per_iteration * 2,
            )
            all_sources = self._merge_sources(all_sources, new_results)

            yield {
                "type": "sources_ready",
                "content": f"Processing {len(new_results)} sources",
                "metadata": {
                    "sources": [
                        {"title": r.title, "url": r.url}
                        for r in new_results[:5]
                    ]
                }
            }

            top_sources = sorted(
                all_sources,
                key=lambda r: r.relevance_score,
                reverse=True
            )[:self.max_sources_per_iteration * iteration_num]

            accumulated_context = self._aggregator.build_context_string(
                top_sources,
                max_total_tokens=self.max_context_tokens,
                model=self.model.model_name,
            )

            # Analyze gaps
            gaps, confidence = await self._analyze_gaps(
                question, accumulated_context
            )

            yield {
                "type": "gap_analysis",
                "content": f"Confidence: {confidence} | Gaps: {len(gaps)}",
                "metadata": {
                    "confidence": confidence,
                    "gaps": gaps,
                }
            }

            iterations.append(ResearchIteration(
                iteration_number=iteration_num,
                queries=queries,
                search_responses=search_responses,
                aggregated_results=new_results,
                gaps_identified=gaps,
                confidence=confidence,
            ))

            if confidence == "high" or not gaps:
                break

        # Final synthesis
        yield {
            "type": "synthesizing",
            "content": "Generating final answer...",
            "metadata": {}
        }

        answer = await self._synthesize(
            question, accumulated_context, len(all_sources)
        )

        total_time = (time.time() - start_time) * 1000

        yield {
            "type": "complete",
            "content": answer,
            "metadata": {
                "confidence": confidence,
                "total_sources": len(all_sources),
                "iterations": len(iterations),
                "total_time_ms": total_time,
                "source_urls": [s.url for s in all_sources[:10]],
            }
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _generate_queries(
        self,
        question: str,
        n: int,
        prior_context: Optional[str] = None,
        prior_gaps: Optional[list[str]] = None,
    ) -> list[str]:
        """Generate diverse search queries for the question."""
        prior_section = ""
        if prior_context:
            prior_section = f"\nPrior research context (partial):\n{prior_context}\n"
        if prior_gaps:
            gaps_text = "\n".join(f"- {g}" for g in prior_gaps)
            prior_section += f"\nGaps to fill:\n{gaps_text}\n"

        prompt = QUERY_GENERATION_PROMPT.format(
            question=question,
            n_queries=n,
            prior_context=prior_section,
        )

        config = GenerationConfig(max_tokens=512, temperature=0.7)
        try:
            response = await self.model.generate(
                [{"role": "user", "content": prompt}],
                config,
            )
            self._total_llm_calls += 1
            self._total_tokens += response.total_tokens
            queries = self._parse_numbered_list(response.content)
            return queries[:n] if queries else [question]
        except Exception as e:
            logger.warning(f"Query generation failed: {e}")
            return [question]

    async def _analyze_gaps(
        self,
        question: str,
        context: str,
    ) -> tuple[list[str], str]:
        """Analyze gathered information for gaps."""
        import re
        prompt = GAP_ANALYSIS_PROMPT.format(
            question=question,
            context=context[:6000],  # Limit for this analysis
        )

        config = GenerationConfig(max_tokens=1024, temperature=0.3)
        try:
            response = await self.model.generate(
                [{"role": "user", "content": prompt}],
                config,
            )
            self._total_llm_calls += 1
            self._total_tokens += response.total_tokens

            # Extract gaps
            gaps_match = re.search(
                r'<gaps>(.*?)</gaps>',
                response.content, re.DOTALL
            )
            gaps = []
            if gaps_match:
                gap_text = gaps_match.group(1)
                for line in gap_text.strip().split("\n"):
                    clean = re.sub(r'^[-•*\s]+', '', line).strip()
                    if clean and len(clean) > 5:
                        gaps.append(clean)

            # Extract confidence
            conf_match = re.search(
                r'<confidence>(low|medium|high)</confidence>',
                response.content, re.IGNORECASE
            )
            confidence = (
                conf_match.group(1).lower() if conf_match else "medium"
            )

            return gaps, confidence

        except Exception as e:
            logger.warning(f"Gap analysis failed: {e}")
            return [], "medium"

    async def _synthesize(
        self,
        question: str,
        context: str,
        n_sources: int,
    ) -> str:
        """Synthesize research context into final answer."""
        # Ensure context fits in model's window
        max_context_tokens = (
            self.model.get_context_limit()
            - 2000  # Reserve for prompt + output
        )
        if self._token_counter.count_tokens(context) > max_context_tokens:
            context = self._token_counter.truncate_text_to_tokens(
                context, max_context_tokens
            )

        prompt = SYNTHESIS_PROMPT.format(
            question=question,
            n_sources=n_sources,
            context=context,
        )

        config = GenerationConfig(
            max_tokens=min(
                4096,
                self.model.get_context_limit() - 
                self._token_counter.count_tokens(prompt) - 100
            ),
            temperature=0.5,
        )

        try:
            response = await self.model.generate(
                [{"role": "user", "content": prompt}],
                config,
            )
            self._total_llm_calls += 1
            self._total_tokens += response.total_tokens
            return response.content
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise

    def _merge_sources(
        self,
        existing: list[AggregatedResult],
        new_results: list[AggregatedResult],
    ) -> list[AggregatedResult]:
        """Merge source lists, avoiding URL duplicates."""
        existing_urls = {r.url for r in existing}
        merged = list(existing)
        for result in new_results:
            if result.url not in existing_urls:
                merged.append(result)
                existing_urls.add(result.url)
        return sorted(
            merged,
            key=lambda r: r.relevance_score,
            reverse=True
        )

    @staticmethod
    def _parse_numbered_list(text: str) -> list[str]:
        """Parse numbered list from model output."""
        import re
        items = re.findall(r'^\d+\.\s*(.+)$', text, re.MULTILINE)
        return [item.strip() for item in items if item.strip()]