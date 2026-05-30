"""
Meta Chain-of-Thought (Meta-CoT): Internalizing Search into Reasoning.

The model learns to interleave web search calls within its reasoning trace,
deciding when to search, what to search for, and how to integrate findings.

Based on: "Meta Chain-of-Thought" - the idea that reasoning models
should natively decide when to retrieve external information
rather than having retrieval forced as a pre-step.

The model outputs special tags:
  <search>query here</search>          -> triggers web search
  <reading>url here</reading>          -> triggers content fetch
  <think>reasoning here</think>        -> internal reasoning
  <answer>final answer here</answer>   -> final response

This creates a fully agentic research loop.
"""

import re
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator
from loguru import logger

from src.models.base_model import BaseModel, GenerationConfig
from src.search.web_search import search_engine, SearchResponse
from src.search.content_extractor import extractor
from src.search.search_aggregator import SearchAggregator
from src.utils.token_counter import TokenCounter
from config.settings import settings


# ── System prompt that teaches model to use search ────────────────────────────

META_COT_SYSTEM = """\
You are a research assistant with access to web search and content reading.
You reason step by step, and when you need information you don't have,
you use special tags to retrieve it.

AVAILABLE ACTIONS:
- <search>your query</search>         Search the web for information
- <reading>URL</reading>              Read the full content of a specific URL
- <think>your thoughts</think>        Internal reasoning (not shown to user)
- <answer>your final answer</answer>  Provide your final answer

RULES:
1. Always think before searching: use <think> to plan your approach
2. Search when you need specific facts, recent information, or verification
3. Read specific URLs when a search result looks highly relevant
4. After gathering information, synthesize before answering
5. Cite sources in your answer using [Source: URL] format
6. If you're confident in your knowledge, you can answer without searching
7. Maximum {max_searches} searches allowed per response

WORKFLOW EXAMPLE:
<think>
The user is asking about X. I need to find current information about Y.
Let me search for that.
</think>
<search>current Y statistics 2024</search>
[Search results will appear here]
<think>
The results show... I should also check Z. Let me search.
</think>
<search>Z detailed explanation</search>
[Search results will appear here]
<think>
Now I have enough information. Let me synthesize.
</think>
<answer>
Based on my research... [detailed answer with citations]
</answer>
"""

SEARCH_RESULT_TEMPLATE = """\
[SEARCH RESULTS for: "{query}"]
{results}
[END SEARCH RESULTS]
"""

READING_RESULT_TEMPLATE = """\
[CONTENT from: {url}]
{content}
[END CONTENT]
"""


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MetaCoTAction:
    """A single action taken during Meta-CoT reasoning."""
    action_type: str          # "think", "search", "reading", "answer"
    content: str              # The content/query
    result: Optional[str] = None  # Result of search/reading
    timestamp_ms: float = 0.0


@dataclass
class MetaCoTResult:
    """Result from Meta-CoT reasoning."""
    answer: str
    reasoning_trace: str
    actions: list[MetaCoTAction]
    total_searches: int
    total_readings: int
    sources_used: list[str]
    total_tokens: int
    total_time_ms: float


# ── Meta-CoT Engine ───────────────────────────────────────────────────────────

class MetaCoT:
    """
    Meta Chain-of-Thought: reasoning model with internalized search.

    The model decides when to search, what to search for, and
    integrates retrieved information into its reasoning trace.

    This is the most powerful research mode - the model drives
    the entire research process rather than following a fixed loop.
    """

    def __init__(
        self,
        model: BaseModel,
        max_searches: int = 5,
        max_readings: int = 3,
        max_iterations: int = 10,
        search_results_per_query: int = 5,
        max_content_tokens: int = 2000,
    ):
        self.model = model
        self.max_searches = max_searches
        self.max_readings = max_readings
        self.max_iterations = max_iterations
        self.search_results_per_query = search_results_per_query
        self.max_content_tokens = max_content_tokens

        self._token_counter = TokenCounter(model.model_name)
        self._aggregator = SearchAggregator(extractor=extractor)

    # ── Public API ────────────────────────────────────────────────────────────

    async def reason(
        self,
        question: str,
        config: Optional[GenerationConfig] = None,
    ) -> MetaCoTResult:
        """
        Run Meta-CoT reasoning on a question.

        The model will interleave thinking and searching until
        it produces a final <answer> tag.
        """
        start_time = time.time()
        config = config or GenerationConfig(
            max_tokens=2048,
            temperature=0.7,
        )

        logger.info(
            f"Meta-CoT started | model={self.model.model_name} | "
            f"max_searches={self.max_searches}"
        )

        system_prompt = META_COT_SYSTEM.format(
            max_searches=self.max_searches
        )

        # Conversation history - grows as we add search results
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        actions: list[MetaCoTAction] = []
        sources_used: list[str] = []
        total_tokens = 0
        search_count = 0
        reading_count = 0
        full_reasoning_trace = ""

        # Agentic loop
        for iteration in range(self.max_iterations):
            logger.debug(
                f"Meta-CoT iteration {iteration + 1} | "
                f"searches={search_count}/{self.max_searches}"
            )

            # Validate that messages fit in context
            fits, token_count, available = (
                self._token_counter.check_fits_in_context(
                    messages,
                    reserved_output_tokens=config.max_tokens,
                )
            )
            if not fits:
                logger.warning(
                    f"Context limit approaching, truncating history"
                )
                messages = self._trim_messages(messages, config.max_tokens)

            # Generate next step
            try:
                response = await self.model.generate(messages, config)
                total_tokens += response.total_tokens
            except Exception as e:
                logger.error(f"Meta-CoT generation failed: {e}")
                break

            content = response.content
            full_reasoning_trace += content + "\n"

            # Parse actions from content
            new_actions = self._parse_actions(content)

            # Check for final answer
            answer_action = next(
                (a for a in new_actions if a.action_type == "answer"),
                None
            )
            if answer_action:
                actions.extend(new_actions)
                logger.info(
                    f"Meta-CoT complete | iterations={iteration+1} | "
                    f"searches={search_count} | readings={reading_count}"
                )
                return MetaCoTResult(
                    answer=answer_action.content,
                    reasoning_trace=full_reasoning_trace,
                    actions=actions,
                    total_searches=search_count,
                    total_readings=reading_count,
                    sources_used=sources_used,
                    total_tokens=total_tokens,
                    total_time_ms=(time.time() - start_time) * 1000,
                )

            # Process search and reading actions
            search_results_text = ""

            for action in new_actions:
                actions.append(action)
                action.timestamp_ms = (time.time() - start_time) * 1000

                if action.action_type == "search":
                    if search_count >= self.max_searches:
                        action.result = "[Search limit reached]"
                        search_results_text += action.result + "\n"
                        continue

                    result_text = await self._execute_search(action.content)
                    action.result = result_text
                    search_results_text += result_text + "\n"
                    search_count += 1

                    # Extract URLs from results for potential reading
                    urls = self._extract_urls_from_results(result_text)
                    sources_used.extend(urls[:2])

                elif action.action_type == "reading":
                    if reading_count >= self.max_readings:
                        action.result = "[Reading limit reached]"
                        search_results_text += action.result + "\n"
                        continue

                    result_text = await self._execute_reading(action.content)
                    action.result = result_text
                    search_results_text += result_text + "\n"
                    reading_count += 1
                    sources_used.append(action.content)

            if not search_results_text and not any(
                a.action_type in ("search", "reading") for a in new_actions
            ):
                # Model didn't search or answer - nudge it
                search_results_text = (
                    "[No action taken. Please either search for "
                    "information or provide your final answer using "
                    "<answer>...</answer>]"
                )

            # Add model's output + search results to conversation
            messages.append({
                "role": "assistant",
                "content": content,
            })
            if search_results_text:
                messages.append({
                    "role": "user",
                    "content": (
                        search_results_text.strip() +
                        "\n\nContinue your reasoning:"
                    ),
                })

        # Max iterations reached without final answer
        # Force a synthesis
        logger.warning(
            "Meta-CoT reached max iterations without final answer. "
            "Forcing synthesis."
        )
        forced_answer = await self._force_synthesis(
            question, messages, config
        )
        total_tokens += forced_answer.total_tokens

        return MetaCoTResult(
            answer=forced_answer.content,
            reasoning_trace=full_reasoning_trace,
            actions=actions,
            total_searches=search_count,
            total_readings=reading_count,
            sources_used=list(set(sources_used)),
            total_tokens=total_tokens,
            total_time_ms=(time.time() - start_time) * 1000,
        )

    async def stream_reason(
        self,
        question: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream Meta-CoT reasoning events in real time."""
        config = config or GenerationConfig(
            max_tokens=2048, temperature=0.7
        )

        system_prompt = META_COT_SYSTEM.format(
            max_searches=self.max_searches
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        search_count = 0
        reading_count = 0

        yield {
            "type": "start",
            "content": "Starting Meta-CoT reasoning...",
        }

        for iteration in range(self.max_iterations):
            try:
                response = await self.model.generate(messages, config)
            except Exception as e:
                yield {"type": "error", "content": str(e)}
                break

            content = response.content
            new_actions = self._parse_actions(content)

            # Stream the thinking
            think_actions = [
                a for a in new_actions if a.action_type == "think"
            ]
            for ta in think_actions:
                yield {"type": "thinking", "content": ta.content}

            # Check for final answer
            answer_action = next(
                (a for a in new_actions if a.action_type == "answer"),
                None
            )
            if answer_action:
                yield {
                    "type": "answer",
                    "content": answer_action.content,
                    "metadata": {
                        "searches": search_count,
                        "readings": reading_count,
                        "iterations": iteration + 1,
                    }
                }
                return

            # Execute searches
            search_results = ""
            for action in new_actions:
                if action.action_type == "search" and search_count < self.max_searches:
                    yield {
                        "type": "searching",
                        "content": f"Searching: {action.content}",
                    }
                    result = await self._execute_search(action.content)
                    action.result = result
                    search_results += result + "\n"
                    search_count += 1
                    yield {
                        "type": "search_result",
                        "content": result[:500],
                    }

                elif action.action_type == "reading" and reading_count < self.max_readings:
                    yield {
                        "type": "reading",
                        "content": f"Reading: {action.content}",
                    }
                    result = await self._execute_reading(action.content)
                    action.result = result
                    search_results += result + "\n"
                    reading_count += 1

            messages.append({"role": "assistant", "content": content})
            if search_results:
                messages.append({
                    "role": "user",
                    "content": search_results.strip() + "\n\nContinue:"
                })

        yield {
            "type": "timeout",
            "content": "Max iterations reached",
        }

    # ── Action executors ──────────────────────────────────────────────────────

    async def _execute_search(self, query: str) -> str:
        """Execute a web search and format results."""
        logger.debug(f"Meta-CoT search: {query}")
        try:
            response = await search_engine.search(query)
            if not response.has_results:
                return SEARCH_RESULT_TEMPLATE.format(
                    query=query,
                    results="No results found.",
                )

            results_text = ""
            for i, result in enumerate(
                response.results[:self.search_results_per_query], 1
            ):
                results_text += (
                    f"{i}. **{result.title}**\n"
                    f"   URL: {result.url}\n"
                    f"   {result.snippet}\n\n"
                )

            return SEARCH_RESULT_TEMPLATE.format(
                query=query,
                results=results_text.strip(),
            )
        except Exception as e:
            logger.warning(f"Search execution error: {e}")
            return SEARCH_RESULT_TEMPLATE.format(
                query=query,
                results=f"Search failed: {e}",
            )

    async def _execute_reading(self, url: str) -> str:
        """Fetch and extract content from a URL."""
        logger.debug(f"Meta-CoT reading: {url}")
        try:
            content = await extractor.extract(
                url,
                max_tokens=self.max_content_tokens,
            )
            if not content.success:
                return READING_RESULT_TEMPLATE.format(
                    url=url,
                    content=f"Could not read page: {content.error}",
                )
            return READING_RESULT_TEMPLATE.format(
                url=url,
                content=content.text,
            )
        except Exception as e:
            logger.warning(f"Reading execution error: {e}")
            return READING_RESULT_TEMPLATE.format(
                url=url,
                content=f"Reading failed: {e}",
            )

    # ── Parsing helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_actions(text: str) -> list[MetaCoTAction]:
        """Parse action tags from model output."""
        actions = []
        tag_pattern = re.compile(
            r'<(search|reading|think|answer)>(.*?)</\1>',
            re.DOTALL | re.IGNORECASE,
        )
        for match in tag_pattern.finditer(text):
            action_type = match.group(1).lower()
            content = match.group(2).strip()
            actions.append(MetaCoTAction(
                action_type=action_type,
                content=content,
            ))
        return actions

    @staticmethod
    def _extract_urls_from_results(results_text: str) -> list[str]:
        """Extract URLs from formatted search results."""
        url_pattern = re.compile(r'URL:\s*(https?://\S+)')
        return url_pattern.findall(results_text)

    def _trim_messages(
        self,
        messages: list[dict],
        reserved_output: int,
    ) -> list[dict]:
        """Trim message history to fit in context."""
        return self._token_counter.truncate_messages_to_fit(
            messages,
            max_tokens=reserved_output,
            preserve_system=True,
            preserve_last_n=4,
        )

    async def _force_synthesis(
        self,
        question: str,
        messages: list[dict],
        config: GenerationConfig,
    ):
        """Force a final synthesis when max iterations reached."""
        synthesis_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Based on all the research above, provide your "
                    "final comprehensive answer now. "
                    "Use <answer>...</answer> tags."
                )
            }
        ]
        synthesis_config = GenerationConfig(
            max_tokens=min(4096, config.max_tokens * 2),
            temperature=0.5,
        )
        return await self.model.generate(
            synthesis_messages, synthesis_config
        )