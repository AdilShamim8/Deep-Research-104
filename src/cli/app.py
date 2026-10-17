"""
Command-line interface for Deep Research 104.

Wraps the FastAPI app's pipeline + reasoning engines into a Typer
CLI for shell-driven research, scripting, and CI usage. Mirrors the
REST API surface so the same code paths run in both modes.

Usage:
    deep-research ask "What are the latest fusion breakthroughs?"
    deep-research cot  "Explain quantum entanglement"
    deep-research tot  "Solve: 2x + 5 = 13"
    deep-research search "fusion energy breakthroughs 2026" --max-results 5
    deep-research serve --port 8000
    deep-research version

Date tag: 17 October 2026.
"""

from __future__ import annotations

import asyncio
import json
import sys

import typer
from config import __build_date__, __version__
from config.settings import settings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="deep-research",
    help="Production-grade autonomous research engine.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_model(provider: str, model_name: str | None = None):
    from src.models import get_model_factory

    return get_model_factory(provider, model_name)


def _print_panel(title: str, body: str, style: str = "cyan") -> None:
    console.print(Panel(body, title=title, border_style=style))


# ── Commands ─────────────────────────────────────────────────────────────────


@app.command()
def version() -> None:
    """Print version, build date, and Python version."""
    table = Table(title="Deep Research 104", show_header=False)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")
    table.add_row("name", settings.app_name)
    table.add_row("version", __version__)
    table.add_row("build_date", __build_date__)
    table.add_row("python", sys.version.split()[0])
    table.add_row("env", settings.app_env)
    table.add_row("log_level", settings.log_level)
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Research question to investigate."),
    provider: str = typer.Option("openai", help="LLM provider."),
    model_name: str | None = typer.Option(None, help="Model override."),
    max_iterations: int = typer.Option(3, min=1, max=10),
    fast_mode: bool = typer.Option(False, help="Single iteration, fewer queries."),
    mode: str = typer.Option("deep_research", help="Pipeline mode."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of Markdown."),
) -> None:
    """Run the full deep research pipeline on a question."""
    from src.pipeline.deep_research import DeepResearchPipeline

    async def _run():
        model = _get_model(provider, model_name)
        pipeline = DeepResearchPipeline(model=model, max_iterations=max_iterations)
        report = await pipeline.research(question, fast_mode=fast_mode)
        if json_output:
            console.print_json(report.to_json())
        else:
            _print_panel("Question", question, "cyan")
            console.print(report.to_markdown())

    asyncio.run(_run())


@app.command()
def cot(
    question: str = typer.Argument(..., help="Question to reason about."),
    provider: str = typer.Option("openai"),
    model_name: str | None = typer.Option(None),
    mode: str = typer.Option("structured", help="CoT mode."),
    n_samples: int = typer.Option(1, min=1, max=20, help="Self-consistency samples."),
    temperature: float = typer.Option(0.7, min=0.0, max=2.0),
) -> None:
    """Run Chain-of-Thought reasoning on a question."""
    from src.models.base_model import GenerationConfig
    from src.reasoning.chain_of_thought import ChainOfThought

    async def _run():
        model = _get_model(provider, model_name)
        engine = ChainOfThought(model)
        config = GenerationConfig(temperature=temperature, max_tokens=4_096)
        if n_samples > 1:
            result = await engine.self_consistency(
                question=question,
                n_samples=n_samples,
                mode=mode,
                config=config,
            )
        else:
            result = await engine.generate(question=question, mode=mode, config=config)
        _print_panel("Reasoning", result.reasoning or "(none)", "yellow")
        _print_panel("Answer", result.answer, "green")
        console.print(
            f"[bold]Confidence:[/bold] {result.confidence:.2f}  "
            f"Votes: {result.supporting_votes}/{result.total_votes}"
        )

    asyncio.run(_run())


@app.command()
def tot(
    problem: str = typer.Argument(..., help="Problem to solve via Tree of Thoughts."),
    provider: str = typer.Option("openai"),
    model_name: str | None = typer.Option(None),
    strategy: str = typer.Option("beam", help="beam | bfs | dfs"),
    branching_factor: int = typer.Option(3, min=2, max=6),
    max_depth: int = typer.Option(4, min=1, max=8),
    beam_width: int = typer.Option(2, min=1, max=5),
) -> None:
    """Solve a problem using Tree of Thoughts search."""
    from src.reasoning.tree_of_thoughts import SearchStrategy, TreeOfThoughts

    strategy_map = {
        "beam": SearchStrategy.BEAM,
        "bfs": SearchStrategy.BFS,
        "dfs": SearchStrategy.DFS,
    }
    if strategy not in strategy_map:
        raise typer.BadParameter(f"Unknown strategy: {strategy!r}")

    async def _run():
        model = _get_model(provider, model_name)
        engine = TreeOfThoughts(
            model=model,
            branching_factor=branching_factor,
            max_depth=max_depth,
            beam_width=beam_width,
            search_strategy=strategy_map[strategy],
        )
        result = await engine.solve(problem=problem)
        _print_panel("Answer", result.answer, "green")
        console.print(
            f"[bold]Score:[/bold] {result.best_score:.3f}  "
            f"Nodes explored: {result.total_nodes_explored}  "
            f"Path length: {len(result.best_path)}"
        )

    asyncio.run(_run())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    max_results: int = typer.Option(10, min=1, max=50),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run a web search across the configured providers."""
    from src.search.web_search import search_engine

    async def _run():
        response = await search_engine.search(query)
        if json_output:
            console.print_json(
                json.dumps(
                    {
                        "query": response.query,
                        "provider": response.provider,
                        "total_results": response.total_results,
                        "results": [r.to_dict() for r in response.results[:max_results]],
                    },
                    default=str,
                )
            )
            return
        table = Table(title=f"Search results — {query[:60]}")
        table.add_column("#", style="dim")
        table.add_column("Title")
        table.add_column("URL")
        table.add_column("Domain")
        for i, r in enumerate(response.results[:max_results], 1):
            table.add_row(str(i), r.title[:60], r.url, r.domain)
        console.print(table)

    asyncio.run(_run())


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    workers: int = typer.Option(4, min=1, max=32, help="uvicorn workers."),
    reload: bool = typer.Option(False, help="Enable auto-reload (dev)."),
) -> None:
    """Start the FastAPI server (production mode by default)."""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        workers=1 if reload else workers,
        reload=reload,
        log_level=settings.log_level.lower(),
        proxy_headers=True,
        no_server_header=True,
    )


@app.command()
def providers() -> None:
    """List available LLM providers."""
    from src.models import PROVIDERS

    table = Table(title="Available LLM providers")
    table.add_column("Provider", style="bold cyan")
    table.add_column("Default model")
    table.add_column("Env var required")
    for p in PROVIDERS:
        if p == "openai":
            table.add_row(p, settings.models.openai_reasoning_model, "OPENAI_API_KEY")
        elif p == "deepseek":
            table.add_row(p, settings.models.deepseek_reasoning_model, "DEEPSEEK_API_KEY")
        elif p == "ollama":
            table.add_row(p, settings.models.ollama_model, "(none — local)")
        elif p == "vllm":
            table.add_row(p, "deepseek-ai/DeepSeek-R1", "(none — local)")
        elif p == "llamacpp":
            table.add_row(p, "llama-cpp", "(not implemented)")
    console.print(table)


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point referenced by [project.scripts] in pyproject.toml."""
    app()


if __name__ == "__main__":
    main()
