"""
Training-time CLI for Deep Research 104.

Provides commands for the training-side toolkit:
    deep-research-train star     — STaR self-taught reasoner
    deep-research-train reward   — Train ORM / PRM reward models
    deep-research-train grpo     — GRPO RL with a verifier
    deep-research-train refine   — Self-refinement loop

These commands require the optional [training] extras (torch,
transformers, trl, peft, datasets). Install with:
    pip install -e .[training]

Date tag: 17 October 2026.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from config import __version__


app = typer.Typer(
    name="deep-research-train",
    help="Training-time toolkit for reasoning models.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def _require_training_extras() -> None:
    """Fail fast if the heavy ML stack isn't installed."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        console.print(
            Panel.fit(
                "[bold red]Missing training dependencies.[/bold red]\n\n"
                "Install with:\n"
                "  pip install -e .[training]\n\n"
                f"Original error: {e}",
                title="Cannot start training CLI",
                border_style="red",
            )
        )
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Print version + whether training extras are available."""
    try:
        import torch
        torch_version = torch.__version__
        cuda = torch.cuda.is_available()
    except ImportError:
        torch_version = "not installed"
        cuda = False
    console.print(f"deep-research-train v{__version__}")
    console.print(f"torch: {torch_version}")
    console.print(f"cuda available: {cuda}")


@app.command()
def star(
    model_name: str = typer.Option(..., help="Base model to fine-tune."),
    dataset_path: str = typer.Option(..., help="Path to Q&A JSON dataset."),
    output_dir: str = typer.Option("./models/star-output"),
    num_iterations: int = typer.Option(3, min=1, max=10),
) -> None:
    """Run STaR (Self-Taught Reasoner) fine-tuning loop."""
    _require_training_extras()
    from src.training.star_trainer import STaRTrainer

    async def _run():
        trainer = STaRTrainer(
            model_name=model_name,
            output_dir=output_dir,
            num_iterations=num_iterations,
        )
        await trainer.train(dataset_path=dataset_path)
        console.print(f"[bold green]Done. Output in {output_dir}[/bold green]")

    asyncio.run(_run())


@app.command()
def reward(
    mode: str = typer.Option("orm", help="orm | prm"),
    base_model: str = typer.Option(..., help="Base model name."),
    preference_data: str = typer.Option(..., help="Preference pairs JSONL."),
    output_dir: str = typer.Option("./models/reward-model"),
) -> None:
    """Train an Outcome (ORM) or Process (PRM) reward model."""
    _require_training_extras()
    from src.training.reward_models import RewardModelTrainer

    async def _run():
        trainer = RewardModelTrainer(
            base_model=base_model,
            output_dir=output_dir,
            mode=mode,
        )
        await trainer.train(preference_path=preference_data)
        console.print(f"[bold green]Done. Output in {output_dir}[/bold green]")

    asyncio.run(_run())


@app.command()
def grpo(
    base_model: str = typer.Option(..., help="Base policy model."),
    verifier_model: str = typer.Option(..., help="Reward/verifier model."),
    output_dir: str = typer.Option("./models/grpo-output"),
    steps: int = typer.Option(1000, min=10),
) -> None:
    """Run GRPO (Group-Relative Policy Optimization)."""
    _require_training_extras()
    from src.training.rl_trainer import GRPOTrainer

    async def _run():
        trainer = GRPOTrainer(
            base_model=base_model,
            verifier_model=verifier_model,
            output_dir=output_dir,
        )
        await trainer.train(num_steps=steps)
        console.print(f"[bold green]Done. Output in {output_dir}[/bold green]")

    asyncio.run(_run())


@app.command()
def refine(
    model_name: str = typer.Option(..., help="Model to refine."),
    task: str = typer.Option(..., help="Task prompt."),
    iterations: int = typer.Option(3, min=1, max=10),
) -> None:
    """Run self-refinement (generate -> critique -> revise)."""
    _require_training_extras()
    from src.models import get_model_factory
    from src.training.self_refinement import SelfRefinementEngine

    async def _run():
        model = get_model_factory("openai", model_name)
        engine = SelfRefinementEngine(model=model, max_iterations=iterations)
        result = await engine.refine(task)
        console.print(Panel(result.final_answer, title="Final answer", border_style="green"))
        console.print(f"[bold]Iterations:[/bold] {result.iterations}")

    asyncio.run(_run())


def main() -> None:
    """Entry point for [project.scripts] deep-research-train."""
    app()


if __name__ == "__main__":
    main()
