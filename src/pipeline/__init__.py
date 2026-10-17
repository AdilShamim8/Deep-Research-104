"""
Research pipeline orchestrators for Deep Research 104.

Public surface:
    - DeepResearchPipeline, ResearchReport
        Iterative search -> reason -> gap-analyze -> refine -> synthesize
    - MetaCoT, MetaCoTResult
        Internalized search via meta chain-of-thought (Single-loop reasoning)

Date tag: 17 October 2026.
"""

from src.pipeline.deep_research import (
    DeepResearchPipeline,
    ResearchIteration,
    ResearchReport,
)
from src.pipeline.meta_cot import MetaCoT, MetaCoTResult

__all__ = [
    "DeepResearchPipeline",
    "ResearchReport",
    "ResearchIteration",
    "MetaCoT",
    "MetaCoTResult",
]
