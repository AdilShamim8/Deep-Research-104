"""
Model providers for Deep Research 104.

Public surface:
    - BaseModel, GenerationConfig, ModelResponse, ModelCapability
    - OpenAIModel, DeepSeekModel, OllamaModel, VLLMModel, LlamaCppModel
    - get_model_factory(provider, model_name) -> BaseModel

All providers implement the BaseModel interface, so downstream code
can `from src.models import get_model_factory` and stay provider-agnostic.

Date tag: 17 October 2026.
"""

from src.models.base_model import (
    BaseModel,
    GenerationConfig,
    ModelCapability,
    ModelResponse,
    estimate_cost,
)
from src.models.local_models import (
    LlamaCppModel,
    OllamaModel,
    VLLMModel,
    get_model_factory,
)
from src.models.openai_models import OpenAIModel
from src.models.deepseek_models import DeepSeekModel

__all__ = [
    # Core types
    "BaseModel",
    "GenerationConfig",
    "ModelResponse",
    "ModelCapability",
    "estimate_cost",
    # Providers
    "OpenAIModel",
    "DeepSeekModel",
    "OllamaModel",
    "VLLMModel",
    "LlamaCppModel",
    # Factory
    "get_model_factory",
]

# Provider registry for introspection / docs
PROVIDERS: tuple[str, ...] = ("openai", "deepseek", "ollama", "vllm", "llamacpp")
