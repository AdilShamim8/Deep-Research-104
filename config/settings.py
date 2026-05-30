"""
Central configuration management using Pydantic Settings.
Loads from environment variables with full validation.
"""

from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Configuration for individual model providers."""
    
    # OpenAI / o-family
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_org_id: Optional[str] = Field(default=None, env="OPENAI_ORG_ID")
    openai_default_model: str = Field(default="o3-mini", env="OPENAI_DEFAULT_MODEL")
    openai_reasoning_model: str = Field(default="o3", env="OPENAI_REASONING_MODEL")
    
    # DeepSeek
    deepseek_api_key: str = Field(default="", env="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        env="DEEPSEEK_BASE_URL"
    )
    deepseek_reasoning_model: str = Field(
        default="deepseek-reasoner",
        env="DEEPSEEK_REASONING_MODEL"
    )
    
    # Local
    local_model_path: str = Field(default="./models", env="LOCAL_MODEL_PATH")
    local_model_type: Literal["ollama", "vllm", "llamacpp"] = Field(
        default="ollama",
        env="LOCAL_MODEL_TYPE"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        env="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(
        default="deepseek-r1:7b",
        env="OLLAMA_MODEL"
    )


class SearchConfig(BaseSettings):
    """Search engine configuration."""
    
    serpapi_key: str = Field(default="", env="SERPAPI_KEY")
    brave_search_api_key: str = Field(default="", env="BRAVE_SEARCH_API_KEY")
    bing_search_api_key: str = Field(default="", env="BING_SEARCH_API_KEY")
    
    max_search_results: int = Field(default=10, env="MAX_SEARCH_RESULTS")
    search_timeout_seconds: int = Field(default=15, env="SEARCH_TIMEOUT_SECONDS")
    max_content_length: int = Field(default=8000, env="MAX_CONTENT_LENGTH")
    
    # Search engine priority
    search_provider: Literal["serpapi", "brave", "bing", "duckduckgo"] = Field(
        default="duckduckgo",
        env="SEARCH_PROVIDER"
    )


class ReasoningConfig(BaseSettings):
    """Reasoning pipeline configuration."""
    
    # Token budgets
    max_tokens_per_request: int = Field(
        default=128000,
        env="MAX_TOKENS_PER_REQUEST"
    )
    reasoning_budget: int = Field(
        default=32768,
        env="REASONING_BUDGET"
    )
    max_output_tokens: int = Field(
        default=8192,
        env="MAX_OUTPUT_TOKENS"
    )
    
    # Sampling
    parallel_samples: int = Field(default=3, env="PARALLEL_SAMPLES")
    sequential_steps: int = Field(default=5, env="SEQUENTIAL_STEPS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    top_p: float = Field(default=0.9, env="TOP_P")
    
    # ToT
    tot_branching_factor: int = Field(default=3, env="TOT_BRANCHING_FACTOR")
    tot_max_depth: int = Field(default=4, env="TOT_MAX_DEPTH")
    tot_beam_width: int = Field(default=2, env="TOT_BEAM_WIDTH")
    
    # Research
    max_research_iterations: int = Field(
        default=5,
        env="MAX_RESEARCH_ITERATIONS"
    )
    max_search_per_iteration: int = Field(
        default=3,
        env="MAX_SEARCH_PER_ITERATION"
    )


class AppSettings(BaseSettings):
    """Master application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )
    
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        env="APP_ENV"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        env="LOG_LEVEL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    # Nested configs - loaded separately for clean imports
    @property
    def models(self) -> ModelConfig:
        return ModelConfig()
    
    @property
    def search(self) -> SearchConfig:
        return SearchConfig()
    
    @property
    def reasoning(self) -> ReasoningConfig:
        return ReasoningConfig()
    
    @field_validator("app_env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        return v.lower()


# Singleton
settings = AppSettings()