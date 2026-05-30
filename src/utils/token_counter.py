"""
Token counting utility - critical for respecting context windows.
Supports multiple model families with accurate token estimates.
"""

from typing import Union
import tiktoken
from loguru import logger


# Model -> encoding mapping
MODEL_ENCODING_MAP = {
    # OpenAI o-family
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "gpt-4o": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    # DeepSeek (uses similar tokenizer to LLaMA)
    "deepseek-reasoner": "cl100k_base",
    "deepseek-chat": "cl100k_base",
}

# Context window limits (in tokens)
MODEL_CONTEXT_LIMITS = {
    "o1": 200_000,
    "o1-mini": 128_000,
    "o1-preview": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "deepseek-reasoner": 128_000,
    "deepseek-chat": 64_000,
    "deepseek-r1:7b": 32_000,
    "deepseek-r1:70b": 128_000,
    "llama3:8b": 8_000,
    "llama3:70b": 8_000,
}

# Maximum OUTPUT tokens per model
MODEL_OUTPUT_LIMITS = {
    "o1": 32_768,
    "o1-mini": 65_536,
    "o3": 100_000,
    "o3-mini": 100_000,
    "gpt-4o": 16_384,
    "deepseek-reasoner": 32_768,
    "deepseek-chat": 8_192,
    "deepseek-r1:7b": 8_192,
    "deepseek-r1:70b": 16_384,
}


class TokenCounter:
    """
    Accurate token counting with safety margins.
    
    Always count tokens BEFORE sending to API to avoid
    context overflow errors in production.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._encoding = self._load_encoding(model)
    
    def _load_encoding(self, model: str) -> tiktoken.Encoding:
        """Load appropriate tiktoken encoding for model."""
        encoding_name = MODEL_ENCODING_MAP.get(model, "cl100k_base")
        try:
            return tiktoken.get_encoding(encoding_name)
        except Exception:
            logger.warning(
                f"Could not load encoding for {model}, "
                f"falling back to cl100k_base"
            )
            return tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if not text:
            return 0
        return len(self._encoding.encode(text))
    
    def count_messages_tokens(
        self,
        messages: list[dict]
    ) -> int:
        """
        Count tokens in a messages array (chat format).
        Includes overhead per message as per OpenAI's counting.
        """
        total = 0
        # Every message has 4 token overhead (role + content markers)
        for message in messages:
            total += 4
            for key, value in message.items():
                if isinstance(value, str):
                    total += self.count_tokens(value)
                elif isinstance(value, list):
                    # Handle multimodal content
                    for item in value:
                        if isinstance(item, dict) and item.get("type") == "text":
                            total += self.count_tokens(item.get("text", ""))
        # Every reply is primed with 3 tokens
        total += 3
        return total
    
    def get_context_limit(self, model: Optional[str] = None) -> int:
        """Get context window limit for model."""
        m = model or self.model
        return MODEL_CONTEXT_LIMITS.get(m, 8_000)
    
    def get_output_limit(self, model: Optional[str] = None) -> int:
        """Get max output tokens for model."""
        m = model or self.model
        return MODEL_OUTPUT_LIMITS.get(m, 4_096)
    
    def check_fits_in_context(
        self,
        messages: list[dict],
        reserved_output_tokens: int = 2048,
        model: Optional[str] = None,
    ) -> tuple[bool, int, int]:
        """
        Check if messages fit in context window.
        
        Returns:
            (fits, token_count, available_tokens)
        """
        m = model or self.model
        context_limit = self.get_context_limit(m)
        token_count = self.count_messages_tokens(messages)
        available = context_limit - token_count - reserved_output_tokens
        fits = available >= 0
        
        if not fits:
            logger.warning(
                f"Messages ({token_count} tokens) exceed context limit "
                f"({context_limit}) for model {m}. "
                f"Overflow: {-available} tokens."
            )
        
        return fits, token_count, max(0, available)
    
    def truncate_text_to_tokens(
        self,
        text: str,
        max_tokens: int,
        truncation_marker: str = "\n\n[Content truncated due to length]"
    ) -> str:
        """
        Truncate text to fit within token limit.
        Preserves beginning of text (most important context).
        """
        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        marker_tokens = self.count_tokens(truncation_marker)
        target_tokens = max_tokens - marker_tokens
        
        # Binary search for right truncation point
        encoded = self._encoding.encode(text)
        truncated = self._encoding.decode(encoded[:target_tokens])
        
        logger.debug(
            f"Truncated text from {current_tokens} to "
            f"{self.count_tokens(truncated + truncation_marker)} tokens"
        )
        return truncated + truncation_marker
    
    def truncate_messages_to_fit(
        self,
        messages: list[dict],
        max_tokens: int,
        preserve_system: bool = True,
        preserve_last_n: int = 2,
    ) -> list[dict]:
        """
        Intelligently truncate messages list to fit token budget.
        
        Strategy:
        1. Always keep system message
        2. Always keep last N messages
        3. Drop middle messages if needed
        4. Truncate content of remaining messages if still too long
        """
        fits, count, _ = self.check_fits_in_context(
            messages,
            reserved_output_tokens=max_tokens
        )
        if fits:
            return messages
        
        # Separate system messages
        system_msgs = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        
        # Always preserve last N non-system messages
        preserved_end = non_system[-preserve_last_n:] if preserve_last_n else []
        middle = non_system[:-preserve_last_n] if preserve_last_n else non_system
        
        # Try removing middle messages one by one (oldest first)
        result = system_msgs + middle + preserved_end
        while middle:
            fits, _, _ = self.check_fits_in_context(
                result, reserved_output_tokens=max_tokens
            )
            if fits:
                break
            middle = middle[1:]  # Remove oldest middle message
            result = system_msgs + middle + preserved_end
        
        return result


# Convenience function
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    return TokenCounter(model).count_tokens(text)