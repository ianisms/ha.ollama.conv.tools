"""Constants for Ollama Tooled Conversation Agent."""
from typing import Final

DOMAIN: Final = "ollama_tooled_ca"

# Configuration
DEFAULT_HOST: Final = "localhost"
DEFAULT_PORT: Final = 11434
DEFAULT_MODEL: Final = "llama2"
CONF_SYSTEM_PROMPT: Final = "system_prompt"

# Performance settings
MAX_CONCURRENT_REQUESTS: Final = 5
REQUEST_TIMEOUT: Final = 30
HEALTH_CHECK_TIMEOUT: Final = 10

# Conversation history
MAX_CONVERSATION_HISTORY: Final = 100
HISTORY_PRUNING_THRESHOLD: Final = 80
