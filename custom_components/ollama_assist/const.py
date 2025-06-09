"""Constants for the Ollama Assist integration.

Copyright (c) 2025 Ian N. Bennett (ianisms)
Licensed under MIT License
See LICENSE file in root of repository
"""

DOMAIN = "ollama_assist"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 11434  # Default Ollama port
DEFAULT_MODEL = "mistral"  # Default Ollama model

# Configuration keys
CONF_SYSTEM_PROMPT = "system_prompt"

# Tool names
TOOL_WEATHER = "get_weather"
TOOL_STOCKS = "get_stock_price"
TOOL_SEARCH = "web_search"

# Default prompts
SYSTEM_PROMPT = """You are a helpful home assistant that can use tools to get information. 
Available tools:
- get_weather: Get current weather for a location
- get_stock_price: Get current stock price for a symbol
- web_search: Search the web for information

Respond naturally and use tools when needed to provide accurate information."""

TOOL_ERROR = "Sorry, I encountered an error while using that tool. {error}"
