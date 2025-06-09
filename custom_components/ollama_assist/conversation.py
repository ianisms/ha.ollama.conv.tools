"""Conversation agent for Ollama Assist.

Copyright (c) 2025 Ian N. Bennett (ianisms)
Licensed under MIT License
See LICENSE file in root of repository
"""
from __future__ import annotations

from typing import Any

from homeassistant.components.conversation import (
    ConversationInput,
    ConversationResult,
)
from homeassistant.core import HomeAssistant

from .ollama_client import OllamaClient
from .tools import WeatherTool, StockTool, WebSearchTool

class OllamaConversation:
    """Class to handle Ollama conversation processing."""

    def __init__(
        self,
        hass: HomeAssistant,
        host: str,
        port: int,
        model: str,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the conversation agent."""
        self.hass = hass
        self.tools = [
            WeatherTool(hass),
            StockTool(hass),
            WebSearchTool(hass)
        ]
        self.client = OllamaClient(
            host=host,
            port=port,
            model=model,
            tools=self.tools
        )
        if system_prompt:
            self.client.system_prompt = system_prompt

    async def initialize(self) -> None:
        """Initialize the Ollama connection and tools."""
        await self.client.test_connection()

    async def async_process(
        self, input: ConversationInput
    ) -> ConversationResult:
        """Process a conversation input."""
        response = await self.client.process_with_tools(input.text)
        
        return ConversationResult(
            response=response,
            conversation_id=input.conversation_id,
        )
