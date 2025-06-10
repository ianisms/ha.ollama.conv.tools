"""Tools for Ollama Assist.

Copyright (c) 2025 Ian N. Bennett (ianisms)
Licensed under MIT License
See LICENSE file in root of repository
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from homeassistant.components import weather
from homeassistant.const import ATTR_TEMPERATURE
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import network

class BaseTool(ABC):
    """Base class for tools."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the tool."""
        self.hass = hass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the description of the tool."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Return the parameters schema."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        pass

class WeatherTool(BaseTool):
    """Tool to get weather information."""

    @property
    def name(self) -> str:
        """Return the name of the tool."""
        return "get_weather"

    @property
    def description(self) -> str:
        """Return the description of the tool."""
        return "Get current weather information from a weather entity"

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the parameters schema."""
        return {
            "entity_id": {
                "type": "string",
                "description": "The entity ID of the weather entity (e.g. weather.home)"
            }
        }

    async def execute(self, **kwargs) -> str:
        """Execute the tool."""
        entity_id = kwargs.get("entity_id")
        if not entity_id:
            return "No weather entity specified"

        state: State = self.hass.states.get(entity_id)
        if not state:
            return f"Weather entity {entity_id} not found"

        if not state.state:
            return f"No weather data available for {entity_id}"

        current_temp = state.attributes.get(ATTR_TEMPERATURE)
        if current_temp is None:
            return "Temperature data not available"

        return (
            f"Current weather: {state.state}, "
            f"Temperature: {current_temp}°"
        )

class StockTool(BaseTool):
    """Tool to get stock information."""

    @property
    def name(self) -> str:
        """Return the name of the tool."""
        return "get_stock_price"

    @property
    def description(self) -> str:
        """Return the description of the tool."""
        return "Get current stock price from an existing stock sensor"

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the parameters schema."""
        return {
            "entity_id": {
                "type": "string",
                "description": "The entity ID of the stock sensor (e.g. sensor.stock_price)"
            }
        }

    async def execute(self, **kwargs) -> str:
        """Execute the tool."""
        entity_id = kwargs.get("entity_id")
        if not entity_id:
            return "No stock entity specified"

        state: State = self.hass.states.get(entity_id)
        if not state:
            return f"Stock entity {entity_id} not found"

        if not state.state:
            return f"No stock data available for {entity_id}"

        return f"Current price: ${state.state}"

class WebSearchTool(BaseTool):
    """Tool for web searches."""

    @property
    def name(self) -> str:
        """Return the name of the tool."""
        return "web_search"

    @property
    def description(self) -> str:
        """Return the description of the tool."""
        return "Search the web for information"

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the parameters schema."""
        return {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (default: 3)",
                "default": 3
            }
        }

    async def execute(self, **kwargs) -> str:
        """Execute the tool."""
        query = kwargs.get("query")
        if not query:
            return "No search query provided"

        # This is a placeholder - actual web search implementation would go here
        return f"Web search functionality coming soon. Query was: {query}"
