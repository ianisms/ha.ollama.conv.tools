"""Tools for Ollama Assist integration.

Copyright (c) 2025 Ian N. Bennett (ianisms)
Licensed under MIT License
See LICENSE file in root of repository
"""
from abc import ABC, abstractmethod
import json
import logging
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.components.weather import (
    DOMAIN as WEATHER_DOMAIN,
    Forecast,
)
from .const import TOOL_WEATHER, TOOL_STOCKS, TOOL_SEARCH

_LOGGER = logging.getLogger(__name__)

class BaseTool(ABC):
    """Abstract base class for tools."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the tool."""
        self.hass = hass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the tool description."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Return the tool parameters schema."""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool with given parameters."""
        pass

class WeatherTool(BaseTool):
    """Tool to get weather information."""

    @property
    def name(self) -> str:
        """Return the tool name."""
        return TOOL_WEATHER

    @property
    def description(self) -> str:
        """Return the tool description."""
        return "Get current weather information for a location"

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the tool parameters schema."""
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for"
                }
            },
            "required": ["location"]
        }

    async def execute(self, **kwargs: Any) -> str:
        """Get weather information for the specified location."""
        location = kwargs.get("location")
        if not location:
            return "Error: Location is required"

        try:
            # Use the first available weather entity
            weather_entities = self.hass.states.async_all(WEATHER_DOMAIN)
            if not weather_entities:
                return "No weather integration configured"

            state = weather_entities[0]
            temp = state.attributes.get("temperature")
            humidity = state.attributes.get("humidity")
            condition = state.attributes.get("condition")

            return (
                f"Current weather in {location}: {condition}, "
                f"Temperature: {temp}°C, Humidity: {humidity}%"
            )
        except Exception as e:
            _LOGGER.error("Error getting weather: %s", e)
            return f"Error getting weather information: {str(e)}"

class StockTool(BaseTool):
    """Tool to get stock prices."""

    @property
    def name(self) -> str:
        """Return the tool name."""
        return TOOL_STOCKS

    @property
    def description(self) -> str:
        """Return the tool description."""
        return "Get current stock price for a symbol"

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the tool parameters schema."""
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The stock symbol to look up"
                }
            },
            "required": ["symbol"]
        }

    async def execute(self, **kwargs: Any) -> str:
        """Get stock price for the specified symbol."""
        symbol = kwargs.get("symbol", "").upper()
        if not symbol:
            return "Error: Stock symbol is required"

        try:
            async with aiohttp.ClientSession() as session:
                # Using Alpha Vantage API (you would need to add API key handling)
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=demo"
                async with session.get(url) as response:
                    data = await response.json()
                    if "Global Quote" in data:
                        quote = data["Global Quote"]
                        price = quote.get("05. price", "N/A")
                        change = quote.get("09. change", "N/A")
                        return f"{symbol}: ${price} (Change: {change})"
                    return f"Could not find stock information for {symbol}"
        except Exception as e:
            _LOGGER.error("Error getting stock price: %s", e)
            return f"Error getting stock price: {str(e)}"

class WebSearchTool(BaseTool):
    """Tool to perform web searches."""

    @property
    def name(self) -> str:
        """Return the tool name."""
        return TOOL_SEARCH

    @property
    def description(self) -> str:
        """Return the tool description."""
        return "Search the web for information"

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the tool parameters schema."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }

    async def execute(self, **kwargs: Any) -> str:
        """Perform a web search for the specified query."""
        query = kwargs.get("query")
        if not query:
            return "Error: Search query is required"

        try:
            async with aiohttp.ClientSession() as session:
                # Using DuckDuckGo API (no key required)
                url = f"https://api.duckduckgo.com/?q={query}&format=json"
                async with session.get(url) as response:
                    data = await response.json()
                    if "AbstractText" in data and data["AbstractText"]:
                        return data["AbstractText"]
                    elif "RelatedTopics" in data and data["RelatedTopics"]:
                        # Return first related topic if no abstract is available
                        return data["RelatedTopics"][0]["Text"]
                    return "No relevant information found"
        except Exception as e:
            _LOGGER.error("Error performing web search: %s", e)
            return f"Error performing web search: {str(e)}"
