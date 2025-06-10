"""Ollama API client.

Copyright (c) 2025 Ian N. Bennett (ianisms)
Licensed under MIT License
See LICENSE file in root of repository
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

import aiohttp
from aiohttp import ClientError

_LOGGER = logging.getLogger(__name__)

class ConnectionError(Exception):
    """Error indicating connection issues."""

class OllamaError(Exception):
    """Error indicating Ollama service issues."""

class Tool(Protocol):
    """Protocol defining the expected interface for tools."""

    name: str
    description: str
    parameters: dict[str, Any]

    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        ...

class OllamaClient:
    """Client to interact with Ollama API."""

    def __init__(
        self, 
        host: str, 
        port: int, 
        model: str = "mistral",
        tools: list = None,
        language: str = "en"
    ) -> None:
        """Initialize the client.

        Args:
            host: Hostname where Ollama is running
            port: Port number for Ollama API
            model: Name of the default model to use (e.g. mistral, llama2)
            tools: List of tools available to the assistant
            language: Language code for prompts and responses
        """
        self.host = host
        self.port = port
        self.model = model
        self.tools = tools or []
        self._language = language
        self.system_prompt = None
        self._base_url = f"http://{host}:{port}/api"
        self._prompts = None
        self._available = False

    def load_prompts(self) -> dict:
        """Load system prompts for current language."""
        prompts_path = Path(__file__).parent / "prompts" / f"{self._language}.json"
        default_path = Path(__file__).parent / "prompts/en.json"

        try:
            if prompts_path.exists():
                with prompts_path.open() as f:
                    return json.load(f)
            
            # Fall back to English if language-specific prompts don't exist
            if default_path.exists():
                with default_path.open() as f:
                    return json.load(f)

        except Exception as e:
            _LOGGER.error("Failed to load prompts for %s: %s", self._language, e)

        return {
            "default_prompts": {
                "no_tools": "You are a helpful home assistant designed to help users with their smart home needs. Provide clear, concise responses that are accurate and relevant to the user's requests.",
                "with_tools": "You are a helpful home assistant with access to tools that can control and monitor smart home devices. Use these tools when appropriate to help users accomplish their tasks."
            },
            "tool_configuration": {
                "intro": "You have access to tools for interacting with the smart home:",
                "tool_list_header": "Available Tools:",
                "list_format": "{name}: {description}",
                "usage_instructions": "To use a tool, respond with: 'Using tool: <tool_name>(<parameters>)'",
                "tool_response": "After using tools, provide a natural response incorporating the results.",
                "parameters_format": "Parameters: {params}"
            },
            "formatting": {
                "error_format": "I encountered an error while trying to help: {error}",
                "success_acknowledgment": "I've completed that action successfully."
            }
        }

    @property
    def language(self) -> str:
        """Get current language."""
        return self._language

    @language.setter
    def language(self, value: str) -> None:
        """Set current language and reload prompts."""
        if value != self._language:
            self._language = value
            self._prompts = self.load_prompts()

    async def test_connection(self) -> bool:
        """Test the connection to Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self._base_url}/version") as response:
                    if response.status != 200:
                        self._available = False
                        raise ConnectionError("Failed to connect to Ollama")
                    self._available = True
                    if not self._prompts:
                        self._prompts = self.load_prompts()
                    return True
        except ClientError as err:
            self._available = False
            raise ConnectionError(f"Failed to connect to Ollama: {err}") from err

    async def get_available_models(self) -> list[str]:
        """Get list of available models from Ollama server."""
        if not self._available:
            await self.test_connection()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self._base_url}/tags") as response:
                    if response.status != 200:
                        raise ConnectionError("Failed to connect to Ollama")
                    data = await response.json()
                    if not isinstance(data, dict) or "models" not in data:
                        raise OllamaError("Invalid response from Ollama server")
                    return [model["name"] for model in data["models"]]
        except ClientError as err:
            raise ConnectionError(f"Failed to get Ollama models: {err}") from err

    def _format_tool_description(self, tool: Tool) -> str:
        """Format a single tool's description with parameters."""
        params = json.dumps(tool.parameters, indent=2)
        return f"{tool.name}: {tool.description}\nParameters: {params}"

    def _create_system_prompt(self) -> str:
        """Create the complete system prompt."""
        if not self._prompts:
            self._prompts = self.load_prompts()

        if self.system_prompt is not None:
            if self.tools:
                tool_descriptions = [
                    self._format_tool_description(tool)
                    for tool in self.tools
                ]
                return f"{self.system_prompt}\n\nAvailable tools:\n{chr(10).join(tool_descriptions)}"
            return self.system_prompt

        default_prompts = self._prompts.get("default_prompts", {})
        tool_config = self._prompts.get("tool_configuration", {})
        
        if not self.tools:
            return default_prompts.get("no_tools", "You are a helpful home assistant.")

        tool_descriptions = []
        list_format = tool_config.get("list_format", "{name}: {description}")
        params_format = tool_config.get("parameters_format", "Parameters: {params}")
        
        for tool in self.tools:
            desc = list_format.format(
                name=tool.name,
                description=tool.description
            )
            params = json.dumps(tool.parameters, indent=2)
            tool_descriptions.append(f"{desc}\n{params_format.format(params=params)}")
        
        return f"""{default_prompts.get("with_tools")}

{tool_config.get("intro")}

{tool_config.get("tool_list_header")}
{chr(10).join(tool_descriptions)}

{tool_config.get("usage_instructions")}
{tool_config.get("tool_response")}"""

    async def generate_response(
        self, 
        prompt: str, 
        model: str | None = None
    ) -> str:
        """Generate a response from Ollama."""
        if not self._available:
            await self.test_connection()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/generate",
                    json={
                        "model": model or self.model,
                        "prompt": prompt,
                        "system": self._create_system_prompt(),
                        "stream": False
                    }
                ) as response:
                    if response.status != 200:
                        raise OllamaError(f"API error: {await response.text()}")
                    
                    result = await response.json()
                    return result.get("response", "")
        except ClientError as err:
            raise ConnectionError(f"Failed to communicate with Ollama: {err}") from err
        except Exception as err:
            raise OllamaError(f"Error generating response: {err}") from err

    def extract_tool_calls(self, response: str) -> list[tuple[str, dict[str, str]]]:
        """Extract tool calls from the response."""
        if not self._prompts:
            self._prompts = self.load_prompts()

        tool_config = self._prompts.get("tool_configuration", {})
        usage_pattern = tool_config.get("usage_instructions", "Using tool:")

        tool_calls = []
        for line in response.split("\n"):
            if usage_pattern.split(":")[0] in line:
                try:
                    tool_part = line.split(":", 1)[1].strip()
                    tool_name = tool_part[:tool_part.index("(")].strip()
                    params_str = tool_part[tool_part.index("(")+1:tool_part.rindex(")")]
                    
                    params = {}
                    for param in params_str.split(","):
                        if ":" in param:
                            key, value = param.split(":", 1)
                            params[key.strip()] = value.strip().strip('"\'')
                    
                    tool_calls.append((tool_name, params))
                except Exception as err:
                    _LOGGER.error("Error parsing tool call: %s", err)
                    continue
        
        return tool_calls

    async def _execute_tool(self, tool_name: str, args: dict[str, str]) -> str:
        """Execute a single tool and format its result."""
        if not self._prompts:
            self._prompts = self.load_prompts()

        formatting = self._prompts.get("formatting", {})
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            error_format = formatting.get(
                "error_format", 
                "I encountered an error while trying to help: {error}"
            )
            return error_format.format(error=f"{tool_name} not found")
            
        try:
            result = await tool.execute(**args)
            success_format = formatting.get(
                "success_acknowledgment", 
                "{tool_name} result: {result}"
            )
            return success_format.format(tool_name=tool_name, result=result)
        except Exception as err:
            _LOGGER.error("Tool execution error: %s", err)
            error_format = formatting.get(
                "error_format", 
                "I encountered an error while trying to help: {error}"
            )
            return error_format.format(error=str(err))

    async def process_with_tools(self, text: str) -> str:
        """Process text with tool support."""
        if not self._available:
            await self.test_connection()

        response = await self.generate_response(text)
        
        tool_calls = self.extract_tool_calls(response)
        if not tool_calls:
            return response

        tool_results = []
        for tool_name, args in tool_calls:
            result = await self._execute_tool(tool_name, args)
            tool_results.append(result)

        if not tool_results:
            return response

        # Create follow-up prompt with tool results
        tool_config = self._prompts.get("tool_configuration", {})
        prompt = f"""Original request: {text}

Tool results:
{chr(10).join(tool_results)}

{tool_config.get("tool_response", "Please provide a natural response incorporating these results.")}"""

        follow_up_response = await self.generate_response(prompt)
        return follow_up_response or response
