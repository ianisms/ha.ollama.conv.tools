"""Ollama API client.

Copyright (c) 2025 Ian N. Bennett (ianisms)
Licensed under MIT License
See LICENSE file in root of repository
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import aiohttp

_LOGGER = logging.getLogger(__name__)

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

    @staticmethod
    def load_default_prompts() -> dict:
        """Load system prompts from JSON file."""
        prompts_path = Path(__file__).parent / "prompts.json"
        try:
            with prompts_path.open() as f:
                return json.load(f)
        except Exception as e:
            _LOGGER.error("Failed to load prompts: %s", e)
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

    def __init__(
        self, 
        host: str, 
        port: int, 
        model: str = "mistral",
        tools: list = None
    ) -> None:
        """Initialize the client.

        Args:
            host: Hostname where Ollama is running
            port: Port number for Ollama API
            model: Name of the default model to use (e.g. mistral, llama2)
            tools: List of tools available to the assistant
        """
        self.host = host
        self.port = port
        self.model = model
        self.tools = tools or []
        self.system_prompt = None
        self._base_url = f"http://{host}:{port}/api"
        self._prompts = self.load_default_prompts()

    async def test_connection(self) -> bool:
        """Test the connection to Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self._base_url}/version") as response:
                    if response.status != 200:
                        raise ConnectionError("Failed to connect to Ollama")
                    return True
        except Exception as e:
            _LOGGER.error("Failed to connect to Ollama: %s", e)
            raise

    @staticmethod
    async def get_available_models(host: str, port: int) -> list[str]:
        """Get list of available models from Ollama server."""
        try:
            base_url = f"http://{host}:{port}/api"
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/tags") as response:
                    if response.status != 200:
                        raise ConnectionError("Failed to connect to Ollama")
                    data = await response.json()
                    if not isinstance(data, dict) or "models" not in data:
                        raise ValueError("Invalid response from Ollama server")
                    return [model["name"] for model in data["models"]]
        except Exception as e:
            _LOGGER.error("Failed to get Ollama models: %s", e)
            raise

    def _get_base_prompt(self) -> str:
        """Get the base system prompt."""
        return self._prompts.get("base_prompt", "You are a helpful home assistant")

    def _get_system_prompt(self) -> str:
        """Get the system prompt with tool definitions."""
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

    def _format_tool_description(self, tool: Tool) -> str:
        """Format a single tool's description with parameters."""
        params = json.dumps(tool.parameters, indent=2)
        return f"{tool.name}: {tool.description}\nParameters: {params}"

    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: str | None = None,
        model: str | None = None
    ) -> str:
        """Generate a response from Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._base_url}/generate",
                    json={
                        "model": model or self.model,
                        "prompt": prompt,
                        "system": system_prompt or self._get_system_prompt(),
                        "stream": False
                    }
                ) as response:
                    if response.status != 200:
                        _LOGGER.error("Ollama API error: %s", await response.text())
                        return "Sorry, I encountered an error processing your request."
                    
                    result = await response.json()
                    return result.get("response", "")
        except Exception as e:
            _LOGGER.error("Error in Ollama processing: %s", e)
            return "I apologize, but I encountered an error processing your request."

    def extract_tool_calls(self, response: str) -> list[tuple[str, dict[str, str]]]:
        """Extract tool calls from the response."""
        tool_calls = []
        for line in response.split("\n"):
            if "Using tool:" in line:
                try:
                    tool_part = line.split("Using tool:")[1].strip()
                    tool_name = tool_part[:tool_part.index("(")].strip()
                    params_str = tool_part[tool_part.index("(")+1:tool_part.rindex(")")]
                    
                    params = {}
                    for param in params_str.split(","):
                        if ":" in param:
                            key, value = param.split(":", 1)
                            params[key.strip()] = value.strip().strip('"\'')
                    
                    tool_calls.append((tool_name, params))
                except Exception as e:
                    _LOGGER.error("Error parsing tool call: %s", e)
                    continue
        
        return tool_calls

    async def _execute_tool(self, tool_name: str, args: dict[str, str]) -> str:
        """Execute a single tool and format its result."""
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            error_format = self._prompts.get("formatting", {}).get(
                "error_format", 
                "I encountered an error while trying to help: {error}"
            )
            return error_format.format(error=f"{tool_name} not found")
            
        try:
            result = await tool.execute(**args)
            success_format = self._prompts.get("formatting", {}).get(
                "success_acknowledgment", 
                "{tool_name} result: {result}"
            )
            return success_format.format(tool_name=tool_name, result=result)
        except Exception as e:
            _LOGGER.error("Tool execution error: %s", e)
            error_format = self._prompts.get("formatting", {}).get(
                "error_format", 
                "I encountered an error while trying to help: {error}"
            )
            return error_format.format(error=str(e))

    def _create_followup_prompt(self, original_text: str, tool_results: list[str]) -> str:
        """Create a follow-up prompt with tool results."""
        tool_config = self._prompts.get("tool_configuration", {})
        formatting = self._prompts.get("formatting", {})
        prefix = formatting.get("output_prefix", "")
        suffix = formatting.get("output_suffix", "")
        
        prompt = f"""Original request: {original_text}

Tool results:
{chr(10).join(tool_results)}

{tool_config.get("tool_response", "Please provide a natural response incorporating these results.")}"""

        if prefix or suffix:
            prompt = f"{prefix}\n{prompt}\n{suffix}".strip()
        
        return prompt

    async def process_with_tools(self, text: str) -> str:
        """Process text with tool support."""
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

        follow_up_prompt = self._create_followup_prompt(text, tool_results)
        follow_up_response = await self.generate_response(follow_up_prompt)
        
        return follow_up_response or response
