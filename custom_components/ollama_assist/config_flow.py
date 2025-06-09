"""Config flow for Ollama Assist integration.

Copyright (c) 2025 Ian N. Bennett (ianisms)
Licensed under MIT License
See LICENSE file in root of repository
"""
from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.exceptions import HomeAssistantError

from .const import (
    DOMAIN,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    CONF_SYSTEM_PROMPT,
)
from .ollama_client import OllamaClient

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_HOST, default=DEFAULT_HOST): str,
        vol.Required(CONF_PORT, default=DEFAULT_PORT): int,
    }
)

async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the user input allows us to connect."""
    try:
        available_models = await OllamaClient.get_available_models(
            data[CONF_HOST],
            data[CONF_PORT]
        )
        if not available_models:
            raise InvalidAuth("No models available on server")

        # Add the models to the return data for use in next step
        return {
            "title": f"Ollama ({data[CONF_HOST]})",
            "models": available_models
        }
    except ConnectionError as err:
        raise CannotConnect from err
    except Exception as err:
        raise InvalidAuth from err

class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Ollama Assist."""

    VERSION = 1
    
    def __init__(self) -> None:
        """Initialize the config flow."""
        self.connection_info: dict[str, Any] = {}
        self.available_models: list[str] = []
        self.model_info: dict[str, Any] = {}
        self._prompts = OllamaClient.load_default_prompts()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial connection step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                info = await validate_input(self.hass, user_input)
                self.connection_info = user_input
                self.available_models = info["models"]
                return await self.async_step_model()
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle model selection step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            data = {**self.connection_info, **user_input}
            self.model_info = user_input
            return await self.async_step_prompt()

        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema({
                vol.Required("model", default=DEFAULT_MODEL): vol.In(self.available_models),
            }),
            errors=errors,
        )

    async def async_step_prompt(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle system prompt configuration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            data = {
                **self.connection_info,
                **self.model_info,
                **user_input
            }
            return self.async_create_entry(
                title=f"Ollama ({data[CONF_HOST]})",
                data=data
            )

        base_prompt = self._prompts.get("base_prompt", "")
        tool_prompt = self._prompts.get("tool_prompt", {})
        default_prompt = f"""{tool_prompt.get("intro", "").format(base_prompt=base_prompt)}

{tool_prompt.get("tool_list_header", "Available tools:")}

{tool_prompt.get("tool_usage", "To use a tool, respond with: 'Using tool: <tool_name>(<parameters>)'")}
{tool_prompt.get("tool_response", "After using tools, provide a natural response incorporating the results.")}"""

        return self.async_show_form(
            step_id="prompt",
            data_schema=vol.Schema({
                vol.Required(CONF_SYSTEM_PROMPT, default=default_prompt): str,
            }),
            errors=errors,
        )

class CannotConnect(HomeAssistantError):
    """Error to indicate we cannot connect."""

class InvalidAuth(HomeAssistantError):
    """Error to indicate there is invalid auth."""
