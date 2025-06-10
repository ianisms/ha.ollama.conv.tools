"""Config flow for Ollama Tooled CA."""
from __future__ import annotations

import logging
import voluptuous as vol
from typing import Any

from homeassistant import config_entries
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.selector import TextSelector

from .const import (
    DOMAIN,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_MODEL,
    CONF_SYSTEM_PROMPT,
)
from .ollama_client import OllamaClient

_LOGGER = logging.getLogger(__name__)

async def validate_input(hass: HomeAssistant, data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the user input allows us to connect."""
    client = OllamaClient(
        host=data[CONF_HOST],
        port=data[CONF_PORT],
        model=data.get("model", DEFAULT_MODEL),
        tools=[],
        language="en"
    )

    # Test connection and get available models
    await client.test_connection()
    models = await client.get_available_models()

    if not models:
        raise ValueError("No models available")

    return {"title": f"Ollama ({data[CONF_HOST]})", "models": models}

class OllamaConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Ollama Tooled CA."""

    VERSION = 2

    def __init__(self) -> None:
        """Initialize flow."""
        self.host: str | None = None
        self.port: int | None = None
        self.models: list[str] | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                info = await validate_input(self.hass, user_input)
                self.host = user_input[CONF_HOST]
                self.port = user_input[CONF_PORT]
                self.models = info["models"]

                return await self.async_step_model()

            except Exception as err:
                _LOGGER.error("Failed to connect: %s", err)
                errors["base"] = "cannot_connect"

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_HOST, default=DEFAULT_HOST): str,
                    vol.Required(CONF_PORT, default=DEFAULT_PORT): int,
                }
            ),
            errors=errors,
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle model selection."""
        errors: dict[str, str] = {}

        if user_input is not None:
            if user_input["model"] in self.models:
                return await self.async_step_prompt(user_input)
            errors["base"] = "invalid_model"

        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema(
                {
                    vol.Required("model", default=DEFAULT_MODEL): vol.In(self.models),
                }
            ),
            errors=errors,
        )

    async def async_step_prompt(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle system prompt configuration."""
        if user_input is not None:
            if CONF_SYSTEM_PROMPT not in user_input:
                user_input[CONF_SYSTEM_PROMPT] = None

            return self.async_create_entry(
                title=f"Ollama ({self.host})",
                data={
                    CONF_HOST: self.host,
                    CONF_PORT: self.port,
                    "model": user_input["model"],
                    CONF_SYSTEM_PROMPT: user_input.get(CONF_SYSTEM_PROMPT),
                },
            )

        return self.async_show_form(
            step_id="prompt",
            data_schema=vol.Schema(
                {
                    vol.Optional(CONF_SYSTEM_PROMPT): str,
                }
            ),
        )

    async def async_step_import(self, user_input: dict[str, Any]) -> FlowResult:
        """Handle import."""
        return await self.async_step_user(user_input)
