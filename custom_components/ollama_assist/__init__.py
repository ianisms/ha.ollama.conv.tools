"""The Ollama Assist integration.

Copyright (c) 2025 Ian N. Bennett (ianisms)
Licensed under MIT License
See LICENSE file in root of repository
"""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant.components.conversation import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_MODEL, CONF_PORT
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv

from .const import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DOMAIN,
    CONF_SYSTEM_PROMPT,
)
from .conversation import OllamaConversation

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.Schema({
            vol.Optional(CONF_HOST, default=DEFAULT_HOST): cv.string,
            vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
            vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): cv.string,
            vol.Optional(CONF_SYSTEM_PROMPT): cv.string,
        })
    },
    extra=vol.ALLOW_EXTRA,
)

async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up the Ollama Assist component."""
    if DOMAIN not in config:
        return True

    hass.data[DOMAIN] = {
        CONF_HOST: config[DOMAIN][CONF_HOST],
        CONF_PORT: config[DOMAIN][CONF_PORT],
        CONF_MODEL: config[DOMAIN][CONF_MODEL],
        CONF_SYSTEM_PROMPT: config[DOMAIN].get(CONF_SYSTEM_PROMPT),
    }

    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Ollama Assist from a config entry."""
    conversation = OllamaConversation(
        hass,
        entry.data[CONF_HOST],
        entry.data[CONF_PORT],
        entry.data.get(CONF_MODEL, DEFAULT_MODEL),
        entry.data.get(CONF_SYSTEM_PROMPT)
    )

    await conversation.initialize()

    @callback
    def async_conversation_match(input_text: str) -> bool:
        """Determine if the input text should be processed by this agent."""
        return True  # Process all conversations

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = conversation

    hass.components.conversation.async_register_agent(
        entry, conversation, async_conversation_match
    )

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    hass.components.conversation.async_unregister_agent(entry)
    del hass.data[DOMAIN][entry.entry_id]
    return True
