"""The Ollama Tooled Conversation Agent integration."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT, Platform
from homeassistant.core import HomeAssistant
import homeassistant.helpers.config_validation as cv
from homeassistant.exceptions import ConfigEntryNotReady

from .const import (
    DOMAIN,
    CONF_SYSTEM_PROMPT,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_MODEL,
    HEALTH_CHECK_TIMEOUT,
)
from .agent import OllamaAgent

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.Schema(
            {
                vol.Optional(CONF_HOST, default=DEFAULT_HOST): cv.string,
                vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
                vol.Optional(CONF_SYSTEM_PROMPT): cv.string,
            }
        )
    },
    extra=vol.ALLOW_EXTRA,
)

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Ollama Tooled CA from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    # Create agent instance
    agent = OllamaAgent(
        hass=hass,
        entry=entry,
    )

    # Test connection and initialize agent
    try:
        # Test connection
        url = f"http://{entry.data[CONF_HOST]}:{entry.data[CONF_PORT]}/api/health"
        async with agent._session.get(url, timeout=HEALTH_CHECK_TIMEOUT) as response:
            if response.status != 200:
                raise ConfigEntryNotReady(f"Ollama health check failed: {response.status}")

        # Initialize agent
        await agent.async_load()
    except Exception as err:
        raise ConfigEntryNotReady(f"Failed to initialize agent: {err}") from err

    # Store agent instance
    hass.data[DOMAIN][entry.entry_id] = agent

    # Set up platforms
    await hass.config_entries.async_forward_entry_setups(entry, ["diagnostics"])

    async def _async_unload(event: Any) -> None:
        """Handle unload."""
        await agent.async_save()

    # Register stop callback
    entry.async_on_unload(
        hass.bus.async_listen_once("homeassistant_stop", _async_unload)
    )

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Unload diagnostics platform
    unload_ok = await hass.config_entries.async_unload_platforms(entry, ["diagnostics"])
    
    if unload_ok:
        # Remove the agent
        agent = hass.data[DOMAIN].pop(entry.entry_id)
        await agent.async_save()

        # Remove the conversation agent
        conversation.async_unset_agent(hass, entry)

    return unload_ok

async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    _LOGGER.debug("Migrating from version %s", config_entry.version)

    if config_entry.version == 1:
        new_data = {**config_entry.data}
        # Add any new fields with defaults
        new_data.setdefault("model", DEFAULT_MODEL)

        config_entry.version = 2
        hass.config_entries.async_update_entry(config_entry, data=new_data)

    return True
