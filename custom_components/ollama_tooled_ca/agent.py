"""Conversation agent implementation for Ollama Tooled CA."""
from __future__ import annotations

import logging
import asyncio
from typing import Any
from collections import deque
from dataclasses import dataclass
from time import time

from homeassistant.components import assist_pipeline, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT
from homeassistant.const import TIMEOUT_BACKOFF
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import dt as dt_util
from homeassistant.helpers.storage import Store
from homeassistant.helpers.typing import ConfigType
from homeassistant.helpers.system_info import async_get_system_info

from .const import CONF_SYSTEM_PROMPT, DEFAULT_MODEL

from .const import (
    DOMAIN,
    MAX_CONVERSATION_HISTORY,
    HISTORY_PRUNING_THRESHOLD,
    MAX_CONCURRENT_REQUESTS,
)
from .tools import WeatherTool, StockTool, WebSearchTool

_LOGGER = logging.getLogger(__name__)

@dataclass
class ConversationHistoryItem:
    """Class to hold conversation history items."""
    text: str
    timestamp: float
    conversation_id: str

class OllamaAgent(conversation.ConversationEntity, conversation.AbstractConversationAgent):
    """Class to handle Ollama conversation processing."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True
    _attr_should_expose = True

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the conversation agent."""
        conversation.AbstractConversationAgent.__init__(self)
        conversation.ConversationEntity.__init__(self, hass)
        
        self.hass = hass
        self.entry = entry
        self._conversation_history = {}
        self._request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._store = Store(hass, 1, f"{DOMAIN}.conversations")
        
        # Set up device info
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Ollama",
            model=entry.data.get("model", "llama2"),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

        # Initialize tools
        self.tools = [
            WeatherTool(hass),
            StockTool(hass),
            WebSearchTool(hass)
        ]

        # Get the shared aiohttp session
        self._session = async_get_clientsession(hass)

        # Store connection info
        self._host = entry.data[CONF_HOST]
        self._port = entry.data[CONF_PORT]
        self._model = entry.data.get("model", DEFAULT_MODEL)
        self._system_prompt = entry.data.get(CONF_SYSTEM_PROMPT)

    def _prune_history(self, conversation_id: str) -> None:
        """Prune conversation history if it exceeds threshold."""
        if conversation_id not in self._conversation_history:
            return

        history = self._conversation_history[conversation_id]
        if len(history) > HISTORY_PRUNING_THRESHOLD:
            # Keep only the most recent items
            self._conversation_history[conversation_id] = deque(
                list(history)[-MAX_CONVERSATION_HISTORY:],
                maxlen=MAX_CONVERSATION_HISTORY
            )

    def _add_to_history(self, conversation_id: str, text: str) -> None:
        """Add an item to conversation history."""
        if conversation_id not in self._conversation_history:
            self._conversation_history[conversation_id] = deque(maxlen=MAX_CONVERSATION_HISTORY)

        history_item = ConversationHistoryItem(
            text=text,
            timestamp=time(),
            conversation_id=conversation_id
        )
        self._conversation_history[conversation_id].append(history_item)
        self._prune_history(conversation_id)

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        # Migrate engine if needed
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self._attr_unique_id, self.entity_id
        )
        # Register as conversation agent
        conversation.async_set_agent(self.hass, self._attr_unique_id, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        # Unregister as conversation agent
        conversation.async_unset_agent(self.hass, self._attr_unique_id)
        await super().async_will_remove_from_hass()

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a conversation input."""
        intent_response = intent.IntentResponse(language=user_input.language)

        # Create chat log
        chat_log = conversation.ChatLog(
            conversation_id=user_input.conversation_id,
            timestamp=dt_util.utcnow(),
            language=user_input.language,
        )

        try:
            async with self._request_semaphore:
                start_time = time()
                
                # Add user input to chat log
                chat_log.async_add_content(
                    conversation.UserContent(text=user_input.text)
                )

                # Add input to history for persistence
                self._add_to_history(user_input.conversation_id, user_input.text)
                
                # Process the input with tools
                url = f"http://{self._host}:{self._port}/api/generate"
                headers = {"Content-Type": "application/json"}
                
                payload = {
                    "model": self._model,
                    "prompt": user_input.text,
                    "system": self._system_prompt if self._system_prompt else "",
                    "tools": [tool.to_dict() for tool in self.tools]
                }

                async with self._session.post(url, json=payload, headers=headers, timeout=TIMEOUT_BACKOFF) as response:
                    if response.status != 200:
                        if response.status == 429:
                            raise conversation.RateLimitError("Rate limited by Ollama")
                        elif response.status == 401:
                            raise conversation.IntentHandleError("Authentication failed")
                        else:
                            raise conversation.IntentHandleError(
                                f"API request failed with status {response.status}"
                            )
                    result = await response.json()
                    response_text = result.get("response", "")

                # Add response to chat log and history
                if response_text:
                    chat_log.async_add_content(
                        conversation.AssistantContent(text=response_text)
                    )
                    self._add_to_history(user_input.conversation_id, f"Assistant: {response_text}")

                # Record statistics and trace
                duration = time() - start_time
                await self._record_statistics(duration, bool(response_text))
                chat_log.async_trace({"duration": duration, "success": bool(response_text)})
                
                # Set speech in intent response
                intent_response.async_set_speech(response_text or "Error processing request")
                
                result = conversation.ConversationResult(
                    response=intent_response,
                    conversation_id=chat_log.conversation_id,
                    chat_log=chat_log,
                    error=not bool(response_text)
                )
                if self._attr_supports_streaming:
                    result.response.response_type = intent.IntentResponseType.ACTION
                return result

        except (conversation.RateLimitError, conversation.IntentHandleError) as err:
            # Already properly formatted errors
            error_msg = str(err)
            _LOGGER.error(error_msg)
            chat_log.async_trace({"error": error_msg})
            intent_response.async_set_speech(error_msg)
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=chat_log.conversation_id,
                chat_log=chat_log,
                error=True
            )
        except Exception as err:
            error_msg = f"Error processing conversation: {err}"
            _LOGGER.error(error_msg)
            chat_log.async_trace({"error": error_msg})
            intent_response.async_set_speech("Error processing request")
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=chat_log.conversation_id,
                chat_log=chat_log,
                error=True
            )

    async def _record_statistics(self, duration: float, success: bool) -> None:
        """Record request statistics."""
        now = dt_util.utcnow()
        
        # Add statistics using HA's statistics system
        self.hass.async_add_executor_job(
            self.hass.statistics.async_add_external_statistics,
            {
                "domain": DOMAIN,
                "start": now,
                "mean": duration,
                "state": 1.0 if success else 0.0,
                "state_characteristic": "latency",
            }
        )

    async def async_save(self) -> None:
        """Save conversation history."""
        data = {
            conversation_id: list(history)
            for conversation_id, history in self._conversation_history.items()
        }
        await self._store.async_save(data)

    async def async_load(self) -> None:
        """Load conversation history."""
        data = await self._store.async_load()
        if data:
            self._conversation_history = {
                conv_id: deque(history, maxlen=MAX_CONVERSATION_HISTORY)
                for conv_id, history in data.items()
            }

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        # We support all languages as we use Home Assistant's translation system
        return ["*"]

    @property
    def language(self) -> str:
        """Return the current language."""
        return self.hass.config.language

    async def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostics information for debugging."""
        system_info = await self.hass.helpers.system_info.async_get_system_info()

        # Get statistics from HA's statistics API
        stats = await self.hass.statistics.async_get_statistics(
            statistic_ids=[f"{DOMAIN}:latency"],
            period="5minute",
            units={"mean": "seconds", "state": "count"}
        )

        return {
            "system_info": system_info,
            "host": self._host,
            "port": self._port,
            "model": self._model,
            "conversation_history_size": sum(len(h) for h in self._conversation_history.values()),
            "active_conversations": len(self._conversation_history),
            "statistics": stats
        }
