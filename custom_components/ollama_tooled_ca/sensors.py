"""Sensors for Ollama Tooled CA."""
from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import Any

from homeassistant.components.sensor import (
    SensorEntity,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, UnitOfTime, UnitOfInformation
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Ollama sensors based on a config entry."""
    agent = hass.data[DOMAIN][entry.entry_id]

    coordinator = DataUpdateCoordinator(
        hass,
        _LOGGER,
        name="ollama_sensors",
        update_method=_async_update_data(agent),
        update_interval=timedelta(minutes=5),
    )

    await coordinator.async_config_entry_first_refresh()

    entities = [
        CacheHitRateSensor(coordinator, entry),
        MemoryUsageSensor(coordinator, entry),
        ResponseTimeSensor(coordinator, entry),
        ConnectionPoolSensor(coordinator, entry),
        RequestsSensor(coordinator, entry),
        ErrorRateSensor(coordinator, entry),
    ]

    async_add_entities(entities)

async def _async_update_data(agent):
    """Fetch data for sensors."""
    async def _update():
        data = {}
        
        # Get cache statistics
        if hasattr(agent, 'cache_manager'):
            cache_stats = agent.cache_manager.get_stats()
            data['cache_hit_rate'] = sum(
                stats.get('hit_rate', 0) 
                for stats in cache_stats.values()
            ) / len(cache_stats) if cache_stats else 0
            data['memory_usage'] = sum(
                stats.get('memory_usage_mb', 0) 
                for stats in cache_stats.values()
            )

        # Get performance statistics
        if hasattr(agent, 'stats_manager'):
            perf_stats = agent.stats_manager.get_performance_summary()
            if 'requests' in perf_stats:
                data['avg_response_time'] = perf_stats['requests'].get('avg_duration', 0)
                data['error_rate'] = 1 - perf_stats['requests'].get('success_rate', 1)
                data['total_requests'] = perf_stats['requests'].get('total_requests', 0)

        # Get connection pool statistics
        if hasattr(agent, 'connection_pool'):
            conn_stats = agent.connection_pool.get_stats()
            data['connection_usage'] = sum(
                stats.get('active_connections', 0) / stats.get('total_connections', 1) 
                for stats in conn_stats.values()
            ) / len(conn_stats) if conn_stats else 0

        return data

    return _update

class OllamaSensorBase(CoordinatorEntity, SensorEntity):
    """Base class for Ollama sensors."""

    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._entry = entry
        self._attr_has_entity_name = True
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=f"Ollama {entry.data.get('host', 'Server')}",
            manufacturer="Ollama",
            model=entry.data.get("model", "Unknown"),
        )

class CacheHitRateSensor(OllamaSensorBase):
    """Sensor for cache hit rate."""

    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return "Cache Hit Rate"

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        return self.coordinator.data.get('cache_hit_rate', 0) * 100

class MemoryUsageSensor(OllamaSensorBase):
    """Sensor for memory usage."""

    _attr_device_class = SensorDeviceClass.DATA_SIZE
    _attr_native_unit_of_measurement = UnitOfInformation.MEGABYTES
    _attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return "Memory Usage"

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        return self.coordinator.data.get('memory_usage', 0)

class ResponseTimeSensor(OllamaSensorBase):
    """Sensor for average response time."""

    _attr_device_class = SensorDeviceClass.DURATION
    _attr_native_unit_of_measurement = UnitOfTime.SECONDS
    _attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return "Average Response Time"

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        return self.coordinator.data.get('avg_response_time', 0)

class ConnectionPoolSensor(OllamaSensorBase):
    """Sensor for connection pool usage."""

    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return "Connection Pool Usage"

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        return self.coordinator.data.get('connection_usage', 0) * 100

class RequestsSensor(OllamaSensorBase):
    """Sensor for total requests."""

    _attr_state_class = SensorStateClass.TOTAL_INCREASING

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return "Total Requests"

    @property
    def native_value(self) -> int:
        """Return the state of the sensor."""
        return self.coordinator.data.get('total_requests', 0)

class ErrorRateSensor(OllamaSensorBase):
    """Sensor for error rate."""

    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return "Error Rate"

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        return self.coordinator.data.get('error_rate', 0) * 100
