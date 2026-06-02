"""Provider factory: builds the configured market-data provider."""

from __future__ import annotations

from daytrader.config.settings import Settings
from daytrader.data.providers.alpaca_provider import AlpacaProvider
from daytrader.data.providers.base import MarketDataProvider


def get_provider(settings: Settings) -> MarketDataProvider:
    """Construct the market-data provider named in ``settings.data.provider``."""
    name = settings.data.provider
    if name == "alpaca":
        return AlpacaProvider(
            api_key=settings.secrets.alpaca_api_key,
            secret_key=settings.secrets.alpaca_secret_key,
            feed=settings.data.feed,
        )
    raise ValueError(f"Unknown data provider: {name!r}")
