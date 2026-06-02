"""Market-data providers behind a common interface.

``AlpacaProvider`` is the default concrete implementation. The ABC exists so a
different source could be dropped in via config without touching strategy code.
"""

from daytrader.data.providers.base import MarketDataProvider, Timeframe
from daytrader.data.providers.factory import get_provider

__all__ = ["MarketDataProvider", "Timeframe", "get_provider"]
