"""Typed configuration loader.

Merges two sources at startup and validates them with pydantic:

* ``config/config.yaml`` — non-secret runtime config (risk, strategies, ...).
* ``.env`` — secrets only (API keys, SMTP password).

Validation happens eagerly: a malformed config or out-of-range risk value
raises immediately on boot rather than surfacing mid-trade. Use
:func:`get_settings` everywhere; it caches a single validated instance.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = three levels up from this file:
#   src/daytrader/config/settings.py -> src/daytrader -> src -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


# ════════════════════════════════════════════════════════════
#  Secrets (.env)
# ════════════════════════════════════════════════════════════
class Secrets(BaseSettings):
    """Secret values, loaded from environment / ``.env``.

    Every field is optional so the app can boot (and be unit-tested) without
    credentials; individual engines validate the specific secrets they need
    and degrade gracefully or log a clear error when one is missing.
    """

    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    finnhub_api_key: str | None = None
    newsapi_api_key: str | None = None

    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    email_from: str | None = None
    email_to: str | None = None

    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    @property
    def has_alpaca(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)


# ════════════════════════════════════════════════════════════
#  config.yaml sub-models
# ════════════════════════════════════════════════════════════
class AppConfig(BaseModel):
    simulation_mode: bool = True
    timezone: str = "America/New_York"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    db_path: str = "data/trading.db"


class DataConfig(BaseModel):
    provider: Literal["alpaca"] = "alpaca"
    feed: Literal["iex", "sip"] = "iex"
    timeframes: list[str] = Field(default_factory=lambda: ["1m", "5m", "15m", "1d"])
    poll_interval_seconds: int = Field(default=60, gt=0)
    # Restrict intraday bars to regular trading hours (09:30-16:00 ET). Without
    # this, vendors that include extended-hours bars corrupt the opening range,
    # session VWAP, and relative-volume baselines.
    rth_only: bool = True


class ResearchFilters(BaseModel):
    min_avg_daily_volume: int = Field(default=1_000_000, ge=0)
    min_price: float = Field(default=5.0, ge=0)
    max_price: float = Field(default=1000.0, gt=0)
    min_gap_percent: float = Field(default=2.0, ge=0)
    min_relative_volume: float = Field(default=1.5, ge=0)

    @model_validator(mode="after")
    def _price_band(self) -> "ResearchFilters":
        if self.max_price <= self.min_price:
            raise ValueError("research.filters.max_price must exceed min_price")
        return self


class SentimentConfig(BaseModel):
    enabled: bool = True
    provider: Literal["finnhub", "newsapi", "none"] = "finnhub"
    lookback_hours: int = Field(default=12, gt=0)
    min_articles: int = Field(default=3, ge=0)


class ResearchConfig(BaseModel):
    universe: list[str] = Field(default_factory=lambda: ["sp500"])
    max_watchlist_size: int = Field(default=15, gt=0)
    filters: ResearchFilters = Field(default_factory=ResearchFilters)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)


class MacdConfig(BaseModel):
    fast: int = Field(default=12, gt=0)
    slow: int = Field(default=26, gt=0)
    signal: int = Field(default=9, gt=0)

    @model_validator(mode="after")
    def _fast_lt_slow(self) -> "MacdConfig":
        if self.fast >= self.slow:
            raise ValueError("indicators.macd.fast must be < slow")
        return self


class IndicatorsConfig(BaseModel):
    ema_periods: list[int] = Field(default_factory=lambda: [9, 21, 50, 200])
    rsi_period: int = Field(default=14, gt=0)
    atr_period: int = Field(default=14, gt=0)
    macd: MacdConfig = Field(default_factory=MacdConfig)


class StopLossConfig(BaseModel):
    method: Literal["atr", "structural"] = "atr"
    atr_multiplier: float = Field(default=2.0, gt=0)


class TakeProfitConfig(BaseModel):
    method: Literal["trailing", "fixed"] = "trailing"
    risk_reward_ratio: float = Field(default=2.0, gt=0)
    trailing_atr_multiplier: float = Field(default=1.5, gt=0)


class KellyConfig(BaseModel):
    """Fractional-Kelly risk sizing. Disabled until per-strategy stats are
    statistically meaningful; always hard-capped by ``max_risk_per_trade_pct``."""

    enabled: bool = False
    fraction: float = Field(default=0.25, gt=0, le=1.0)
    min_trades: int = Field(default=50, gt=0)
    lookback_trades: int = Field(default=200, gt=0)


class RiskConfig(BaseModel):
    starting_equity: float = Field(default=100_000.0, gt=0)
    max_risk_per_trade_pct: float = Field(default=1.0, gt=0, le=100)
    max_daily_drawdown_pct: float = Field(default=3.0, gt=0, le=100)
    max_open_positions: int = Field(default=5, gt=0)
    max_position_size_pct: float = Field(default=20.0, gt=0, le=100)
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = Field(default_factory=TakeProfitConfig)
    slippage_pct: float = Field(default=0.05, ge=0)
    commission_per_trade: float = Field(default=0.0, ge=0)
    # Marketable-limit entry collar (basis points past the reference price).
    # Caps the worst acceptable entry instead of blindly crossing the spread.
    entry_collar_bps: float = Field(default=15.0, ge=0)
    kelly: KellyConfig = Field(default_factory=KellyConfig)


class StrategyToggle(BaseModel):
    """Common base: every strategy has an ``enabled`` flag plus extra params."""

    model_config = {"extra": "allow"}
    enabled: bool = False


class PfGateConfig(BaseModel):
    """Rolling profit-factor deployment gate.

    Blocks new entries when the last ``lookback_trades`` closed round-trips
    have profit factor below ``min_pf``. Requires at least ``min_trades``
    history before the gate activates.
    """

    mode: Literal["off", "shadow", "enforce"] = "off"
    lookback_trades: int = Field(default=20, gt=0)
    min_trades: int = Field(default=10, gt=0)
    min_pf: float = Field(default=0.90, ge=0)


class VolGateConfig(BaseModel):
    """Day-level volatility gate (prior-session VIX / SPY ATR proxy).

    Blocks new entries on elevated-volatility sessions while still managing
    open positions. Modes mirror the regime filter: off | shadow | enforce.
    """

    mode: Literal["off", "shadow", "enforce"] = "off"
    vix_symbol: str = "VIX"
    max_vix: float = Field(default=28.0, ge=0)
    max_vix_percentile: float = Field(default=0.85, ge=0, le=1.0)
    use_spy_proxy: bool = True
    max_spy_atr_percentile: float = Field(default=0.80, ge=0, le=1.0)


class RegimeFilterConfig(BaseModel):
    """Gates which strategies may run in which market regime.

    Modes: ``off`` (no classification), ``shadow`` (classify + log what would
    be blocked, but let everything through), ``enforce`` (actually block).
    """

    mode: Literal["off", "shadow", "enforce"] = "off"
    # strategy name -> regimes in which it is allowed to trade.
    allowed: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "opening_range_breakout": ["trend"],
            "momentum_scalp": ["trend"],
            "vwap_pullback": ["balanced"],
        }
    )


class StrategiesConfig(BaseModel):
    model_config = {"extra": "allow"}
    vwap_pullback: StrategyToggle = Field(default_factory=StrategyToggle)
    opening_range_breakout: StrategyToggle = Field(default_factory=StrategyToggle)
    momentum_scalp: StrategyToggle = Field(default_factory=StrategyToggle)
    regime_filter: RegimeFilterConfig = Field(default_factory=RegimeFilterConfig)
    pf_gate: PfGateConfig = Field(default_factory=PfGateConfig)
    vol_gate: VolGateConfig = Field(default_factory=VolGateConfig)


class ScheduleConfig(BaseModel):
    premarket_research: str = "07:00"
    market_open: str = "09:30"
    market_close: str = "16:00"
    # Flatten BEFORE the close: live market orders submitted after 16:00 are
    # rejected, which would silently carry positions overnight.
    flatten_time: str = "15:55"
    report_time: str = "16:15"

    @field_validator("*")
    @classmethod
    def _valid_hhmm(cls, v: str) -> str:
        try:
            hh, mm = v.split(":")
            if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
                raise ValueError
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"schedule time must be 'HH:MM', got {v!r}") from exc
        return v


class ReportingConfig(BaseModel):
    output_dir: str = "data/reports"
    chart_timeframe: str = "5m"
    send_email: bool = True
    attach_individual_charts: bool = True


class MetaFilterConfig(BaseModel):
    """Meta-label signal filter (secondary ML quality gate).

    ``shadow`` scores every signal and logs what it would block without
    blocking; ``enforce`` drops signals below ``threshold``. With no trained
    model on disk the filter is inert regardless of mode.
    """

    mode: Literal["off", "shadow", "enforce"] = "off"
    threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    model_path: str = "data/models/meta_label.pkl"


class MLConfig(BaseModel):
    meta_filter: MetaFilterConfig = Field(default_factory=MetaFilterConfig)


# ════════════════════════════════════════════════════════════
#  Top-level settings
# ════════════════════════════════════════════════════════════
class Settings(BaseModel):
    """Fully validated application configuration."""

    app: AppConfig = Field(default_factory=AppConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    ml: MLConfig = Field(default_factory=MLConfig)

    # Secrets are attached after construction (not part of YAML).
    secrets: Secrets = Field(default_factory=Secrets)

    @property
    def db_url(self) -> str:
        """Absolute SQLAlchemy URL for the SQLite database."""
        path = Path(self.app.db_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path}"

    @classmethod
    def load(
        cls,
        config_path: Path | str | None = None,
        secrets: Secrets | None = None,
    ) -> "Settings":
        """Load and validate settings from YAML + ``.env``.

        Args:
            config_path: Path to the YAML config. Defaults to
                ``config/config.yaml`` at the project root.
            secrets: Pre-built :class:`Secrets` (mainly for tests). When
                omitted, secrets are read from the environment / ``.env``.
        """
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        raw: dict = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        settings = cls.model_validate(raw)
        settings.secrets = secrets or Secrets()
        return settings


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, validated settings singleton."""
    return Settings.load()
