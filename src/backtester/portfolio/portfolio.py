"""Portfolio: cash, positions, equity tracking, and trade logging."""

from dataclasses import dataclass, field
from datetime import date

from backtester.portfolio.position import Position
from backtester.portfolio.order import Trade, TradeLogEntry


@dataclass(frozen=True)
class PortfolioState:
    """Read-only snapshot passed to strategies."""

    cash: float
    total_equity: float
    num_positions: int
    position_symbols: frozenset[str]
    margin_used: float = 0.0


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    equity_history: list[tuple[date, float]] = field(default_factory=list)
    trade_log: list[Trade] = field(default_factory=list)
    activity_log: list[TradeLogEntry] = field(default_factory=list)

    @property
    def total_equity(self) -> float:
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value

    @property
    def num_positions(self) -> int:
        return len(self.positions)

    @property
    def margin_used(self) -> float:
        """Sum of abs(market_value) for short positions.

        This is the raw short exposure before applying the margin
        requirement multiplier.  The engine multiplies by
        ``config.margin_requirement`` to get the actual margin reserved.
        """
        return sum(
            abs(pos.market_value)
            for pos in self.positions.values()
            if pos.is_short
        )

    def available_capital(self, margin_requirement: float = 1.5) -> float:
        """Cash minus margin reserved for short positions."""
        return self.cash - self.margin_used * margin_requirement

    def snapshot(self) -> PortfolioState:
        return PortfolioState(
            cash=self.cash,
            total_equity=self.total_equity,
            num_positions=self.num_positions,
            position_symbols=frozenset(self.positions.keys()),
            margin_used=self.margin_used,
        )

    def record_equity(self, current_date: date) -> None:
        self.equity_history.append((current_date, self.total_equity))

    def get_position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def open_position(self, symbol: str) -> Position:
        """Get or create a position for the symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def close_position(self, symbol: str) -> None:
        """Remove a position after all lots are sold/covered."""
        if symbol in self.positions and self.positions[symbol].total_quantity == 0:
            del self.positions[symbol]

    def max_position_value(self) -> float:
        if not self.positions:
            return 0.0
        return max(pos.market_value for pos in self.positions.values())

    def position_weight(self, symbol: str) -> float:
        """Return position's weight as fraction of total equity."""
        equity = self.total_equity
        if equity <= 0 or symbol not in self.positions:
            return 0.0
        return self.positions[symbol].market_value / equity
