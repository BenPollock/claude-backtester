"""Fee models for simulated order execution."""

from abc import ABC, abstractmethod

from backtester.portfolio.order import Order
from backtester.types import Side


class FeeModel(ABC):
    @abstractmethod
    def compute(self, order: Order, fill_price: float, quantity: int) -> float:
        """Return commission/fee amount for this fill."""
        ...


class PerTradeFee(FeeModel):
    """Flat fee per trade regardless of size."""

    def __init__(self, fee: float = 0.0):
        self._fee = fee

    def compute(self, order: Order, fill_price: float, quantity: int) -> float:
        return self._fee


class PercentageFee(FeeModel):
    """Fee as a percentage of notional value, specified in basis points.

    fee = fill_price * quantity * bps / 10_000
    """

    def __init__(self, bps: float = 5.0):
        self._bps = bps

    def compute(self, order: Order, fill_price: float, quantity: int) -> float:
        notional = fill_price * quantity
        return notional * self._bps / 10_000


class TieredFee(FeeModel):
    """Tiered percentage fee with different rates by notional value bands.

    Bands are specified as a list of (threshold, bps) tuples, sorted by
    ascending threshold. Each tuple means "from this threshold up to the
    next threshold, charge this bps rate." The final tier applies to all
    notional value above its threshold.

    Example:
        tiers = [(0, 10), (10_000, 5), (100_000, 2)]
        - $0 to $10,000: 10 bps
        - $10,000 to $100,000: 5 bps
        - $100,000+: 2 bps

    Fees are marginal: each band's rate only applies to the portion of
    notional value within that band (like tax brackets).
    """

    def __init__(self, tiers: list[tuple[float, float]]):
        if not tiers:
            raise ValueError("tiers must be a non-empty list of (threshold, bps) tuples")
        # Sort by threshold ascending
        self._tiers = sorted(tiers, key=lambda t: t[0])

    def compute(self, order: Order, fill_price: float, quantity: int) -> float:
        notional = fill_price * quantity
        total_fee = 0.0

        for i, (threshold, bps) in enumerate(self._tiers):
            # Determine the upper bound of this tier
            if i + 1 < len(self._tiers):
                next_threshold = self._tiers[i + 1][0]
            else:
                next_threshold = float("inf")

            # Amount of notional value in this tier
            tier_base = max(0.0, min(notional, next_threshold) - threshold)
            total_fee += tier_base * bps / 10_000

        return total_fee


class SECFee(FeeModel):
    """SEC Section 31 regulatory fee, applied to sell orders only.

    The SEC charges a fee on the aggregate dollar amount of sales.
    Currently ~$8 per $1,000,000 of notional value (8.0 per million).
    """

    def __init__(self, rate_per_million: float = 8.0):
        self._rate_per_million = rate_per_million

    def compute(self, order: Order, fill_price: float, quantity: int) -> float:
        if order.side != Side.SELL:
            return 0.0
        notional = fill_price * quantity
        return notional * self._rate_per_million / 1_000_000


class TAFFee(FeeModel):
    """FINRA Trading Activity Fee (TAF).

    Charged per share sold, with a maximum cap per trade.
    Currently ~$0.000119 per share, capped at $5.95 per trade.
    """

    def __init__(self, per_share: float = 0.000119, max_per_trade: float = 5.95):
        self._per_share = per_share
        self._max_per_trade = max_per_trade

    def compute(self, order: Order, fill_price: float, quantity: int) -> float:
        if order.side != Side.SELL:
            return 0.0
        raw_fee = quantity * self._per_share
        return min(raw_fee, self._max_per_trade)


class CompositeFee(FeeModel):
    """Combines multiple fee models by summing their individual fees.

    Useful for modeling realistic fee structures where multiple fee
    components apply (e.g., brokerage commission + SEC fee + TAF fee).
    """

    def __init__(self, models: list[FeeModel]):
        self._models = list(models)

    def compute(self, order: Order, fill_price: float, quantity: int) -> float:
        return sum(m.compute(order, fill_price, quantity) for m in self._models)
