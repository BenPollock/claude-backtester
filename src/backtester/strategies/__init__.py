"""Strategies package -- auto-discovers and registers all strategy modules."""

from backtester.strategies.registry import discover_strategies

discover_strategies()
