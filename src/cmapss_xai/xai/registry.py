from __future__ import annotations

from collections.abc import Callable

from cmapss_xai.xai.base import BaseExplainer


class ExplainerRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], BaseExplainer]] = {}

    def register(self, name: str, factory: Callable[[], BaseExplainer]) -> None:
        normalized = name.lower().strip()
        if not normalized:
            raise ValueError("Explainer name must not be empty")
        if normalized in self._factories:
            raise ValueError(f"Explainer already registered: {normalized}")
        self._factories[normalized] = factory

    def create(self, name: str) -> BaseExplainer:
        normalized = name.lower().strip()
        if normalized not in self._factories:
            available = ", ".join(sorted(self._factories)) or "none"
            raise KeyError(f"Unknown explainer {normalized}. Available: {available}")
        return self._factories[normalized]()

    def names(self) -> list[str]:
        return sorted(self._factories)
