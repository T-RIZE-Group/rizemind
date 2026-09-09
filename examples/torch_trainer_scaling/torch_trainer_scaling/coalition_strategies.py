from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

Coalition = tuple[int, ...]


def normalize_coalition(
    members: tuple[int, ...] | list[int], total_trainers: int
) -> Coalition:
    coalition = tuple(sorted(set(members)))
    for member in coalition:
        if member < 0 or member >= total_trainers:
            raise ValueError(
                f"Coalition member {member} is outside the valid range "
                f"[0, {total_trainers - 1}]."
            )
    return coalition


def coalition_to_mask(coalition: Coalition) -> int:
    mask = 0
    for member in coalition:
        mask |= 1 << member
    return mask


def coalition_to_label(coalition: Coalition) -> str:
    if not coalition:
        return "empty"
    return ",".join(str(member) for member in coalition)


def parse_coalition_text(raw_value: str, total_trainers: int) -> Coalition:
    cleaned = raw_value.strip()
    if cleaned.lower() in {"", "empty", "none", "{}"}:
        return ()

    members = [int(value.strip()) for value in cleaned.split(",") if value.strip()]
    return normalize_coalition(members, total_trainers)


def load_manual_coalitions(
    raw_value: str | None, file_path: str | None, total_trainers: int
) -> list[Coalition]:
    coalitions: list[Coalition] = []

    if raw_value:
        for chunk in raw_value.split(";"):
            cleaned = chunk.strip()
            if cleaned:
                coalitions.append(parse_coalition_text(cleaned, total_trainers))

    if file_path:
        for line in Path(file_path).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            coalitions.append(parse_coalition_text(stripped, total_trainers))

    return coalitions


@dataclass(frozen=True)
class CoalitionStrategyConfig:
    strategy: str
    total_trainers: int
    seed: int
    budget: int | None = None
    budget_per_size: int | None = None
    min_size: int = 0
    max_size: int | None = None
    include_empty: bool = True
    include_grand: bool = True
    manual_coalitions: tuple[Coalition, ...] = ()


class CoalitionFeederStrategy(ABC):
    name: str

    def __init__(self, config: CoalitionStrategyConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self) -> list[Coalition]:
        """Generate coalitions to evaluate."""

    def _grand_coalition(self) -> Coalition:
        return tuple(range(self.config.total_trainers))

    def _is_allowed(self, coalition: Coalition) -> bool:
        if not coalition and not self.config.include_empty:
            return False
        if coalition == self._grand_coalition() and not self.config.include_grand:
            return False
        return True

    def _finalize(self, coalitions: list[Coalition]) -> list[Coalition]:
        seen: set[Coalition] = set()
        ordered: list[Coalition] = []

        for coalition in coalitions:
            if not self._is_allowed(coalition):
                continue
            if coalition not in seen:
                ordered.append(coalition)
                seen.add(coalition)

        if self.config.include_empty and () not in seen:
            ordered.insert(0, ())
            seen.add(())

        grand = self._grand_coalition()
        if self.config.include_grand and grand not in seen:
            ordered.append(grand)

        return ordered


class AllCoalitionsStrategy(CoalitionFeederStrategy):
    name = "all"

    def generate(self) -> list[Coalition]:
        coalitions = [
            coalition
            for size in range(self.config.total_trainers + 1)
            for coalition in itertools.combinations(
                range(self.config.total_trainers), size
            )
        ]
        return self._finalize(list(coalitions))


class UniformCoalitionsStrategy(CoalitionFeederStrategy):
    name = "uniform"

    def generate(self) -> list[Coalition]:
        if self.config.budget is None or self.config.budget <= 0:
            raise ValueError("Uniform strategy requires a positive --budget.")

        rng = random.Random(self.config.seed)
        total_masks = 1 << self.config.total_trainers
        population = [
            tuple(
                idx
                for idx in range(self.config.total_trainers)
                if mask & (1 << idx) != 0
            )
            for mask in range(total_masks)
        ]
        allowed_population = [
            coalition for coalition in population if self._is_allowed(coalition)
        ]

        required: list[Coalition] = []
        if self.config.include_empty:
            required.append(())
        if self.config.include_grand:
            required.append(self._grand_coalition())
        required = list(dict.fromkeys(required))

        if self.config.budget < len(required):
            raise ValueError(
                "Budget is smaller than the number of forced coalitions "
                "(empty and/or grand coalition)."
            )
        if self.config.budget > len(allowed_population):
            raise ValueError(
                "Budget exceeds the number of valid coalitions for the current "
                "include-empty/include-grand settings."
            )

        remaining_population = [
            coalition for coalition in allowed_population if coalition not in required
        ]
        needed = self.config.budget - len(required)
        ordered = list(required)
        if needed > 0:
            ordered.extend(rng.sample(remaining_population, needed))
        return self._finalize(ordered)


class StratifiedCoalitionsStrategy(CoalitionFeederStrategy):
    name = "stratified"

    def generate(self) -> list[Coalition]:
        if self.config.budget_per_size is None or self.config.budget_per_size <= 0:
            raise ValueError(
                "Stratified strategy requires a positive --budget-per-size."
            )

        max_size = (
            self.config.total_trainers
            if self.config.max_size is None
            else self.config.max_size
        )
        if self.config.min_size < 0 or max_size > self.config.total_trainers:
            raise ValueError("Stratum size bounds are outside the valid trainer range.")
        if self.config.min_size > max_size:
            raise ValueError("--min-size cannot be greater than --max-size.")

        rng = random.Random(self.config.seed)
        chosen: list[Coalition] = []

        for size in range(self.config.min_size, max_size + 1):
            all_of_size = list(
                itertools.combinations(range(self.config.total_trainers), size)
            )
            if len(all_of_size) <= self.config.budget_per_size:
                chosen.extend(tuple(coalition) for coalition in all_of_size)
                continue

            sampled = rng.sample(all_of_size, self.config.budget_per_size)
            sampled.sort()
            chosen.extend(tuple(coalition) for coalition in sampled)

        return self._finalize(chosen)


class ManualCoalitionsStrategy(CoalitionFeederStrategy):
    name = "manual"

    def generate(self) -> list[Coalition]:
        if not self.config.manual_coalitions:
            raise ValueError(
                "Manual strategy requires coalitions from --coalitions or --coalitions-file."
            )
        return self._finalize(list(self.config.manual_coalitions))


STRATEGY_REGISTRY: dict[str, type[CoalitionFeederStrategy]] = {
    AllCoalitionsStrategy.name: AllCoalitionsStrategy,
    UniformCoalitionsStrategy.name: UniformCoalitionsStrategy,
    StratifiedCoalitionsStrategy.name: StratifiedCoalitionsStrategy,
    ManualCoalitionsStrategy.name: ManualCoalitionsStrategy,
}


def build_strategy(config: CoalitionStrategyConfig) -> CoalitionFeederStrategy:
    if config.strategy not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(
            f"Unknown strategy '{config.strategy}'. Available strategies: {available}"
        )
    return STRATEGY_REGISTRY[config.strategy](config)
