"""Hand-crafted strategy agents for Iterated Prisoner's Dilemma."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import random

COOPERATE = 0
DEFECT = 1


class BaseAgent:
    def reset(self) -> None:
        pass

    def act(self, own_history: List[int], opp_history: List[int], round_idx: int) -> int:
        raise NotImplementedError

    def observe(self, *_args, **_kwargs) -> None:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Agent", "")


class AlwaysCooperateAgent(BaseAgent):
    def act(self, own_history: List[int], opp_history: List[int], round_idx: int) -> int:
        return COOPERATE


class AlwaysDefectAgent(BaseAgent):
    def act(self, own_history: List[int], opp_history: List[int], round_idx: int) -> int:
        return DEFECT


@dataclass
class RandomAgent(BaseAgent):
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def act(self, own_history: List[int], opp_history: List[int], round_idx: int) -> int:
        return COOPERATE if self.rng.random() < 0.5 else DEFECT


class TitForTatAgent(BaseAgent):
    def act(self, own_history: List[int], opp_history: List[int], round_idx: int) -> int:
        return COOPERATE if not opp_history else opp_history[-1]


class GrimTriggerAgent(BaseAgent):
    def __init__(self) -> None:
        self.triggered = False

    def reset(self) -> None:
        self.triggered = False

    def act(self, own_history: List[int], opp_history: List[int], round_idx: int) -> int:
        if self.triggered:
            return DEFECT
        if opp_history and opp_history[-1] == DEFECT:
            self.triggered = True
            return DEFECT
        return COOPERATE


class CopykittenAgent(BaseAgent):
    def act(self, own_history: List[int], opp_history: List[int], round_idx: int) -> int:
        if len(opp_history) < 2:
            return COOPERATE
        return DEFECT if opp_history[-1] == DEFECT and opp_history[-2] == DEFECT else COOPERATE


def build_heuristic_agent(kind: str, seed: int | None = None) -> BaseAgent:
    key = kind.lower()
    if key in {"allc", "always_cooperate"}:
        return AlwaysCooperateAgent()
    if key in {"alld", "always_defect"}:
        return AlwaysDefectAgent()
    if key in {"random", "rand"}:
        return RandomAgent(seed=seed)
    if key in {"tft", "tit_for_tat"}:
        return TitForTatAgent()
    if key in {"grim", "grim_trigger"}:
        return GrimTriggerAgent()
    if key in {"copykitten", "tf2t", "tit_for_two_tats"}:
        return CopykittenAgent()
    raise ValueError(f"Unknown heuristic agent kind: {kind}")
