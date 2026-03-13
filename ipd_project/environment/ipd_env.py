"""Iterated Prisoner's Dilemma environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import random

COOPERATE = 0
DEFECT = 1


@dataclass(frozen=True)
class PayoffMatrix:
    """Payoffs for (A action, B action)."""

    reward_cc: Tuple[float, float] = (3.0, 3.0)
    reward_cd: Tuple[float, float] = (0.0, 5.0)
    reward_dc: Tuple[float, float] = (5.0, 0.0)
    reward_dd: Tuple[float, float] = (1.0, 1.0)

    def reward(self, action_a: int, action_b: int) -> Tuple[float, float]:
        if action_a == COOPERATE and action_b == COOPERATE:
            return self.reward_cc
        if action_a == COOPERATE and action_b == DEFECT:
            return self.reward_cd
        if action_a == DEFECT and action_b == COOPERATE:
            return self.reward_dc
        return self.reward_dd


class IteratedPrisonersDilemmaEnv:
    """IPD environment with configurable memory length and action noise."""

    def __init__(
        self,
        memory: int = 1,
        noise: float = 0.0,
        payoff_matrix: PayoffMatrix | None = None,
        seed: int | None = None,
    ) -> None:
        if memory < 1:
            raise ValueError("memory must be >= 1")
        if not 0.0 <= noise <= 1.0:
            raise ValueError("noise must be in [0, 1]")

        self.memory = memory
        self.noise = noise
        self.payoff = payoff_matrix or PayoffMatrix()
        self.rng = random.Random(seed)
        self.history: List[int] = []

    @property
    def state_size(self) -> int:
        return 4 ** self.memory

    def _encode_joint_action(self, action_a: int, action_b: int) -> int:
        return (action_a << 1) | action_b

    def _decode_joint_action(self, encoded: int) -> Tuple[int, int]:
        return (encoded >> 1) & 1, encoded & 1

    def reset(self) -> int:
        """Initialize with cooperative history and return initial state."""
        self.history = [self._encode_joint_action(COOPERATE, COOPERATE)] * self.memory
        return self._encode_state()

    def _encode_state(self) -> int:
        state = 0
        for joint in self.history:
            state = state * 4 + joint
        return state

    def decode_state(self, state: int) -> List[int]:
        seq = [0] * self.memory
        value = state
        for idx in range(self.memory - 1, -1, -1):
            seq[idx] = value % 4
            value //= 4
        return seq

    def apply_noise(self, action: int) -> int:
        if self.rng.random() < self.noise:
            return COOPERATE if action == DEFECT else DEFECT
        return action

    def step(self, action_a: int, action_b: int) -> tuple[int, float, float, int, int]:
        """Take one step and return next_state, rewards, executed actions."""
        exec_a = self.apply_noise(action_a)
        exec_b = self.apply_noise(action_b)
        reward_a, reward_b = self.payoff.reward(exec_a, exec_b)

        self.history.append(self._encode_joint_action(exec_a, exec_b))
        if len(self.history) > self.memory:
            self.history.pop(0)

        next_state = self._encode_state()
        return next_state, reward_a, reward_b, exec_a, exec_b
