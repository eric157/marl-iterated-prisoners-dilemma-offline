"""Q-learning agent for Iterated Prisoner's Dilemma."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import random
import numpy as np

COOPERATE = 0
DEFECT = 1


@dataclass
class QLearningConfig:
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.999
    epsilon_min: float = 0.02
    seed: int | None = None


@dataclass
class QLearningAgent:
    state_size: int
    config: QLearningConfig = field(default_factory=QLearningConfig)

    def __post_init__(self) -> None:
        self.q_table = np.zeros((self.state_size, 2), dtype=np.float64)
        self.epsilon = self.config.epsilon
        self.rng = random.Random(self.config.seed)

    def reset(self) -> None:
        pass

    def act(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return COOPERATE if self.rng.random() < 0.5 else DEFECT
        row = self.q_table[state]
        if abs(row[COOPERATE] - row[DEFECT]) < 1e-12:
            return COOPERATE if self.rng.random() < 0.5 else DEFECT
        return int(np.argmax(row))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        current_q = self.q_table[state, action]
        next_best = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.config.gamma * next_best
        self.q_table[state, action] = current_q + self.config.alpha * (td_target - current_q)

    def end_episode(self) -> None:
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def export_rows(self) -> List[dict]:
        rows: List[dict] = []
        for state in range(self.state_size):
            qc = float(self.q_table[state, COOPERATE])
            qd = float(self.q_table[state, DEFECT])
            rows.append(
                {
                    "state": state,
                    "q_cooperate": qc,
                    "q_defect": qd,
                    "preferred": "C" if qc >= qd else "D",
                }
            )
        return rows
