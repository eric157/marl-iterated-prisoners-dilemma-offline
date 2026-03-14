"""Q-learning agents for Iterated Prisoner's Dilemma."""

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
class OpponentAwareConfig:
    alpha: float = 0.08
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.9994
    epsilon_min: float = 0.01
    seed: int | None = None

    prosocial_weight: float = 0.2
    coop_bonus: float = 0.05
    exploit_penalty: float = 0.06
    negative_alpha_scale: float = 0.35
    optimistic_init: float = 0.9
    tie_break_cooperate: bool = True


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

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        opp_action: int | None = None,
        opp_reward: float | None = None,
    ) -> None:
        del opp_action, opp_reward
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


@dataclass
class OpponentAwareQLearningAgent:
    """Joint-action Q-learning with opponent response modeling and prosocial shaping."""

    state_size: int
    config: OpponentAwareConfig = field(default_factory=OpponentAwareConfig)

    def __post_init__(self) -> None:
        self.q_table = np.full((self.state_size, 2, 2), self.config.optimistic_init, dtype=np.float64)
        # Bias initial preferences toward mutual cooperation.
        self.q_table[:, COOPERATE, COOPERATE] += self.config.coop_bonus
        self.q_table[:, DEFECT, COOPERATE] -= self.config.exploit_penalty

        # response_counts[state, my_action, opp_action]
        self.response_counts = np.ones((self.state_size, 2, 2), dtype=np.float64)
        self.epsilon = self.config.epsilon
        self.rng = random.Random(self.config.seed)

    def reset(self) -> None:
        pass

    def _predicted_opp_coop_prob(self, state: int, my_action: int) -> float:
        row = self.response_counts[state, my_action]
        denom = float(np.sum(row))
        if denom <= 0:
            return 0.5
        return float(row[COOPERATE] / denom)

    def _expected_value(self, state: int, my_action: int) -> float:
        p_opp_c = self._predicted_opp_coop_prob(state, my_action)
        q_c = self.q_table[state, my_action, COOPERATE]
        q_d = self.q_table[state, my_action, DEFECT]
        return float(p_opp_c * q_c + (1.0 - p_opp_c) * q_d)

    def act(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return COOPERATE if self.rng.random() < 0.5 else DEFECT
        v_c = self._expected_value(state, COOPERATE)
        v_d = self._expected_value(state, DEFECT)
        if abs(v_c - v_d) < 1e-12:
            if self.config.tie_break_cooperate:
                return COOPERATE
            return COOPERATE if self.rng.random() < 0.5 else DEFECT
        return COOPERATE if v_c > v_d else DEFECT

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        opp_action: int | None = None,
        opp_reward: float | None = None,
    ) -> None:
        if opp_action is None:
            raise ValueError("OpponentAwareQLearningAgent requires observed opponent action.")

        self.response_counts[state, action, opp_action] += 1.0

        opp_rew = reward if opp_reward is None else opp_reward
        shaped_reward = (1.0 - self.config.prosocial_weight) * reward + self.config.prosocial_weight * opp_rew
        if action == COOPERATE and opp_action == COOPERATE:
            shaped_reward += self.config.coop_bonus
        elif action == DEFECT and opp_action == COOPERATE:
            shaped_reward -= self.config.exploit_penalty

        current_q = self.q_table[state, action, opp_action]
        if done:
            next_best = 0.0
        else:
            next_best = max(
                self._expected_value(next_state, COOPERATE),
                self._expected_value(next_state, DEFECT),
            )
        td_target = shaped_reward + self.config.gamma * next_best
        td_error = td_target - current_q
        lr = self.config.alpha if td_error >= 0 else self.config.alpha * self.config.negative_alpha_scale
        self.q_table[state, action, opp_action] = current_q + lr * td_error

    def end_episode(self) -> None:
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def export_rows(self) -> List[dict]:
        rows: List[dict] = []
        for state in range(self.state_size):
            q_coop = self._expected_value(state, COOPERATE)
            q_defect = self._expected_value(state, DEFECT)
            rows.append(
                {
                    "state": state,
                    "q_cooperate": float(q_coop),
                    "q_defect": float(q_defect),
                    "preferred": "C" if q_coop >= q_defect else "D",
                    "p_opp_cooperate_if_c": float(self._predicted_opp_coop_prob(state, COOPERATE)),
                    "p_opp_cooperate_if_d": float(self._predicted_opp_coop_prob(state, DEFECT)),
                }
            )
        return rows
