"""Offline experiment runner for MARL Iterated Prisoner's Dilemma."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ipd_project.environment.ipd_env import COOPERATE, DEFECT, IteratedPrisonersDilemmaEnv
from ipd_project.agents.heuristic_agents import BaseAgent, build_heuristic_agent
from ipd_project.agents.q_agent import (
    OpponentAwareConfig,
    OpponentAwareQLearningAgent,
    QLearningAgent,
    QLearningConfig,
)


EXPERIMENT_DEFS = {
    "q_vs_allc": {"name": "Q_vs_AllC", "a": "q", "b": "allc"},
    "q_vs_alld": {"name": "Q_vs_AllD", "a": "q", "b": "alld"},
    "q_vs_random": {"name": "Q_vs_Random", "a": "q", "b": "random"},
    "q_vs_tft": {"name": "Q_vs_TFT", "a": "q", "b": "tft"},
    "q_vs_grim": {"name": "Q_vs_Grim", "a": "q", "b": "grim"},
    "q_vs_q": {"name": "Q_vs_Q", "a": "q", "b": "q"},
}


@dataclass
class RunConfig:
    episodes: int = 10000
    rounds: int = 150
    memory: int = 1
    noise: float = 0.01
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.9992
    epsilon_min: float = 0.02
    learner: str = "q"

    prosocial_weight: float = 0.2
    coop_bonus: float = 0.05
    exploit_penalty: float = 0.06
    negative_alpha_scale: float = 0.35
    optimistic_init: float = 0.9
    tie_break_cooperate: bool = True

    seed: int = 42
    run_sweeps: bool = True
    run_tournament: bool = True
    run_population: bool = True


class AgentWrapper:
    """Adapter to unify Q-learning and heuristic agent interfaces."""

    def __init__(
        self,
        kind: str,
        state_size: int,
        learner: str,
        q_cfg: QLearningConfig,
        oaq_cfg: OpponentAwareConfig,
        seed: int,
    ):
        self.kind = kind
        self.learner = learner
        if kind == "q":
            if learner == "oaq":
                self.agent = OpponentAwareQLearningAgent(state_size=state_size, config=oaq_cfg)
            else:
                self.agent = QLearningAgent(state_size=state_size, config=q_cfg)
        else:
            self.agent = build_heuristic_agent(kind, seed=seed)

    def reset(self) -> None:
        self.agent.reset()

    def act(self, state: int, own_history: List[int], opp_history: List[int], round_idx: int) -> int:
        if self.kind == "q":
            return self.agent.act(state)
        return self.agent.act(own_history, opp_history, round_idx)

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
        if self.kind == "q":
            self.agent.update(
                state,
                action,
                reward,
                next_state,
                done,
                opp_action=opp_action,
                opp_reward=opp_reward,
            )

    def end_episode(self) -> None:
        if self.kind == "q":
            self.agent.end_episode()

    def q_rows(self) -> List[dict]:
        if self.kind != "q":
            return []
        return self.agent.export_rows()


def tail_mean(series: List[dict], key: str, frac: float = 0.1) -> float:
    if not series:
        return 0.0
    count = max(1, int(len(series) * frac))
    values = [row[key] for row in series[-count:]]
    return float(np.mean(values))


def build_q_config(cfg: RunConfig, seed: int) -> QLearningConfig:
    return QLearningConfig(
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon,
        epsilon_decay=cfg.epsilon_decay,
        epsilon_min=cfg.epsilon_min,
        seed=seed,
    )


def build_oaq_config(cfg: RunConfig, seed: int) -> OpponentAwareConfig:
    return OpponentAwareConfig(
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon,
        epsilon_decay=cfg.epsilon_decay,
        epsilon_min=cfg.epsilon_min,
        seed=seed,
        prosocial_weight=cfg.prosocial_weight,
        coop_bonus=cfg.coop_bonus,
        exploit_penalty=cfg.exploit_penalty,
        negative_alpha_scale=cfg.negative_alpha_scale,
        optimistic_init=cfg.optimistic_init,
        tie_break_cooperate=cfg.tie_break_cooperate,
    )


def run_experiment(defn: dict, cfg: RunConfig, seed_offset: int = 0) -> dict:
    env = IteratedPrisonersDilemmaEnv(memory=cfg.memory, noise=cfg.noise, seed=cfg.seed + seed_offset)
    state_size = env.state_size

    a = AgentWrapper(
        defn["a"],
        state_size,
        cfg.learner,
        build_q_config(cfg, cfg.seed + seed_offset + 1),
        build_oaq_config(cfg, cfg.seed + seed_offset + 1),
        cfg.seed + seed_offset + 11,
    )
    b = AgentWrapper(
        defn["b"],
        state_size,
        cfg.learner,
        build_q_config(cfg, cfg.seed + seed_offset + 2),
        build_oaq_config(cfg, cfg.seed + seed_offset + 2),
        cfg.seed + seed_offset + 17,
    )

    series: List[dict] = []
    for episode in range(cfg.episodes):
        a.reset()
        b.reset()
        state = env.reset()

        hist_a: List[int] = []
        hist_b: List[int] = []
        total_ra = 0.0
        total_rb = 0.0
        coop_count = 0

        for round_idx in range(cfg.rounds):
            action_a = a.act(state, hist_a, hist_b, round_idx)
            action_b = b.act(state, hist_b, hist_a, round_idx)

            next_state, reward_a, reward_b, exec_a, exec_b = env.step(action_a, action_b)
            done = round_idx == cfg.rounds - 1

            a.update(
                state,
                action_a,
                reward_a,
                next_state,
                done,
                opp_action=exec_b,
                opp_reward=reward_b,
            )
            b.update(
                state,
                action_b,
                reward_b,
                next_state,
                done,
                opp_action=exec_a,
                opp_reward=reward_a,
            )

            hist_a.append(exec_a)
            hist_b.append(exec_b)
            total_ra += reward_a
            total_rb += reward_b
            coop_count += int(exec_a == COOPERATE) + int(exec_b == COOPERATE)
            state = next_state

        a.end_episode()
        b.end_episode()

        series.append(
            {
                "episode": episode,
                "cooperation": coop_count / (2 * cfg.rounds),
                "joint_reward": (total_ra + total_rb) / (2 * cfg.rounds),
            }
        )

    summary = {
        "final_cooperation": series[-1]["cooperation"],
        "tail_cooperation": tail_mean(series, "cooperation"),
        "tail_reward": tail_mean(series, "joint_reward"),
    }

    return {
        "name": defn["name"],
        "series": series,
        "summary": summary,
        "q_table_a": a.q_rows(),
        "q_table_b": b.q_rows(),
    }


def run_baseline_tournament(cfg: RunConfig) -> dict:
    strategies = ["allc", "alld", "random", "tft", "grim", "copykitten"]
    totals = {s: [] for s in strategies}
    coops = {s: [] for s in strategies}

    for i, s1 in enumerate(strategies):
        for j, s2 in enumerate(strategies):
            if i == j:
                continue
            env = IteratedPrisonersDilemmaEnv(memory=1, noise=cfg.noise, seed=cfg.seed + i * 101 + j)
            a = build_heuristic_agent(s1, seed=cfg.seed + i)
            b = build_heuristic_agent(s2, seed=cfg.seed + j)
            a.reset()
            b.reset()
            state = env.reset()
            hist_a: List[int] = []
            hist_b: List[int] = []
            r_a = 0.0
            r_b = 0.0
            c_a = 0
            c_b = 0

            for round_idx in range(max(40, min(220, cfg.rounds))):
                act_a = a.act(hist_a, hist_b, round_idx)
                act_b = b.act(hist_b, hist_a, round_idx)
                state, ra, rb, ea, eb = env.step(act_a, act_b)
                hist_a.append(ea)
                hist_b.append(eb)
                r_a += ra
                r_b += rb
                c_a += int(ea == COOPERATE)
                c_b += int(eb == COOPERATE)

            denom = float(max(40, min(220, cfg.rounds)))
            totals[s1].append(r_a / denom)
            totals[s2].append(r_b / denom)
            coops[s1].append(c_a / denom)
            coops[s2].append(c_b / denom)

    rows = []
    for s in strategies:
        rows.append(
            {
                "strategy": s.upper(),
                "mean_reward": float(np.mean(totals[s])) if totals[s] else 0.0,
                "mean_cooperation": float(np.mean(coops[s])) if coops[s] else 0.0,
            }
        )

    rows = sorted(rows, key=lambda x: x["mean_reward"], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return {"leaderboard": rows}


def run_population(cfg: RunConfig) -> dict:
    population_size = 10
    episodes = max(160, cfg.episodes // 5)
    rounds = max(60, min(cfg.rounds, 110))

    env = IteratedPrisonersDilemmaEnv(memory=cfg.memory, noise=cfg.noise, seed=cfg.seed + 777)
    if cfg.learner == "oaq":
        agents = [
            OpponentAwareQLearningAgent(state_size=env.state_size, config=build_oaq_config(cfg, cfg.seed + i + 1000))
            for i in range(population_size)
        ]
    else:
        agents = [
            QLearningAgent(state_size=env.state_size, config=build_q_config(cfg, cfg.seed + i + 1000))
            for i in range(population_size)
        ]

    rng = np.random.default_rng(cfg.seed + 99)
    series: List[dict] = []

    for episode in range(episodes):
        order = list(range(population_size))
        rng.shuffle(order)
        coop_episode = []
        reward_episode = []

        for i in range(0, population_size, 2):
            a_idx = order[i]
            b_idx = order[i + 1]
            a = agents[a_idx]
            b = agents[b_idx]

            state = env.reset()
            hist_a: List[int] = []
            hist_b: List[int] = []
            total_ra = 0.0
            total_rb = 0.0
            coop_count = 0

            for round_idx in range(rounds):
                act_a = a.act(state)
                act_b = b.act(state)
                next_state, ra, rb, ea, eb = env.step(act_a, act_b)
                done = round_idx == rounds - 1
                a.update(state, act_a, ra, next_state, done, opp_action=eb, opp_reward=rb)
                b.update(state, act_b, rb, next_state, done, opp_action=ea, opp_reward=ra)
                hist_a.append(ea)
                hist_b.append(eb)
                total_ra += ra
                total_rb += rb
                coop_count += int(ea == COOPERATE) + int(eb == COOPERATE)
                state = next_state

            a.end_episode()
            b.end_episode()
            coop_episode.append(coop_count / (2 * rounds))
            reward_episode.append((total_ra + total_rb) / (2 * rounds))

        series.append(
            {
                "episode": episode,
                "cooperation": float(np.mean(coop_episode)),
                "joint_reward": float(np.mean(reward_episode)),
            }
        )

    return {
        "series": series,
        "summary": {
            "tail_cooperation": tail_mean(series, "cooperation"),
            "tail_reward": tail_mean(series, "joint_reward"),
            "episodes": episodes,
        },
    }


def run_sweeps(cfg: RunConfig) -> dict:
    base = {"name": "Sweep_Q_vs_Q", "a": "q", "b": "q"}
    sweeps = {
        "gamma": [0.1, 0.5, 0.9, 0.99],
        "noise": [0.0, 0.01, 0.05, 0.1],
        "memory": [1, 3, 5],
    }
    output: Dict[str, List[dict]] = {"gamma": [], "noise": [], "memory": []}

    for key, values in sweeps.items():
        for idx, value in enumerate(values):
            sweep_cfg = copy.deepcopy(cfg)
            sweep_cfg.episodes = max(180, cfg.episodes // 4)
            sweep_cfg.rounds = max(50, min(cfg.rounds, 110))
            if key == "gamma":
                sweep_cfg.gamma = float(value)
            elif key == "noise":
                sweep_cfg.noise = float(value)
            else:
                sweep_cfg.memory = int(value)

            run = run_experiment(base, sweep_cfg, seed_offset=500 + idx)
            output[key].append(
                {
                    "value": value,
                    "tail_cooperation": run["summary"]["tail_cooperation"],
                    "tail_reward": run["summary"]["tail_reward"],
                }
            )
    return output


def to_records_for_csv(results: dict) -> pd.DataFrame:
    rows = []
    for exp_name, payload in results["experiments"].items():
        row = {
            "experiment": exp_name,
            "tail_cooperation": payload["summary"]["tail_cooperation"],
            "tail_reward": payload["summary"]["tail_reward"],
            "final_cooperation": payload["summary"]["final_cooperation"],
        }
        rows.append(row)

    if results.get("population"):
        rows.append(
            {
                "experiment": "Population_Q_vs_Q",
                "tail_cooperation": results["population"]["summary"]["tail_cooperation"],
                "tail_reward": results["population"]["summary"]["tail_reward"],
                "final_cooperation": np.nan,
            }
        )

    return pd.DataFrame(rows)


def run_all(cfg: RunConfig, selected_experiments: List[str]) -> dict:
    experiments: Dict[str, dict] = {}
    for idx, key in enumerate(selected_experiments):
        if key not in EXPERIMENT_DEFS:
            continue
        exp = run_experiment(EXPERIMENT_DEFS[key], cfg, seed_offset=idx * 100)
        experiments[exp["name"]] = exp

    sweeps = run_sweeps(cfg) if cfg.run_sweeps else {}
    tournament = run_baseline_tournament(cfg) if cfg.run_tournament else {}
    population = run_population(cfg) if cfg.run_population else {}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg.__dict__,
        "experiments": experiments,
        "sweeps": sweeps,
        "tournament": tournament,
        "population": population,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline MARL IPD experiments.")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--memory", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.9992)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--learner", type=str, default="q", choices=["q", "oaq"])
    parser.add_argument("--prosocial-weight", type=float, default=0.2)
    parser.add_argument("--coop-bonus", type=float, default=0.05)
    parser.add_argument("--exploit-penalty", type=float, default=0.06)
    parser.add_argument("--negative-alpha-scale", type=float, default=0.35)
    parser.add_argument("--optimistic-init", type=float, default=0.9)
    parser.add_argument("--tie-break", type=str, default="cooperate", choices=["cooperate", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiments", type=str, default=",".join(EXPERIMENT_DEFS.keys()))
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--no-sweeps", action="store_true")
    parser.add_argument("--no-tournament", action="store_true")
    parser.add_argument("--no-population", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        episodes=args.episodes,
        rounds=args.rounds,
        memory=args.memory,
        noise=args.noise,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        learner=args.learner,
        prosocial_weight=args.prosocial_weight,
        coop_bonus=args.coop_bonus,
        exploit_penalty=args.exploit_penalty,
        negative_alpha_scale=args.negative_alpha_scale,
        optimistic_init=args.optimistic_init,
        tie_break_cooperate=args.tie_break == "cooperate",
        seed=args.seed,
        run_sweeps=not args.no_sweeps,
        run_tournament=not args.no_tournament,
        run_population=not args.no_population,
    )

    selected_experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]
    results = run_all(cfg, selected_experiments)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"results_{ts}.json"
    latest_path = out_dir / "latest_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_df = to_records_for_csv(results)
    summary_df.to_csv(out_dir / f"summary_{ts}.csv", index=False)
    summary_df.to_csv(out_dir / "latest_summary.csv", index=False)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {out_dir / f'summary_{ts}.csv'}")


if __name__ == "__main__":
    main()
