"""League/opponent-mixture training for IPD learners."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from ipd_project.agents.heuristic_agents import BaseAgent, build_heuristic_agent
from ipd_project.agents.q_agent import (
    OpponentAwareConfig,
    OpponentAwareQLearningAgent,
    QLearningAgent,
    QLearningConfig,
)
from ipd_project.environment.ipd_env import COOPERATE, IteratedPrisonersDilemmaEnv


LEAGUE_OPPONENTS = ["allc", "alld", "random", "tft", "grim", "copykitten"]


def build_q_cfg(args: argparse.Namespace, seed: int) -> QLearningConfig:
    return QLearningConfig(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        seed=seed,
    )


def build_oaq_cfg(args: argparse.Namespace, seed: int) -> OpponentAwareConfig:
    return OpponentAwareConfig(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        seed=seed,
        prosocial_weight=args.prosocial_weight,
        coop_bonus=args.coop_bonus,
        exploit_penalty=args.exploit_penalty,
        negative_alpha_scale=args.negative_alpha_scale,
        optimistic_init=args.optimistic_init,
        tie_break_cooperate=args.tie_break == "cooperate",
    )


def make_learner(args: argparse.Namespace, state_size: int, seed: int):
    if args.learner == "oaq":
        return OpponentAwareQLearningAgent(state_size=state_size, config=build_oaq_cfg(args, seed))
    return QLearningAgent(state_size=state_size, config=build_q_cfg(args, seed))


def act_heuristic(agent: BaseAgent, own_hist: List[int], opp_hist: List[int], round_idx: int) -> int:
    return agent.act(own_hist, opp_hist, round_idx)


def train_league(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    env = IteratedPrisonersDilemmaEnv(memory=args.memory, noise=args.noise, seed=args.seed + 1)
    learner = make_learner(args, env.state_size, args.seed + 101)
    opponent_counts = {k: 0 for k in LEAGUE_OPPONENTS}
    series: List[dict] = []

    for episode in range(args.train_episodes):
        opp_key = str(rng.choice(LEAGUE_OPPONENTS))
        opponent_counts[opp_key] += 1
        opponent = build_heuristic_agent(opp_key, seed=args.seed + 2000 + episode)
        opponent.reset()
        learner.reset()
        state = env.reset()
        learner_hist: List[int] = []
        opp_hist: List[int] = []
        total_reward = 0.0
        coop_count = 0

        for round_idx in range(args.rounds):
            act_l = learner.act(state)
            act_o = act_heuristic(opponent, opp_hist, learner_hist, round_idx)
            next_state, r_l, r_o, e_l, e_o = env.step(act_l, act_o)
            done = round_idx == args.rounds - 1
            learner.update(
                state,
                act_l,
                r_l,
                next_state,
                done,
                opp_action=e_o,
                opp_reward=r_o,
            )
            learner_hist.append(e_l)
            opp_hist.append(e_o)
            total_reward += r_l
            coop_count += int(e_l == COOPERATE)
            state = next_state

        learner.end_episode()
        series.append(
            {
                "episode": episode,
                "opponent": opp_key,
                "cooperation": coop_count / args.rounds,
                "reward": total_reward / args.rounds,
            }
        )

    return {
        "learner": learner,
        "training_series": series,
        "opponent_counts": opponent_counts,
    }


def clone_for_eval(agent):
    out = copy.deepcopy(agent)
    if hasattr(out, "epsilon"):
        out.epsilon = 0.0
    return out


def evaluate_matchup(agent, opponent_key: str, args: argparse.Namespace, seed_offset: int) -> dict:
    env = IteratedPrisonersDilemmaEnv(memory=args.memory, noise=args.noise, seed=args.seed + seed_offset)
    rounds = args.rounds
    cooperations: List[float] = []
    rewards: List[float] = []

    for ep in range(args.eval_episodes):
        a = clone_for_eval(agent)
        if opponent_key == "self":
            b = clone_for_eval(agent)
        else:
            b = build_heuristic_agent(opponent_key, seed=args.seed + seed_offset + ep)
            b.reset()
        a.reset()
        state = env.reset()
        hist_a: List[int] = []
        hist_b: List[int] = []
        total_ra = 0.0
        total_rb = 0.0
        coop_count = 0

        for r in range(rounds):
            act_a = a.act(state)
            if opponent_key == "self":
                act_b = b.act(state)
            else:
                act_b = act_heuristic(b, hist_b, hist_a, r)
            state, ra, rb, ea, eb = env.step(act_a, act_b)
            hist_a.append(ea)
            hist_b.append(eb)
            total_ra += ra
            total_rb += rb
            coop_count += int(ea == COOPERATE) + int(eb == COOPERATE)

        cooperations.append(coop_count / (2 * rounds))
        rewards.append((total_ra + total_rb) / (2 * rounds))

    return {
        "tail_cooperation": float(np.mean(cooperations)),
        "tail_reward": float(np.mean(rewards)),
    }


def evaluate_league(agent, args: argparse.Namespace) -> dict:
    keys = ["self", "allc", "alld", "random", "tft", "grim", "copykitten"]
    out: Dict[str, dict] = {}
    for idx, key in enumerate(keys):
        out[key] = evaluate_matchup(agent, key, args, seed_offset=5000 + idx * 137)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="League training for IPD learners.")
    parser.add_argument("--learner", type=str, default="oaq", choices=["q", "oaq"])
    parser.add_argument("--train-episodes", type=int, default=6000)
    parser.add_argument("--eval-episodes", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--memory", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.01)

    parser.add_argument("--alpha", type=float, default=0.16)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.9994)
    parser.add_argument("--epsilon-min", type=float, default=0.0)

    parser.add_argument("--prosocial-weight", type=float, default=0.4)
    parser.add_argument("--coop-bonus", type=float, default=0.02)
    parser.add_argument("--exploit-penalty", type=float, default=0.1)
    parser.add_argument("--negative-alpha-scale", type=float, default=0.5)
    parser.add_argument("--optimistic-init", type=float, default=0.8)
    parser.add_argument("--tie-break", type=str, choices=["cooperate", "random"], default="cooperate")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results/league_runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trained = train_league(args)
    eval_summary = evaluate_league(trained["learner"], args)
    tail_window = max(1, len(trained["training_series"]) // 10)
    tail_rows = trained["training_series"][-tail_window:]

    score = (
        0.45 * eval_summary["self"]["tail_cooperation"]
        + 0.20 * eval_summary["self"]["tail_reward"] / 3.0
        + 0.15 * eval_summary["tft"]["tail_cooperation"]
        + 0.10 * (1.0 - eval_summary["alld"]["tail_cooperation"])
        + 0.10 * eval_summary["grim"]["tail_cooperation"]
    ) * 100.0

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "league_opponents": LEAGUE_OPPONENTS,
        "opponent_counts": trained["opponent_counts"],
        "training_tail_cooperation": float(np.mean([x["cooperation"] for x in tail_rows])),
        "training_tail_reward": float(np.mean([x["reward"] for x in tail_rows])),
        "evaluation": eval_summary,
        "league_score": float(score),
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"league_results_{ts}.json"
    latest_json = out_dir / "latest_league_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with latest_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved league results: {out_json}")
    print("Evaluation summary:")
    for key, val in eval_summary.items():
        print(f"  {key:>10s} coop={val['tail_cooperation']:.3f} reward={val['tail_reward']:.3f}")
    print(f"League score: {score:.3f}")


if __name__ == "__main__":
    main()
