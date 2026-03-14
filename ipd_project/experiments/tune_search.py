"""Random-search tuner for offline IPD learners."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from ipd_project.experiments.run_experiments import RunConfig, run_all


def compute_score(result: dict, objective: str = "balanced") -> float:
    experiments = result.get("experiments", {})
    qvq = experiments.get("Q_vs_Q", {}).get("summary", {})
    qvtft = experiments.get("Q_vs_TFT", {}).get("summary", {})
    qalld = experiments.get("Q_vs_AllD", {}).get("summary", {})
    pop = result.get("population", {}).get("summary", {})

    qvq_coop = float(qvq.get("tail_cooperation", 0.0))
    qvq_rew = float(qvq.get("tail_reward", 0.0)) / 3.0
    qvtft_coop = float(qvtft.get("tail_cooperation", 0.0))
    qalld_coop = float(qalld.get("tail_cooperation", 0.0))
    pop_coop = float(pop.get("tail_cooperation", 0.0))
    pop_rew = float(pop.get("tail_reward", 0.0)) / 3.0

    qvq_rew = max(0.0, min(1.0, qvq_rew))
    pop_rew = max(0.0, min(1.0, pop_rew))

    if objective == "cooperation":
        score01 = 0.50 * qvq_coop + 0.20 * qvtft_coop + 0.16 * pop_coop + 0.08 * (1.0 - qalld_coop) + 0.06 * qvq_rew
    elif objective == "reward":
        score01 = 0.42 * qvq_rew + 0.18 * pop_rew + 0.16 * qvq_coop + 0.14 * qvtft_coop + 0.10 * (1.0 - qalld_coop)
    else:
        score01 = 0.40 * qvq_coop + 0.20 * qvq_rew + 0.17 * qvtft_coop + 0.08 * (1.0 - qalld_coop) + 0.10 * pop_coop + 0.05 * pop_rew
    return float(score01 * 100.0)


def evaluate_candidate(cfg: RunConfig, seeds: List[int], experiments: List[str], objective: str) -> Dict[str, float]:
    scores: List[float] = []
    qvq_coop: List[float] = []
    qvq_rew: List[float] = []
    for seed in seeds:
        trial_cfg = RunConfig(**asdict(cfg))
        trial_cfg.seed = seed
        result = run_all(trial_cfg, experiments)
        scores.append(compute_score(result, objective))
        qvq = result.get("experiments", {}).get("Q_vs_Q", {}).get("summary", {})
        qvq_coop.append(float(qvq.get("tail_cooperation", 0.0)))
        qvq_rew.append(float(qvq.get("tail_reward", 0.0)))
    return {
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "qvq_tail_coop_mean": float(np.mean(qvq_coop)),
        "qvq_tail_reward_mean": float(np.mean(qvq_rew)),
    }


def sample_candidate(rng: np.random.Generator, learner: str, base: RunConfig) -> RunConfig:
    cfg = RunConfig(**asdict(base))
    cfg.learner = learner
    cfg.alpha = float(rng.choice([0.03, 0.05, 0.08, 0.1, 0.12, 0.16]))
    cfg.gamma = float(rng.choice([0.9, 0.95, 0.97, 0.99]))
    cfg.epsilon_decay = float(rng.choice([0.9988, 0.9990, 0.9992, 0.9994, 0.9996]))
    cfg.epsilon_min = float(rng.choice([0.0, 0.005, 0.01, 0.02, 0.03]))

    if learner == "oaq":
        cfg.prosocial_weight = float(rng.choice([0.05, 0.1, 0.2, 0.3, 0.4]))
        cfg.coop_bonus = float(rng.choice([0.0, 0.02, 0.05, 0.08, 0.12]))
        cfg.exploit_penalty = float(rng.choice([0.0, 0.03, 0.06, 0.1, 0.15]))
        cfg.negative_alpha_scale = float(rng.choice([0.2, 0.35, 0.5, 0.7, 1.0]))
        cfg.optimistic_init = float(rng.choice([0.0, 0.4, 0.8, 1.2, 1.6, 2.0]))
        cfg.tie_break_cooperate = bool(rng.choice([True, False], p=[0.8, 0.2]))
    else:
        cfg.prosocial_weight = 0.0
        cfg.coop_bonus = 0.0
        cfg.exploit_penalty = 0.0
        cfg.negative_alpha_scale = 1.0
        cfg.optimistic_init = 0.0
        cfg.tie_break_cooperate = bool(rng.choice([True, False], p=[0.3, 0.7]))
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune IPD learner via random search.")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--learner", type=str, default="oaq", choices=["q", "oaq"])
    parser.add_argument("--objective", type=str, default="balanced", choices=["balanced", "cooperation", "reward"])
    parser.add_argument("--episodes", type=int, default=900)
    parser.add_argument("--rounds", type=int, default=140)
    parser.add_argument("--memory", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--seeds", type=str, default="11,13,17,19,23")
    parser.add_argument("--experiments", type=str, default="q_vs_q,q_vs_tft,q_vs_alld,q_vs_allc,q_vs_random,q_vs_grim")
    parser.add_argument("--out-dir", type=str, default="results/tuning_runs")
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = RunConfig(
        episodes=args.episodes,
        rounds=args.rounds,
        memory=args.memory,
        noise=args.noise,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.9992,
        epsilon_min=0.02,
        learner=args.learner,
        run_sweeps=False,
        run_tournament=False,
        run_population=True,
    )

    rng = np.random.default_rng(args.seed)
    leaderboard: List[dict] = []
    best_row: dict | None = None
    for t in range(args.trials):
        candidate = sample_candidate(rng, args.learner, base)
        stats = evaluate_candidate(candidate, seeds, experiments, args.objective)
        row = {
            "trial": t + 1,
            "learner": args.learner,
            "objective": args.objective,
            "config": asdict(candidate),
            **stats,
        }
        leaderboard.append(row)
        if best_row is None or row["score_mean"] > best_row["score_mean"]:
            best_row = row
        print(
            f"trial={t + 1:02d} score={row['score_mean']:.3f} "
            f"qvq_coop={row['qvq_tail_coop_mean']:.3f} qvq_reward={row['qvq_tail_reward_mean']:.3f}"
        )

    leaderboard.sort(key=lambda x: x["score_mean"], reverse=True)
    if best_row is None:
        raise RuntimeError("No trial results found.")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"tuning_report_{ts}.json"
    latest_json = out_dir / "latest_tuning_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trials": args.trials,
        "seeds": seeds,
        "objective": args.objective,
        "top": leaderboard[:10],
        "best": leaderboard[0],
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with latest_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved tuning report: {out_json}")
    print("Best config:")
    print(json.dumps(leaderboard[0], indent=2))


if __name__ == "__main__":
    main()
