"""Multi-seed benchmark harness for strong/offline IPD evaluation."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from ipd_project.experiments.run_experiments import RunConfig, run_all


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def ci95(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci = 1.96 * std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": float(ci)}


def get_metric(result: dict, exp_name: str, key: str) -> float:
    return float(result.get("experiments", {}).get(exp_name, {}).get("summary", {}).get(key, 0.0))


def compute_balanced_score(result: dict) -> float:
    qvq_coop = get_metric(result, "Q_vs_Q", "tail_cooperation")
    qvq_reward = get_metric(result, "Q_vs_Q", "tail_reward")
    qvtft_coop = get_metric(result, "Q_vs_TFT", "tail_cooperation")
    qalld_coop = get_metric(result, "Q_vs_AllD", "tail_cooperation")
    pop = result.get("population", {}).get("summary", {})
    pop_coop = float(pop.get("tail_cooperation", 0.0))
    pop_reward = float(pop.get("tail_reward", 0.0))

    qvq_reward_norm = max(0.0, min(1.0, qvq_reward / 3.0))
    pop_reward_norm = max(0.0, min(1.0, pop_reward / 3.0))

    score01 = (
        0.40 * qvq_coop
        + 0.20 * qvq_reward_norm
        + 0.17 * qvtft_coop
        + 0.08 * (1.0 - qalld_coop)
        + 0.10 * pop_coop
        + 0.05 * pop_reward_norm
    )
    return float(score01 * 100.0)


@dataclass
class BenchmarkArgs:
    episodes: int
    rounds: int
    memory: int
    noise_list: List[float]
    seeds: List[int]
    learners: List[str]
    experiments: List[str]
    out_dir: str
    alpha: float
    gamma: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    prosocial_weight: float
    coop_bonus: float
    exploit_penalty: float
    negative_alpha_scale: float
    optimistic_init: float
    tie_break: str


def parse_args() -> BenchmarkArgs:
    parser = argparse.ArgumentParser(description="Run multi-seed offline benchmark suite.")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--rounds", type=int, default=160)
    parser.add_argument("--memory", type=int, default=1)
    parser.add_argument("--noise-list", type=str, default="0.0,0.01,0.05,0.1")
    parser.add_argument("--seeds", type=str, default="11,13,17,19,23,29,31,37,41,43")
    parser.add_argument("--learners", type=str, default="q,oaq")
    parser.add_argument("--experiments", type=str, default="q_vs_q,q_vs_tft,q_vs_alld,q_vs_allc,q_vs_random,q_vs_grim")
    parser.add_argument("--out-dir", type=str, default="results/sota_suite")

    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.9994)
    parser.add_argument("--epsilon-min", type=float, default=0.01)

    parser.add_argument("--prosocial-weight", type=float, default=0.2)
    parser.add_argument("--coop-bonus", type=float, default=0.05)
    parser.add_argument("--exploit-penalty", type=float, default=0.06)
    parser.add_argument("--negative-alpha-scale", type=float, default=0.35)
    parser.add_argument("--optimistic-init", type=float, default=0.9)
    parser.add_argument("--tie-break", type=str, choices=["cooperate", "random"], default="cooperate")

    ns = parser.parse_args()
    return BenchmarkArgs(
        episodes=ns.episodes,
        rounds=ns.rounds,
        memory=ns.memory,
        noise_list=parse_csv_floats(ns.noise_list),
        seeds=parse_csv_ints(ns.seeds),
        learners=parse_csv_strings(ns.learners),
        experiments=parse_csv_strings(ns.experiments),
        out_dir=ns.out_dir,
        alpha=ns.alpha,
        gamma=ns.gamma,
        epsilon=ns.epsilon,
        epsilon_decay=ns.epsilon_decay,
        epsilon_min=ns.epsilon_min,
        prosocial_weight=ns.prosocial_weight,
        coop_bonus=ns.coop_bonus,
        exploit_penalty=ns.exploit_penalty,
        negative_alpha_scale=ns.negative_alpha_scale,
        optimistic_init=ns.optimistic_init,
        tie_break=ns.tie_break,
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: List[dict] = []
    all_payloads: List[dict] = []

    total_runs = len(args.learners) * len(args.noise_list) * len(args.seeds)
    run_idx = 0
    for learner in args.learners:
        for noise in args.noise_list:
            for seed in args.seeds:
                run_idx += 1
                cfg = RunConfig(
                    episodes=args.episodes,
                    rounds=args.rounds,
                    memory=args.memory,
                    noise=noise,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    epsilon=args.epsilon,
                    epsilon_decay=args.epsilon_decay,
                    epsilon_min=args.epsilon_min,
                    learner=learner,
                    prosocial_weight=args.prosocial_weight,
                    coop_bonus=args.coop_bonus,
                    exploit_penalty=args.exploit_penalty,
                    negative_alpha_scale=args.negative_alpha_scale,
                    optimistic_init=args.optimistic_init,
                    tie_break_cooperate=args.tie_break == "cooperate",
                    seed=seed,
                    run_sweeps=False,
                    run_tournament=False,
                    run_population=True,
                )
                print(f"[{run_idx}/{total_runs}] learner={learner} noise={noise:.2f} seed={seed}")
                result = run_all(cfg, args.experiments)
                score = compute_balanced_score(result)
                all_payloads.append(result)

                raw_rows.append(
                    {
                        "learner": learner,
                        "noise": noise,
                        "seed": seed,
                        "score_balanced": score,
                        "qvq_tail_coop": get_metric(result, "Q_vs_Q", "tail_cooperation"),
                        "qvq_tail_reward": get_metric(result, "Q_vs_Q", "tail_reward"),
                        "qvtft_tail_coop": get_metric(result, "Q_vs_TFT", "tail_cooperation"),
                        "qalld_tail_coop": get_metric(result, "Q_vs_AllD", "tail_cooperation"),
                        "pop_tail_coop": float(result.get("population", {}).get("summary", {}).get("tail_cooperation", 0.0)),
                        "pop_tail_reward": float(result.get("population", {}).get("summary", {}).get("tail_reward", 0.0)),
                    }
                )

    raw_df = pd.DataFrame(raw_rows)

    grouped_rows: List[dict] = []
    for (learner, noise), g in raw_df.groupby(["learner", "noise"], sort=True):
        row = {"learner": learner, "noise": noise, "n": int(len(g))}
        for metric in [
            "score_balanced",
            "qvq_tail_coop",
            "qvq_tail_reward",
            "qvtft_tail_coop",
            "qalld_tail_coop",
            "pop_tail_coop",
            "pop_tail_reward",
        ]:
            stats = ci95(g[metric].tolist())
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_ci95"] = stats["ci95"]
        grouped_rows.append(row)

    grouped_df = pd.DataFrame(grouped_rows).sort_values("score_balanced_mean", ascending=False)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw_csv = out_dir / f"benchmark_raw_{ts}.csv"
    group_csv = out_dir / f"benchmark_summary_{ts}.csv"
    latest_raw = out_dir / "latest_benchmark_raw.csv"
    latest_summary = out_dir / "latest_benchmark_summary.csv"
    latest_json = out_dir / "latest_benchmark_summary.json"

    raw_df.to_csv(raw_csv, index=False)
    raw_df.to_csv(latest_raw, index=False)
    grouped_df.to_csv(group_csv, index=False)
    grouped_df.to_csv(latest_summary, index=False)

    json_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "args": asdict(args),
        "top_configs": grouped_df.head(10).to_dict(orient="records"),
        "raw_count": int(len(raw_df)),
    }
    with latest_json.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    print(f"Saved raw:     {raw_csv}")
    print(f"Saved summary: {group_csv}")
    print("Top rows:")
    print(grouped_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
