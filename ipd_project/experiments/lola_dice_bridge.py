"""Bridge runner for external LOLA/DiCE IPD implementations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import types
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


MODULE_FILE = {
    "dice": "ipd_DiCE.py",
    "dice_om": "ipd_DiCE_om.py",
    "exact": "ipd_exact.py",
    "exact_om": "ipd_exact_om.py",
}
DEFAULT_REPO_URL = "https://github.com/alexis-jacq/LOLA_DiCE"


def parse_int_csv(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LOLA/DiCE via external repo bridge.")
    parser.add_argument("--variant", type=str, choices=list(MODULE_FILE.keys()), default="dice")
    parser.add_argument("--lookaheads", type=str, default="0,1,2")
    parser.add_argument("--n-update", type=int, default=120)
    parser.add_argument("--len-rollout", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-dir", type=str, default="external/LOLA_DiCE")
    parser.add_argument("--repo-url", type=str, default=DEFAULT_REPO_URL)
    parser.add_argument("--no-auto-clone", action="store_true")
    parser.add_argument("--out-dir", type=str, default="results/lola_dice_runs")
    return parser.parse_args()


def ensure_repo(repo_dir: Path, repo_url: str, auto_clone: bool) -> None:
    if repo_dir.exists():
        return
    if not auto_clone:
        raise FileNotFoundError(f"Repo dir not found: {repo_dir}")
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)


def load_module(repo_dir: Path, variant: str):
    module_file = repo_dir / MODULE_FILE[variant]
    if not module_file.exists():
        raise FileNotFoundError(f"Missing LOLA module file: {module_file}")

    # Some LOLA_DiCE files import `gym`; on modern setups we may only have gymnasium.
    try:
        import gym  # type: ignore
    except ModuleNotFoundError:
        import gymnasium as gym  # type: ignore

        sys.modules["gym"] = gym
        sys.modules["gym.spaces"] = gym.spaces
    if not hasattr(gym.spaces, "prng"):
        gym.spaces.prng = types.SimpleNamespace(np_random=np.random.default_rng(0))

    # Inject minimal package-like `envs` exposing IPD only (skip coin_game imports).
    envs_pkg = types.ModuleType("envs")
    envs_pkg.__path__ = [str(repo_dir / "envs")]
    sys.modules["envs"] = envs_pkg

    common_file = repo_dir / "envs" / "common.py"
    common_spec = importlib.util.spec_from_file_location("envs.common", str(common_file))
    if common_spec is None or common_spec.loader is None:
        raise RuntimeError(f"Could not load envs.common from {common_file}")
    common_mod = importlib.util.module_from_spec(common_spec)
    common_spec.loader.exec_module(common_mod)
    sys.modules["envs.common"] = common_mod

    pd_file = repo_dir / "envs" / "prisoners_dilemma.py"
    pd_spec = importlib.util.spec_from_file_location("envs.prisoners_dilemma", str(pd_file))
    if pd_spec is None or pd_spec.loader is None:
        raise RuntimeError(f"Could not load envs.prisoners_dilemma from {pd_file}")
    pd_mod = importlib.util.module_from_spec(pd_spec)
    pd_spec.loader.exec_module(pd_mod)
    sys.modules["envs.prisoners_dilemma"] = pd_mod
    envs_pkg.IPD = pd_mod.IteratedPrisonersDilemma

    sys.path.insert(0, str(repo_dir))
    spec = importlib.util.spec_from_file_location(f"lola_{variant}", str(module_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {module_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_hp(mod, args: argparse.Namespace) -> None:
    if not hasattr(mod, "hp"):
        raise RuntimeError("Loaded module has no `hp` config object.")
    mod.hp.n_update = int(args.n_update)
    mod.hp.len_rollout = int(args.len_rollout)
    mod.hp.batch_size = int(args.batch_size)
    mod.hp.seed = int(args.seed)
    if hasattr(mod.hp, "use_baseline"):
        mod.hp.use_baseline = True

    # Rebuild environment using updated rollout/batch shape if factory exists.
    if hasattr(mod, "IPD"):
        mod.ipd = mod.IPD(mod.hp.len_rollout, mod.hp.batch_size)


def run_variant(mod, lookaheads: List[int], seed: int) -> List[dict]:
    rows: List[dict] = []
    for la in lookaheads:
        torch.manual_seed(seed)
        np.random.seed(seed)
        scores = mod.play(mod.Agent(), mod.Agent(), la)
        final = float(scores[-1]) if scores else 0.0
        tail_count = max(1, len(scores) // 10)
        tail = float(np.mean(scores[-tail_count:])) if scores else 0.0
        rows.append(
            {
                "lookaheads": la,
                "final_joint_score": final,
                "tail_joint_score": tail,
                "n_updates": len(scores),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    lookaheads = parse_int_csv(args.lookaheads)
    repo_dir = Path(args.repo_dir)
    ensure_repo(repo_dir, args.repo_url, auto_clone=not args.no_auto_clone)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mod = load_module(repo_dir, args.variant)
    set_hp(mod, args)
    rows = run_variant(mod, lookaheads, args.seed)
    rows_sorted = sorted(rows, key=lambda x: x["tail_joint_score"], reverse=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "repo_dir": str(repo_dir),
        "lookaheads": lookaheads,
        "hp": {
            "n_update": args.n_update,
            "len_rollout": args.len_rollout,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "results": rows,
        "best": rows_sorted[0] if rows_sorted else None,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"lola_{args.variant}_{ts}.json"
    latest_json = out_dir / f"latest_lola_{args.variant}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with latest_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved LOLA bridge results: {out_json}")
    for row in rows:
        print(
            f"lookaheads={row['lookaheads']} tail_joint_score={row['tail_joint_score']:.3f} "
            f"final_joint_score={row['final_joint_score']:.3f}"
        )


if __name__ == "__main__":
    main()
