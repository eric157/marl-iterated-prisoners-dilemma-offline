# MARL IPD Offline Experiments

This repository is the **offline/Python experimentation track** for the Iterated Prisoner's Dilemma (IPD) project.

It is intentionally separate from the GitHub Pages dashboard repository.

## What this repo contains

- Reproducible Python simulations
- Q-learning and baseline strategy agents
- Batch experiment runner
- Saved JSON/CSV outputs for analysis
- Plot utilities for thesis/report figures

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m ipd_project.experiments.run_experiments --episodes 10000 --rounds 150 --out-dir results
python -m ipd_project.analysis.plots --input results/latest_results.json --out-dir reports/figures
```

## New Strong-Performance Pipeline (2026)

### 1) Multi-seed benchmark suite (mean/std/95% CI)

```bash
python -m ipd_project.experiments.benchmark_suite --episodes 1800 --rounds 150 --seeds 11,13,17,19,23,29,31,37,41,43 --noise-list 0.0,0.01,0.05,0.1 --learners q
python -m ipd_project.experiments.benchmark_suite --episodes 1800 --rounds 150 --seeds 11,13,17,19,23,29,31,37,41,43 --noise-list 0.0,0.01,0.05,0.1 --learners oaq
```

### 2) Random-search tuning

```bash
python -m ipd_project.experiments.tune_search --trials 24 --learner oaq --objective balanced --episodes 900 --rounds 140 --seeds 11,13,17,19,23
```

### 3) League/opponent-mixture training

```bash
python -m ipd_project.experiments.league_training --learner oaq --train-episodes 6000 --eval-episodes 500 --rounds 150
```

### 4) LOLA/DiCE bridge integration (external repo)

```bash
python -m ipd_project.experiments.lola_dice_bridge --variant dice --lookaheads 0,1,2 --n-update 120 --len-rollout 120 --batch-size 96
```

The bridge expects the external repo at `external/LOLA_DiCE` (auto-clone in workflow) and PyTorch installed.
Optional deps:

```bash
pip install -r requirements_lola.txt
```

## Default experiment set

- Q-learning vs Always Cooperate (AllC)
- Q-learning vs Always Defect (AllD)
- Q-learning vs Random
- Q-learning vs Tit For Tat (TFT)
- Q-learning vs Grim Trigger
- Q-learning vs Q-learning

## Folder structure

```text
ipd_project/
  agents/
    heuristic_agents.py
    q_agent.py
  environment/
    ipd_env.py
  experiments/
    run_experiments.py
    benchmark_suite.py
    tune_search.py
    league_training.py
    lola_dice_bridge.py
  analysis/
    plots.py
results/
reports/
```

## Notes

- Payoff matrix defaults to: (C,C)=3, (C,D)=0, (D,C)=5, (D,D)=1
- Supports memory length N with state size 4^N
- Supports action noise (miscommunication)

