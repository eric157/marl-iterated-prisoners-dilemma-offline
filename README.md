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
  analysis/
    plots.py
results/
reports/
```

## Notes

- Payoff matrix defaults to: (C,C)=3, (C,D)=0, (D,C)=5, (D,D)=1
- Supports memory length N with state size 4^N
- Supports action noise (miscommunication)

