# Offline SOTA Progress Snapshot (March 14, 2026)

## Completed

1. Multi-seed benchmark harness with `n=10` seeds and `mean/std/95% CI`
   - `ipd_project/experiments/benchmark_suite.py`
2. Opponent-aware learner integrated
   - `learner=oaq` in `run_experiments.py`
3. Full robustness run at noise `0%, 1%, 5%, 10%` for both `q` and tuned `oaq`
4. League/opponent-mixture training module
   - `ipd_project/experiments/league_training.py`
5. LOLA/DiCE integration bridge started and executed
   - `ipd_project/experiments/lola_dice_bridge.py`

## Key Result: Tuned `oaq` vs Tuned `q` (10 seeds, full noise sweep)

Average deltas (`oaq - q`):

- Balanced score: `+9.3183`
- `Q_vs_Q` tail cooperation: `+0.1454`
- Population tail cooperation: `+0.0820`

By noise:

- `0.00`: score `+10.3099`, QvQ coop `+0.1648`
- `0.01`: score `+9.7996`, QvQ coop `+0.1556`
- `0.05`: score `+8.8872`, QvQ coop `+0.1357`
- `0.10`: score `+8.2764`, QvQ coop `+0.1256`

## Output Files

- Q baseline full run summary:
  - `results/sota_full_q/latest_benchmark_summary.csv`
- Tuned OAQ full run summary:
  - `results/sota_full_oaq_tuned/latest_benchmark_summary.csv`
- League training output:
  - `results/league_runs_oaq_tuned/latest_league_results.json`
- LOLA bridge output (sample run):
  - `results/lola_bridge_runs/latest_lola_dice_om.json`

## Caveat

This is strong progress, but not yet a formal SOTA claim against published LOLA/POLA/LOQA benchmarks with identical protocols.  
Next step is standardized cross-framework benchmarking (OpenSpiel/Axelrod + LOLA/POLA baselines under matched settings).
