# Drone swarm SAR: MAPPO tactical policy (Task 2)

Core ML component: a parameter-shared MAPPO policy (centralised critic, decentralised
execution) choosing each drone's action from {advance, hold, reroute, verify, return}.
A fuzzy safety filter and a loop-trap guard sit between the policy and the flight
controller; an approval gate shows the supervisor a multi-step plan before high-risk actions.

## Run
```bash
pip install -r requirements.txt
for s in 0 1 2; do
  python src/train.py    --config configs/agent.yaml --run_dir runs/seed$s --seed $s
  python src/evaluate.py --config configs/agent.yaml --run_dir runs/seed$s --seed $s
done
python src/evaluate.py --config configs/agent.yaml --run_dir runs/seed0 --seed 0 --xai-demo   # logs + audit
python src/aggregate.py runs/seed0 runs/seed1 runs/seed2                                      # mean +- SD over seeds
```
Record your own wall-clock time per run; do not quote a figure you have not measured.
Regenerate every figure, log and table from one final run of the final config.

## Task 2 requirement -> where it lives
| Requirement | File / output |
|---|---|
| Choice of model, obs -> action mapping | `src/env.py` (`observe`, Table 1), `src/mappo.py` |
| Training and hyperparameter rationale | `configs/agent.yaml` (`mappo`, `reward`, `train`) |
| Overfitting risks and evidence | train vs held-out layouts, noise shift, ablations: `runs/*/results.json`, `fig_training_curves.png` |
| Loop traps | `LoopGuard` in `src/safety.py`; `loops` metric (counted passively even with the guard off); `ablation_no_loop_guard` |
| Safety rules (fuzzy) | `FuzzySafetyFilter` docstring in `src/safety.py` |
| XAI plan before high-risk actions | `src/xai.py` (`build_plan`, `render_text`, `render_flow`) |
| Automation bias safeguards | decoy plans, approval records, `audit.jsonl`, `automation_bias_rate()` |
| Execution logs | `runs/*/execution.log` (JSON lines), `runs/*/audit.jsonl` |
| Evaluation screenshots | `fig_training_curves.png`, `fig_eval_comparison.png`, `fig_action_distribution.png` |

## Before you submit
- Every number in `configs/agent.yaml` is a starting point. Replace it with a value your runs justify, and cite the evidence.
- The grid world and the goal-allocator stub (`assign_goals`) stand in for the Task 1 environment and LLM allocator. Say so in the report.
- Supervisor results are simulated (`ScriptedSupervisor`), not human-study data.
- Call it MAPPO only because a centralised critic is implemented. The critic uses privileged ground truth during training only.
- Horizon truncation is treated as terminal in GAE. Mention this if asked.
- No recurrent hidden state is used (the Bayesian belief already summarises history). Treat a GRU as future work.
- Declare AI-tool use in the submission.
- Report mean +- SD over seeds (`aggregate.py`); with 3 seeds this is descriptive, not a significance test.
- The automation-bias rate mirrors `supervisor_attention` (expected about 1 - attention); present it as a demonstration of the measurement.
- Set `mappo.value_clip: false` for one run if you want to test whether critic clipping limits value learning; otherwise state it as a limitation.
