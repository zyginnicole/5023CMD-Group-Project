# Drone swarm SAR: MAPPO tactical policy (Task 2) - v3

Core ML component: parameter-shared MAPPO policy (centralised critic, decentralised execution) choosing
each drone's action from {advance, hold, reroute, verify, return}. A fuzzy safety filter and a loop-trap
guard sit between the policy and the flight controller; an approval gate shows the supervisor a multi-step
plan card (with feature-attribution drivers) before high-risk actions.

## Run (from the project root)
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v                      # sanity tests (Bayes, fuzzy rules, GAE, loop guard, gate)
for s in 0 1 2; do
  python src/train.py    --config configs/agent.yaml --run_dir runs/seed$s --seed $s
  python src/evaluate.py --config configs/agent.yaml --run_dir runs/seed$s --seed $s
done
python src/evaluate.py --config configs/agent.yaml --run_dir runs/seed0 --seed 0 --xai-demo   # execution.log + audit.jsonl
python src/plan_card_demo.py --config configs/agent.yaml --run_dir runs/seed0 --seed 0        # plan_card.txt/.png
python src/collision_attribution.py --config configs/agent.yaml --run_dir runs/seed0 --seed 0
python src/aggregate.py runs/seed0 runs/seed1 runs/seed2                                      # mean +- SD, generalisation gap
python src/deploy.py   --config configs/agent.yaml --run_dir runs/seed0                       # TorchScript export + latency
python src/viva_inspect.py --config configs/agent.yaml --run_dir runs/seed0                   # inner workings printout
# rationale evidence (long; resumable; pick groups/seeds to fit your time budget)
python src/experiments.py sweep      --config configs/agent.yaml --out runs/sweep --groups lr,epochs,minibatch
python src/experiments.py sweep      --config configs/agent.yaml --out runs/sweep --groups base,pool,reward,model,gamma
python src/experiments.py forgetting --config configs/agent.yaml --out runs/forgetting
python src/experiments.py attention  --config configs/agent.yaml --run_dir runs/seed0
```
Record your own wall-clock times (timing.json, deployment_report.json); never quote an unmeasured figure.
Regenerate every figure, log and table from one final run of the final config.

## Task 2 requirement -> evidence
| Requirement | File / output |
|---|---|
| Choice of model; obs -> action mapping | `env.py` (`observe`, `obs_groups`), `mappo.py`; `viva_inspect.py`; MAPPO vs IPPO vs heuristic/random: `experiments.py sweep --groups model`, `results.json` |
| Learning rate, epochs, batch size | `configs/agent.yaml`; `experiments.py sweep` (groups lr, epochs, minibatch, gamma) -> `sweep_table.txt`, `fig_sweep.png`; KL / clip fraction in `fig_training_curves.png` |
| Reward structure | `reward:` block; `sweep --groups reward` (each term zeroed) |
| Overfitting | train vs held-out layouts, noise shift (`results.json`); gap vs layout-pool size (`sweep --groups pool`); `aggregate.py` gap line |
| Catastrophic forgetting | `experiments.py forgetting` (task A -> task B, with/without replay) |
| Loop traps | `LoopGuard` (`safety.py`); `loops` metric; `ablation_no_loop_guard` |
| Safety rules (fuzzy) | `FuzzySafetyFilter` docstring, `tests/test_core.py` |
| XAI plan before high-risk actions | `xai.py` plan card: proposal, confidence, alternatives, filter verdict, look-ahead, gradient x input drivers; `plan_card.txt/.png` |
| Automation bias | decoy plans, `audit.jsonl`, `automation_bias_rate()`; `experiments.py attention` (mission cost of over-trust, operator workload) |
| Deployment | `deployment:` block, `deploy.py` -> `deployment_report.json`; controller fallback to the rule-based policy on inference failure (evaluation mode) |
| Execution logs / screenshots | `execution.log`, `audit.jsonl`, `train_log.jsonl`, `config_used.yaml`, `fig_*.png` |

## Changes in v3 (all results must be regenerated)
- Dead drones are masked out of the PPO loss (previously they contributed gradient with no consequence).
- `mappo.critic: local` gives an IPPO ablation; `value()` / buffers handle per-drone values.
- Plan card now shows gradient x input drivers and policy entropy; look-ahead is labelled as run on the simulator copy.
- `train.py` refactored (`train_policy`, `train_phase`); every run saves `config_used.yaml`.
- New: `experiments.py`, `deploy.py`, `viva_inspect.py`, `tests/`.

## Before you submit
- Every number in `configs/agent.yaml` is a starting point. Replace it with a value your runs justify, and cite the evidence.
- The grid world and goal allocator stub (`assign_goals`) stand in for the Task 1 environment and LLM allocator. Say so.
- Supervisor results are simulated (`ScriptedSupervisor`), not human-study data; decoy approval is ~ 1 - attention by construction.
- Call it MAPPO only because a centralised critic is implemented. The critic uses privileged ground truth during training only.
- Horizon truncation is treated as terminal in GAE. No recurrent state is used (Bayesian belief summarises history).
- Gradient x input is a first-order local explanation, not a causal one. Softmax confidence is not calibrated.
- Report mean +- SD over seeds; with 3 seeds this is descriptive, not a significance test.
- Declare AI-tool use in the submission.
