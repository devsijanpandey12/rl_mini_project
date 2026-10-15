"""Reproducible experiment orchestration, embedded in the standalone notebook."""
from dataclasses import replace, asdict
from time import perf_counter
from pathlib import Path
import json
import importlib.metadata
import platform
import sys

SEEDS = (11, 23, 37, 41, 53, 67, 79, 83, 97, 101)
EPISODES = 5000
MAX_STEPS = 200
SMOOTH = 100
TAIL = 500
CHECKPOINTS = (0, 100, 500, 1000, 2000, 3500, 5000)
OUTPUT = Path.cwd() / "results"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
for folder in (OUTPUT, FIGURES, TABLES):
    folder.mkdir(parents=True, exist_ok=True)

BASE = Config(alpha=.3, gamma=.95, epsilon=.1, episodes=EPISODES,
              max_steps=MAX_STEPS, seed=SEEDS[0])
CONDITIONS = {}
CONDITION_LABELS = {}
RUNTIMES = {}
EVAL_STARTS = np.array(TaxiEnv().initial_states, dtype=int)


def config_key(config):
    return tuple((key, value) for key, value in asdict(config).items() if key != "seed")


def run_condition(config, label):
    key = config_key(config)
    if key not in CONDITIONS:
        started = perf_counter()
        CONDITIONS[key] = [train_agent(TaxiEnv(), replace(config, seed=seed),
                                      snapshot_episodes=CHECKPOINTS) for seed in SEEDS]
        RUNTIMES[key] = perf_counter() - started
        CONDITION_LABELS[key] = label
        print(f"{label}: {len(SEEDS)} independent seeds x {EPISODES:,} episodes "
              f"in {RUNTIMES[key]:.1f} s", flush=True)
    else:
        print(f"{label}: reuse identical condition", flush=True)
    return CONDITIONS[key]


def metric_lists(group):
    return {label: [run.metrics for run in runs] for label, runs in group.items()}


def mean_se(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.


def training_summary(group):
    rows = []
    for label, runs in group.items():
        row = {"condition": label, "seeds": len(runs)}
        for metric in ("return", "steps", "illegal_actions", "wall_hits", "success", "truncated"):
            row[f"late_{metric}"], row[f"late_{metric}_se"] = mean_se(
                [run.metrics[metric].tail(TAIL).mean() for run in runs])
        row["early_500_return"] = np.mean([r.metrics["return"].head(TAIL).mean() for r in runs])
        row["whole_training_return"] = np.mean([r.metrics["return"].mean() for r in runs])
        row["total_illegal_actions"] = np.mean([r.metrics["illegal_actions"].sum() for r in runs])
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(figure, name):
    figure.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.show()


def enrich_evaluation(frame, env):
    frame = frame.copy()
    frame["optimal_steps"] = [shortest_steps(int(s)) for s in frame["start_state"]]
    frame["excess_steps"] = frame["steps"] - frame["optimal_steps"]
    frame["optimal_delivery"] = frame["success"] & (frame["steps"] == frame["optimal_steps"])
    frame["optimal_return"] = 21 - frame["optimal_steps"]
    frame["return_gap"] = frame["optimal_return"] - frame["return"]
    return frame


def evaluate_group(group, include_references=True):
    """One episode at each of ALL 300 valid starts per model and mode."""
    env = TaxiEnv()
    frames = []
    for label, runs in group.items():
        for run in runs:
            for epsilon in (0., .1):
                frame = evaluate_policy(TaxiEnv(), run.q, EVAL_STARTS, epsilon=epsilon,
                                        max_steps=MAX_STEPS, seed=100_000 + run.config.seed + int(epsilon * 10000),
                                        masking=run.config.action_masking)
                frame["agent"] = label
                frame["mode"] = "Greedy" if epsilon == 0 else "Exploratory"
                frame["evaluation_epsilon"] = epsilon
                frame["train_seed"] = run.config.seed
                frame["masking"] = run.config.action_masking
                frames.append(enrich_evaluation(frame, env))
    if include_references:
        for seed in SEEDS:
            frame = evaluate_policy(TaxiEnv(), np.zeros((500, 6)), EVAL_STARTS, epsilon=1.,
                                    max_steps=MAX_STEPS, seed=100_000 + seed, policy="random")
            frame["agent"], frame["mode"], frame["evaluation_epsilon"] = "Random", "Reference", 1.
            frame["train_seed"], frame["masking"] = seed, False
            frames.append(enrich_evaluation(frame, env))
        # A planning reference has the full model. It is an upper bound, not another learner.
        oracle_frame = evaluate_policy(TaxiEnv(), q_reference, EVAL_STARTS, epsilon=0.,
                                       max_steps=MAX_STEPS, seed=909_011)
        oracle_frame["agent"], oracle_frame["mode"] = "Shortest-path reference", "Reference"
        oracle_frame["evaluation_epsilon"], oracle_frame["train_seed"], oracle_frame["masking"] = 0., -1, False
        frames.append(enrich_evaluation(oracle_frame, env))
    return pd.concat(frames, ignore_index=True)


def evaluation_summary(frame):
    metrics = ["return", "steps", "illegal_actions", "wall_hits", "wrong_dropoffs", "success",
               "truncated", "excess_steps", "optimal_delivery", "return_gap"]
    per_seed = frame.groupby(["agent", "mode", "train_seed"], sort=False)[metrics].mean().reset_index()
    rows = []
    for (agent, mode), subset in per_seed.groupby(["agent", "mode"], sort=False):
        row = {"agent": agent, "mode": mode, "seed_replicates": len(subset)}
        for metric in metrics:
            row[metric], row[metric + "_se"] = mean_se(subset[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def checkpoint_diagnostics(runs, q_star):
    env = TaxiEnv()
    rows = []
    for run in runs:
        for episode, q in sorted(run.snapshots.items()):
            frame = evaluate_policy(TaxiEnv(), q, EVAL_STARTS, epsilon=0., max_steps=MAX_STEPS,
                                    seed=200_000 + run.config.seed + episode)
            frame = enrich_evaluation(frame, env)
            rows.append({"condition": "Q-learning", "seed": run.config.seed, "episode": episode,
                         "success": frame["success"].mean(), "optimal_rate": frame["optimal_delivery"].mean(),
                         "mean_return": frame["return"].mean(),
                         "q_mae": np.abs(q[env.decision_states] - q_star[env.decision_states]).mean()})
    return pd.DataFrame(rows)


def endpoint_sweep_evaluation(groups):
    """Compare alpha/gamma/epsilon policies under the SAME greedy evaluation."""
    env = TaxiEnv()
    rows = []
    for study, group in groups.items():
        for label, runs in group.items():
            seed_rows = []
            for run in runs:
                frame = evaluate_policy(TaxiEnv(), run.q, EVAL_STARTS, epsilon=0., max_steps=MAX_STEPS,
                                        seed=400_000 + run.config.seed)
                frame = enrich_evaluation(frame, env)
                seed_rows.append({metric: frame[metric].mean() for metric in
                                  ("return", "steps", "illegal_actions", "success", "optimal_delivery", "truncated")})
            summary = {"study": study, "condition": label}
            for metric in seed_rows[0]:
                summary[metric], summary[metric + "_se"] = mean_se([row[metric] for row in seed_rows])
            rows.append(summary)
    return pd.DataFrame(rows)


def export_evidence(groups, evaluation, diagnostics, sweep_evaluation, checks):
    frames, configs, arrays = [], [], {"optimal_q_gamma_0_95": q_reference}
    for condition_id, (key, runs) in enumerate(CONDITIONS.items()):
        config = asdict(runs[0].config)
        config.pop("seed")
        configs.append({"condition_id": condition_id, "label": CONDITION_LABELS[key], **config,
                        "runtime_seconds": RUNTIMES[key]})
        for run in runs:
            frame = run.metrics.copy()
            frame["condition_id"], frame["seed"] = condition_id, run.config.seed
            frames.append(frame)
            arrays[f"q_condition_{condition_id}_seed_{run.config.seed}"] = run.q
            arrays[f"visits_condition_{condition_id}_seed_{run.config.seed}"] = run.visits
    all_training = pd.concat(frames, ignore_index=True)
    all_training.to_csv(TABLES / "training_episodes.csv.gz", index=False, compression="gzip")
    pd.DataFrame(configs).to_csv(TABLES / "conditions.csv", index=False)
    evaluation.to_csv(TABLES / "evaluation_episodes.csv.gz", index=False, compression="gzip")
    evaluation_summary(evaluation).to_csv(TABLES / "evaluation_summary.csv", index=False)
    env = TaxiEnv()
    states, actions = np.indices((env.n_states, env.n_actions))
    pd.DataFrame({"state": states.ravel(), "action": actions.ravel(),
                  "probability": 1.0, "next_state": env.P_next.ravel(),
                  "reward": env.P_reward.ravel(), "terminated": env.P_done.ravel(),
                  "action_allowed_by_mask": env.action_masks.ravel(),
                  "illegal_service_attempt": env.illegal.ravel(),
                  "wall_noop": env.wall.ravel(), "wrong_landmark_dropoff": env.wrong_dropoff.ravel()
                 }).to_csv(TABLES / "complete_transition_model.csv", index=False)
    reference_rows = [{"state": int(state), "row": env.decode(int(state))[0],
                       "column": env.decode(int(state))[1], "passenger": env.decode(int(state))[2],
                       "destination": env.decode(int(state))[3], "optimal_steps": shortest_steps(int(state)),
                       "optimal_undiscounted_return": 21 - shortest_steps(int(state))}
                      for state in EVAL_STARTS]
    pd.DataFrame(reference_rows).to_csv(TABLES / "exact_reference_all_starts.csv", index=False)
    diagnostics.to_csv(TABLES / "checkpoint_diagnostics.csv", index=False)
    sweep_evaluation.to_csv(TABLES / "parameter_greedy_evaluation.csv", index=False)
    checks.to_csv(TABLES / "implementation_checks.csv", index=False)
    for name, group in groups.items():
        training_summary(group).to_csv(TABLES / f"{name}_training_summary.csv", index=False)
    np.savez_compressed(OUTPUT / "trained_q_and_visits.npz", **arrays)
    packages = {p: importlib.metadata.version(p) for p in
                ("numpy", "pandas", "matplotlib", "pillow", "gymnasium", "ipykernel", "nbformat", "nbclient")}
    manifest = {"project": "Taxi pickup and drop-off with tabular Q-learning", "seeds": list(SEEDS),
                "episodes_per_run": EPISODES, "step_cap": MAX_STEPS, "conditions": len(CONDITIONS),
                "training_runs": len(CONDITIONS) * len(SEEDS), "training_episodes": len(all_training),
                "evaluation_episodes_main": len(evaluation), "checkpoint_evaluation_episodes": len(diagnostics) * len(EVAL_STARTS),
                "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                "packages": packages, "total_training_seconds": sum(RUNTIMES.values()),
                "baseline": asdict(BASE), "all_initial_states_evaluated": len(EVAL_STARTS),
                "uncertainty": "standard error across independently trained seed means",
                "training_policy_model_planning": False, "evaluation_updates": False}
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
