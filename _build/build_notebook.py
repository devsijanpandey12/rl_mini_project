"""Build the single self-contained Taxi project notebook."""
from pathlib import Path
import ast
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "taxi_parts"
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))


def code(source, hidden=False):
    cell = nbf.v4.new_code_cell(textwrap.dedent(source).strip())
    if hidden:
        cell.metadata.update({"jupyter": {"source_hidden": True}, "tags": ["hide-input"]})
    cells.append(cell)


core = (PARTS / "core.py").read_text(encoding="utf-8")
nodes = ast.parse(core).body
core_lines = core.splitlines()


def definitions(*names, imports=False):
    selected = []
    for node in nodes:
        if (imports and isinstance(node, (ast.Import, ast.ImportFrom))) or getattr(node, "name", None) in names:
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            selected.append("\n".join(core_lines[start - 1:node.end_lineno]))
    return "\n\n".join(selected)


md(r"""
    # Taxi Pickup and Drop-off Using Q-Learning
    ### Learning efficient service, measuring exploration costs, and testing supplied action rules

    **Reinforcement learning mini-project · video deadline: 16 October 2026**

    **Student:** _Add your name, roll number, and class._

    **Task:** A taxi must navigate a small road grid, collect a waiting passenger, and deliver
    them to their requested destination. It learns through rewards and penalties rather
    than receiving the route in advance.

    **Research question:** How do learning rate, discount factor, and exploration rate affect
    learning speed, delivery success, efficiency, and illegal service attempts? Does supplying
    an action-validity mask change these outcomes?

    This notebook contains the environment implementation, algorithms, tests, experiments,
    graphs, animated demonstration, measured discussion, and presentation preparation.
    It is self-contained: it does not import development files from this project folder.
    All saved graphs and findings are produced by code, not manually invented results.

    **Run:** choose **Python (taxi_rl)**, then **Restart Kernel and Run All Cells**. The full study
    uses 10 seeds and 5,000 episodes per condition. Outputs are saved in `results/` beside
    the notebook. Gymnasium 1.2.3 is used to validate the custom simulator; training uses
    the simulator defined here. No dataset, GPU, or external service is required.
""")

md("""
    ## 1. Assignment alignment and the contribution

    | Teacher's requested component | This project's evidence |
    |---|---|
    | Construct Gridworld and its transitions | Own 5 × 5 Taxi simulator; explicit transition arrays; full comparison with Gymnasium |
    | Q-learning with the TD update | Equation, readable implementation, hand calculation, terminal/cutoff tests |
    | Epsilon-greedy action selection | Uniform exploration and unbiased maximizing ties |
    | Train over multiple episodes | Ten independent seeds, fresh Q-tables, common episode-start sequences |
    | Plot learning and convergence behavior | Reward, success, steps, illegal actions, checkpoint policy quality and Q error |
    | Investigate alpha, gamma, epsilon | Controlled one-factor-at-a-time experiments with equal budgets |
    | Analyze exploration and off-policy control | Frozen greedy versus exploratory evaluation; SARSA comparison |
    | Investigate risk and practical behavior | Illegal service attempts, wall no-ops, truncation, and a separate masking extension |
    | Demonstrate understanding | State reasoning, exact optimal reference, failure accounting, viva answers and video outline |

    **Scope:** Taxi is our chosen adaptation of the sample. It has no cliffs or modeled road
    collisions. Its penalties let us study operational rule violations and exploration costs;
    these results do not reproduce the Cliff Walking safety experiment. SARSA, masking, and
    exact reference routes are extensions to the required Q-learning parameter study.
""")

md("""
    ## 2. Setup and reproducibility

    Environment: **taxi_rl**. Versions and interpreter path are recorded in the run manifest.
    Training and evaluation use separate random generators. Starts and action randomness
    also use separate generators so changing a parameter preserves the episode-start sequence.
""")
code("""
    import os
    import sys
    from pathlib import Path
    # Support Windows kernels launched without first running conda activate.
    _dll_handles = []
    _library_bin = Path(sys.prefix) / "Library" / "bin"
    if os.name == "nt" and _library_bin.is_dir():
        os.environ["PATH"] = str(Path(sys.prefix)) + os.pathsep + str(_library_bin) + os.pathsep + os.environ.get("PATH", "")
        _dll_handles.append(os.add_dll_directory(str(_library_bin)))

    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import gymnasium as gym
    from matplotlib import animation
    from IPython.display import display, Markdown, HTML
    get_ipython().run_line_magic("matplotlib", "inline")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.dpi": 110, "savefig.dpi": 180, "animation.embed_limit": 60})
    pd.set_option("display.max_columns", 25)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("Python:", sys.version.split()[0], "| interpreter:", sys.executable)
    display(pd.DataFrame({"package": ["NumPy", "pandas", "Matplotlib", "Gymnasium"],
                          "version": [np.__version__, pd.__version__, matplotlib.__version__, gym.__version__]}))
""")

md(r"""
    ## 3. Define the task as a Markov decision process

    **Environment:** the standard deterministic, dry, fixed-destination Taxi-v3 road map.
    Its task conventions are documented by [Farama](https://gymnasium.farama.org/environments/toy_text/taxi/).
    Current online documentation describes Taxi-v4; this project pins Gymnasium 1.2.3 and
    checks directly against its installed Taxi-v3 implementation.

    **State:** $(row, column, passenger, destination)$. Passenger is R/G/Y/B or **on board**.
    Position alone is insufficient: at the same square, the correct action depends on whether
    the passenger has been collected and where they must go.

    $$s = ((5\,row + column)\,5 + passenger)\,4 + destination.$$

    - 500 encodings = 25 taxi positions × 5 passenger locations × 4 destinations.
    - 400 valid nonterminal decision states: 300 waiting-passenger and 100 onboard cases.
    - 300 equally likely initial states: passenger is waiting and differs from destination.
    - Four successful-delivery observations are also reachable, then the episode stops.

    | Action | Code | Effect |
    |---|---|---|
    | South / North / East / West | 0 / 1 / 2 / 3 | Move one cell if a road permits it |
    | Pickup | 4 | Collect the passenger when taxi and passenger coincide |
    | Drop-off | 5 | Deliver at the destination, or leave an onboard passenger at another landmark |

    | Outcome | Reward | Episode ends? |
    |---|---|---|
    | Ordinary movement, wall attempt, legal pickup | −1 | No |
    | Illegal pickup or drop-off attempt | −10 | No |
    | Drop an onboard passenger at a wrong landmark | −1 | No; passenger waits there |
    | Successful delivery | +20 | Yes |

    **Important detail:** wrong-landmark drop-off is legal in this benchmark. It is recorded
    separately from penalized illegal service attempts. Walls block movement; they are not
    modeled collisions. We cap unsuccessful episodes at 200 actions and report **truncation**
    separately from successful termination.

    The state is Markov for this deterministic variant: the next outcome depends on the current
    encoded state and action, without needing the trip's history. Each state–action pair has
    one next-state/reward outcome with probability 1.
""")
code(definitions("TaxiEnv", imports=True))

md(r"""
    ## 4. Q-learning, epsilon-greedy exploration, and SARSA

    $$Q(s,a) \leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right].$$

    Alpha controls the size of each correction. Gamma weights delayed rewards. Epsilon is
    the chance of selecting a uniformly random action; otherwise we select a maximizing action.
    Maximizing ties are broken uniformly. Exploration can select an action that is also greedy.

    **Terminal versus cutoff:** a successful delivery updates toward **+20**, with zero continuation.
    A 200-step administrative cutoff still performs the last nonterminal update and bootstraps.
    See [Farama's time-limit explanation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

    **Off-policy:** Q-learning's target assumes the next action is greedy even while the behavior
    explores. **SARSA** instead uses $Q(s',a')$ for the actual next behavior action. The code selects
    that action before the update and executes the same action next.

    **Masking extension:** supplied masks exclude blocked movement and illegal service attempts
    from exploration, exploitation, and the max backup. Legal wrong-landmark drop-off remains
    available. Masking supplies additional rules; it is evaluated separately from the main
    unmasked learner, following the concept in [Farama's masking tutorial](https://gymnasium.farama.org/tutorials/training_agents/action_masking_taxi/).
""")
code(definitions("Config", "TrainingRun", "epsilon_greedy", "td_update"))

md("""
    ### A hand-worked update

    The next cell uses illustrative Q-values, not learned experiment results. For a legal
    pickup with reward −1, old Q=2, next max Q=4, alpha=.3, and gamma=.95:
    target = −1 + .95 × 4 = 2.8; TD error = .8; updated Q = 2 + .3 × .8 = **2.24**.
""")
code("""
    example_env = TaxiEnv()
    s = example_env.encode(0, 0, 0, 3)
    ns, reward, done = example_env.transition(s, 4)
    example_q = np.zeros((500, 6))
    example_q[s, 4] = 2
    example_q[ns] = [1, 4, -2, 0, 0, 0]
    error = td_update(example_q, s, 4, reward, ns, done, alpha=.3, gamma=.95)
    display(pd.DataFrame([{"old_Q": 2, "reward": reward, "next_max_Q": 4,
                           "target": reward + .95 * 4, "TD_error": error, "new_Q": example_q[s, 4]}]))
    assert np.isclose(example_q[s, 4], 2.24)
""")

md("""
    ## 5. Training implementation

    Each condition starts with a new zero Q-table. The simulator returns sampled transitions;
    training performs no model planning or BFS lookups. Direct array lookup makes the simulator
    efficient while preserving the same tested transitions as `step()`.

    Metrics include return, steps, illegal attempts, wall no-ops, wrong-landmark drop-offs,
    success, truncation, epsilon, TD error, Q change, and visited state–action coverage.
    Coverage uses the common 400 × 6 denominator; masked agents deliberately cannot visit
    excluded actions, so that coverage number is not an accuracy ranking between conditions.
""")
code(definitions("train_agent"))

md(r"""
    ## 6. Evaluation and an independent optimal reference

    **Frozen evaluation** never updates Q. Each trained table is evaluated from all 300 valid
    starting situations at epsilon=0 and epsilon=.1. We retain capped failures in all summaries.
    Random tie-breaking can still introduce randomness when epsilon=0.

    **Exact route reference:** breadth-first search (BFS) calculates distances through the actual
    road walls. For a waiting passenger:

    $$T^*(s)=d(taxi,pickup)+1+d(pickup,destination)+1,$$
    $$G^*(s)=21-T^*(s).$$

    The two extra actions are pickup and drop-off. For an onboard passenger, only travel and
    drop-off remain. Taxi has a different optimum for each start, rather than one universal
    route length. Model-based value iteration supplies an independent $Q^*$ and its routes
    are checked against BFS. Neither reference is given to the training agents.

    **Reward-accounting identity:** with T actions, I illegal attempts, and S equal to 1 for
    success and 0 otherwise, recorded return must equal **−T − 9I + 21S**.
""")
code(definitions("_validate_evaluation", "_heuristic_action", "rollout", "evaluate_policy",
                 "value_iteration", "_grid_distances", "shortest_steps"))

md("""
    ## 7. Automated implementation checks

    These checks include every Gymnasium transition and mask, numerical TD targets,
    terminal/cutoff behavior, SARSA action sequencing, reproducibility, and optimal routes
    for every valid start. A missing Gymnasium reference is explicitly reported as SKIP;
    the delivered full run requires all checks to PASS.
""")
code((PARTS / "checks.py").read_text(encoding="utf-8") + "\n\ncheck_results = run_core_checks()\ndisplay(check_results)\nassert (check_results['status'] == 'PASS').all(), 'Complete the reference validation before submission.'\nprint(f'{len(check_results)} check groups passed.')")

md("""
    ## 8. Controlled experimental protocol

    Full budget: **5,000 training episodes × 10 independent seeds** per condition; 200-action
    cap. Seeds: 11, 23, 37, 41, 53, 67, 79, 83, 97, 101. Baseline **alpha=.3, gamma=.95, epsilon=.1**.

    | Study | Values | Other settings |
    |---|---|---|
    | Algorithm comparison | Q-learning / SARSA | Baseline parameters, unmasked |
    | Learning rate | .1, .3, .7 | Q-learning; gamma=.95, epsilon=.1 |
    | Discount | .5, .95, 1 | Q-learning; alpha=.3, epsilon=.1 |
    | Exploration | .01, .1, .3 | Q-learning; alpha=.3, gamma=.95 |
    | Exploration schedule | Constant .3 / linear .3→.01 | Q-learning; decay ends at episode 4,000 |
    | Supplied rule extension | Mask off / mask on | Q-learning, baseline parameters |

    Identical baseline conditions are reused. These are sensitivity studies, not an exhaustive
    parameter search. All comparisons have the same training starts for each seed.

    Curves show trailing **100-episode means**, calculated per seed, then aggregated. Shading
    is **±1 standard error across seed means**, not a 95% confidence interval. Late summaries
    use the final 500 episodes. Scalar evaluation first averages all 300 trips within a trained
    table, then summarizes across the ten tables; 300 trips are not 300 trained agents.

    Every comparison uses common **undiscounted return**, even when training gamma changes.
    Endpoint greedy evaluation also separates policy quality from the cost of training exploration.
""")
code((PARTS / "visuals.py").read_text(encoding="utf-8"), hidden=True)
code((PARTS / "experiment_helpers.py").read_text(encoding="utf-8"), hidden=True)

md("""
    ## 9. Visualize the task and inspect transitions

    Our declared demonstration uses **state 423**: taxi at (4,1), passenger waiting at R,
    destination B. This start and training seed 11 are selected before the experiment plots.
    **Route-map legend:** orange dashed segments mean the passenger is **not on board**;
    teal solid segments mean the passenger **is on board**. These describe passenger status
    during each road move, rather than which algorithm is used. Dark navy lines are walls
    and the map boundary. The purple square marks the waiting passenger; the flag marks
    the destination. Icons in static route maps show the supplied initial state.

    Only road-connected cell-to-cell moves produce segments. Pickup/drop-off, illegal
    service attempts and blocked moves still count as actions but create no line segment.
    Repeated moves can overlap; line length does not tell us the total action count.
""")
code("""
    env = TaxiEnv()
    DEMO_START = env.encode(4, 1, 0, 3)
    assert DEMO_START == 423
    q_reference, reference_iterations = value_iteration(env, gamma=BASE.gamma)
    reference_route = rollout(TaxiEnv(), q_reference, DEMO_START, seed=900_011)
    optimal_steps_all = np.array([shortest_steps(int(s)) for s in EVAL_STARTS])
    OPTIMAL_MEAN_RETURN = float((21 - optimal_steps_all).mean())
    OPTIMAL_MEAN_STEPS = float(optimal_steps_all.mean())
    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    draw_taxi_map(ax, env, DEMO_START, title="Taxi task | exact reference route", path=reference_route)
    fig.subplots_adjust(bottom=.29, top=.88)
    save_figure(fig, "01_task_map_and_reference")
    display(pd.DataFrame([{"state": DEMO_START, "decoded": env.decode(DEMO_START),
                           "optimal_actions": shortest_steps(DEMO_START), "optimal_return": 21-shortest_steps(DEMO_START)},
                          {"state": "All 300 starts (mean)", "decoded": "Uniform initial distribution",
                           "optimal_actions": OPTIMAL_MEAN_STEPS, "optimal_return": OPTIMAL_MEAN_RETURN}]))
    examples = [(env.encode(0,0,0,3),4), (env.encode(0,1,0,3),4),
                (env.encode(0,1,0,3),2), (env.encode(0,0,4,3),5),
                (env.encode(4,3,4,3),5)]
    display(pd.DataFrame([{"state": s, "decoded": env.decode(s), "action": env.action_names[a],
                           "next_state": env.transition(s,a)[0], "reward": env.transition(s,a)[1],
                           "terminated": env.transition(s,a)[2]} for s,a in examples]))
    print("Value-iteration sweeps:", reference_iterations)
""")

md("""
    ## 10. Baseline learning: Q-learning and SARSA

    Hypothesis: both algorithms can learn the service task, while their different targets
    may produce different behavior under continued exploration. We measure these outcomes
    rather than assuming either algorithm must be superior.

    **Learning-curve legend:** teal = Q-learning; orange = SARSA. Here colors identify
    algorithms, whereas the route-map colors identify passenger status. Solid curves are
    trailing 100-episode means averaged across ten independent seeds; the matching shaded
    bands show ±1 standard error. Training exploration remains epsilon = 0.1.
""")
code("""
    q_runs = run_condition(BASE, "Q-learning")
    sarsa_runs = run_condition(replace(BASE, algorithm="sarsa"), "SARSA")
    baseline_group = {"Q-learning": q_runs, "SARSA": sarsa_runs}
    baseline_summary = training_summary(baseline_group)
    save_figure(plot_learning(metric_lists(baseline_group), SMOOTH), "02_baseline_learning")
    display(baseline_summary.round(3))
""")

md("""
    ## 11. Learning-rate experiment: alpha

    Change only alpha. Larger corrections can propagate information faster; small corrections
    may require more experience. Compare early reward, late reward, uncertainty, and final
    greedy policy quality before making a preference claim.
""")
code("""
    alpha_group = {f"alpha={v:g}": run_condition(replace(BASE, alpha=v), f"alpha={v:g}")
                   for v in (.1,.3,.7)}
    save_figure(plot_sweep(metric_lists(alpha_group), "alpha", SMOOTH), "03_alpha_experiment")
    alpha_summary = training_summary(alpha_group)
    display(alpha_summary.round(3))
""")

md("""
    ## 12. Discount-factor experiment: gamma

    Change only gamma. A low discount reduces the influence of delayed delivery rewards.
    Gamma=1 is an undiscounted episodic comparison. Training objectives differ, but the
    plotted return is undiscounted for every condition. A looping policy can still fail at
    gamma=1; the cap and failure records remain necessary.
""")
code("""
    gamma_group = {f"gamma={v:g}": run_condition(replace(BASE, gamma=v), f"gamma={v:g}")
                   for v in (.5,.95,1.)}
    save_figure(plot_sweep(metric_lists(gamma_group), "gamma", SMOOTH), "04_gamma_experiment")
    gamma_summary = training_summary(gamma_group)
    display(gamma_summary.round(3))
""")

md("""
    ## 13. Exploration-rate experiment: epsilon

    Change only constant epsilon. Exploration can discover useful actions and also incur
    illegal attempts or extra travel. Zero initialization is optimistic for many unsuccessful
    actions and ties are random, so low epsilon does not mean the agent never explores.
""")
code("""
    epsilon_group = {f"epsilon={v:g}": run_condition(replace(BASE, epsilon=v), f"epsilon={v:g}")
                     for v in (.01,.1,.3)}
    save_figure(plot_sweep(metric_lists(epsilon_group), "epsilon", SMOOTH), "05_epsilon_experiment")
    epsilon_summary = training_summary(epsilon_group)
    display(epsilon_summary.round(3))
""")

md("""
    ## 14. Extension: reduce exploration during training

    Fixed .3 and linear .3→.01 start at the same exploration probability. The linear schedule
    reaches .01 at episode 4,000 and retains that floor. This is a practical finite-budget
    experiment; a positive exploration floor does not imply greedy-in-the-limit convergence.
""")
code("""
    decay_runs = run_condition(replace(BASE, epsilon=.3, schedule="linear", epsilon_min=.01,
                                       decay_episodes=4000), "Linear epsilon .3 to .01")
    schedule_group = {"Fixed epsilon .3": epsilon_group["epsilon=0.3"], "Linear .3 to .01": decay_runs}
    fig, ax = plt.subplots(figsize=(10,3))
    for label,runs in schedule_group.items():
        ax.plot(runs[0].metrics.episode, runs[0].metrics.epsilon, label=label)
    ax.set(xlabel="Training episode", ylabel="Exploration probability", title="The exploration schedules used")
    ax.legend(frameon=False); ax.grid(alpha=.2)
    save_figure(fig,"06_exploration_schedules")
    save_figure(plot_sweep(metric_lists(schedule_group), "exploration schedule", SMOOTH), "07_schedule_experiment")
    schedule_summary = training_summary(schedule_group)
    display(schedule_summary.round(3))
""")

md("""
    ## 15. Extension: supplied action rules through masking

    Train a second Q-learning agent with the same parameters and starts, but restrict action
    selection and max backups to the mask. This is a comparison with additional supplied
    knowledge, not a claim that the masked learner independently discovered those rules.
    The mask excludes penalized illegal operations and wall no-ops, but allows a legal
    wrong-landmark drop-off. Therefore, it does not enforce efficient delivery by itself.
""")
code("""
    masked_runs = run_condition(replace(BASE, action_masking=True), "Q-learning + mask")
    mask_group = {"Q-learning": q_runs, "Q-learning + mask": masked_runs}
    mask_summary = training_summary(mask_group)
    save_figure(plot_learning(metric_lists(mask_group), SMOOTH), "08_masking_experiment")
    display(mask_summary.round(3))
    assert all((r.metrics.illegal_actions == 0).all() and (r.metrics.wall_hits == 0).all() for r in masked_runs)
""")

md("""
    ## 16. Frozen evaluation: all initial situations, two execution modes

    Evaluate each learned table from all 300 starts at epsilon=0 and .1, with no updates.
    A random reference uses ten independent RNG replicas. The shortest-path reference has
    the full model and is an upper bound, not a third learning algorithm. All policies share
    the same starts and step cap. Means include unsuccessful capped episodes.

    `optimal_delivery` requires success and exactly the BFS minimum action count.
    `excess_steps` is steps minus the per-start minimum and includes capped failures.
""")
code("""
    learned_group = {"Q-learning": q_runs, "SARSA": sarsa_runs, "Q + mask": masked_runs}
    frozen_copies = [r.q.copy() for runs in learned_group.values() for r in runs]
    evaluation = evaluate_group(learned_group)
    assert all(np.array_equal(old,run.q) for old,run in
               zip(frozen_copies,[r for runs in learned_group.values() for r in runs]))
    eval_summary = evaluation_summary(evaluation)
    learned_evaluation = evaluation[evaluation.agent.isin(learned_group)]
    save_figure(plot_evaluation(learned_evaluation), "09_frozen_learned_policy_evaluation")
    display(eval_summary[["agent","mode","seed_replicates","return","return_se","success",
                          "steps","illegal_actions","wall_hits","optimal_delivery","truncated"]].round(3))

    # Plot references separately so the random agent does not obscure learned-policy details.
    compared = pd.concat([eval_summary[eval_summary["mode"] == "Greedy"],
                          eval_summary[eval_summary["mode"] == "Reference"]], ignore_index=True)
    fig, axes = plt.subplots(1,2,figsize=(13,4.6),layout="constrained")
    y = np.arange(len(compared))
    for ax,metric,title in zip(axes,("return","steps"),("Undiscounted return","Actions per episode")):
        bars=ax.barh(y,compared[metric],xerr=compared[metric+"_se"],color=["#236B8E","#D98144","#25867A","#A4A9B1","#8464A5"],capsize=3)
        ax.set_yticks(y,compared.agent); ax.invert_yaxis()
        ax.set_title(title); ax.grid(axis="x",alpha=.2); ax.set_axisbelow(True)
        if metric=="return": ax.set_xscale("symlog",linthresh=30); ax.set_xlabel("Mean return (symlog; linear within ±30)")
        else: ax.set_xlabel("Mean steps, including capped failures")
        ax.margins(x=.25)
        for bar,value,error in zip(bars,compared[metric],compared[metric+"_se"]):
            outer=value+error if value>=0 else value-error
            ax.annotate(f"{value:.2f}",(outer,bar.get_y()+bar.get_height()/2),xytext=(5 if value>=0 else -5,0),
                        textcoords="offset points",va="center",ha="left" if value>=0 else "right",fontsize=9)
    fig.suptitle("Learned greedy policies versus random and planning references | mean ±1 seed SE")
    save_figure(fig,"10_policy_reference_comparison")
""")

md("""
    ### Inspect an unsuccessful frozen policy, if one occurs

    Successful exploratory training does not guarantee that every finite learned table
    produces a successful greedy policy. We inspect the first capped greedy SARSA trip
    in sorted seed/start order, reproduce its exact evaluation random stream, and show
    its final decisions and action values. Repeated wall actions or state cycles explain
    how this particular frozen policy fails. Constant-step-size noise and limited experience
    are possible contributors; this inspection alone does not establish their causal effects.
""")
code("""
    failed_sarsa=evaluation[(evaluation.agent=="SARSA") & (evaluation["mode"]=="Greedy") & evaluation.truncated]
    SARSA_CAPPED_TRIPS=len(failed_sarsa)
    if not failed_sarsa.empty:
        failure=failed_sarsa.sort_values(["train_seed","start_state"]).iloc[0]
        failure_seed=int(failure.train_seed)
        failure_start=int(failure.start_state)
        failure_run=next(r for r in sarsa_runs if r.config.seed==failure_seed)
        start_index=int(np.flatnonzero(EVAL_STARTS==failure_start)[0])
        child_seed=np.random.SeedSequence(100_000+failure_seed).spawn(len(EVAL_STARTS))[start_index]
        failure_route=rollout(TaxiEnv(),failure_run.q,failure_start,epsilon=0,max_steps=MAX_STEPS,seed=child_seed)
        assert not failure_route["success"] and failure_route["steps"]==int(failure.steps)
        assert failure_route["return"]==failure["return"]
        trace=pd.DataFrame([{ "step":i+1,"state":s,"position_and_service_state":env.decode(s),
                              "action":env.action_names[a],"reward":reward,"next_state":ns}
                            for i,(s,a,reward,ns) in enumerate(zip(failure_route["states"][:-1],failure_route["actions"],
                                                                      failure_route["rewards"],failure_route["states"][1:]))])
        print(f"Reproduced capped greedy SARSA trip: seed={failure_seed}, start={failure_start}. "
              f"There were {SARSA_CAPPED_TRIPS} capped trips out of {len(SEEDS)*len(EVAL_STARTS)} greedy SARSA evaluations.")
        display(trace.tail(10))
        inspected_state=failure_route["states"][-2]
        values=failure_run.q[inspected_state]
        display(pd.DataFrame({"action":env.action_names,"estimated_Q":values,
                              "greedy_maximizer":values==values.max(),
                              "next_state":[env.transition(inspected_state,a)[0] for a in range(6)],
                              "immediate_reward":[env.transition(inspected_state,a)[1] for a in range(6)]}))
        trace.to_csv(TABLES/"reproduced_sarsa_failure_trace.csv",index=False)
    else:
        print("No capped greedy SARSA trips in this run.")
""")

md("""
    ## 17. Compare parameter settings under common greedy execution

    Training return combines learning quality with the immediate cost of exploration.
    The next table evaluates every alpha, gamma, and epsilon setting with epsilon=0 over
    all 300 starts. This prevents a low-exploration setting from looking better solely
    because it performs fewer random actions during training.
""")
code("""
    sweep_evaluation = endpoint_sweep_evaluation({"alpha":alpha_group,"gamma":gamma_group,"epsilon":epsilon_group})
    display(sweep_evaluation.round(3))
    fig,axes=plt.subplots(1,3,figsize=(13,4.1),layout="constrained")
    for ax,study in zip(axes,("alpha","gamma","epsilon")):
        frame=sweep_evaluation[sweep_evaluation.study==study]
        ax.bar(frame.condition,frame.optimal_delivery,yerr=frame.optimal_delivery_se,capsize=4,
               color=["#236B8E","#D98144","#25867A"])
        ax.set_title(study+": final greedy route quality")
        ax.set_ylim(0,1.12); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
        ax.set_ylabel("Trips using minimum possible actions"); ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
        for x,value in enumerate(frame.optimal_delivery): ax.text(x,value+.025,f"{value:.1%}",ha="center",fontsize=9)
    fig.suptitle("Same 300 starts and epsilon=0 for every setting | mean ±1 SE across 10 trained seeds")
    save_figure(fig,"11_parameter_endpoint_optimality")
""")

md("""
    ## 18. Convergence behavior at saved checkpoints

    Evaluate Q-learning snapshots over all 300 starts and compare their Q-values with the
    model solution at the same gamma=.95. Q error includes only the 400 valid decision
    states and their six actions. SARSA's exploratory target differs, so it is not scored
    against Q* as though it estimated the same quantity.

    Delivery success, optimal route frequency, and complete Q-table accuracy are distinct.
    Good performance from common starts can precede accurate rarely visited action values.
""")
code("""
    diagnostics=checkpoint_diagnostics(q_runs,q_reference)
    save_figure(plot_checkpoint(diagnostics),"12_checkpoint_convergence")
    display(diagnostics.groupby("episode")[["success","optimal_rate","mean_return","q_mae"]].mean().round(4))
""")

md("""
    ## 19. Inspect the policy and watch a real trip

    A policy map must specify passenger and destination context. The two maps hold the
    destination at B and compare a passenger waiting at R with a passenger on board.
    Labels P/D indicate pickup/drop-off actions; arrows show one maximizing movement action.
    Ties in actual rollouts remain random.

    The demonstration uses the declared start 423 and first training seed 11. The all-start
    tables above are the performance evidence; this one trip is an illustration.

    **Route colors:** orange dashed = passenger not on board; teal solid = passenger on
    board. The same key applies to both panels and the animation. Orange in the random
    panel can include wandering; it does not imply progress toward pickup. Dark navy lines
    are walls/boundary. The plotted segments show road moves, so stationary actions are
    included in the reported action count without adding a segment; repeated moves overlap.
""")
code(r"""
    save_figure(plot_policy_slices(env,q_runs[0].q,passenger=0,destination=3),"13_conditioned_policy_and_values")
    before_route=rollout(TaxiEnv(),np.zeros((500,6)),DEMO_START,max_steps=35,seed=300_011,policy="random")
    learned_route=rollout(TaxiEnv(),q_runs[0].q,DEMO_START,epsilon=0,max_steps=MAX_STEPS,seed=300_011)
    fig,axes=plt.subplots(1,2,figsize=(12,7.4))
    for ax,label,route in zip(axes,("Before: uniform random (35-action cap)","After: trained Q-learning (200-action cap)"),(before_route,learned_route)):
        outcome="delivered" if route["success"] else "not delivered"
        draw_taxi_map(ax,env,DEMO_START,title=f"{label}\n{route['steps']} actions | return {route['return']:.0f} | {outcome}",path=route)
    fig.subplots_adjust(bottom=.29,top=.83,wspace=.23)
    fig.text(.08,.025,"Icons show the common initial state; routes show road moves (stationary actions are not segments). Demo caps differ as labeled.",fontsize=8.3,color="#65788A")
    save_figure(fig,"14_before_and_after_actual_trips")
    display(pd.DataFrame([{ "policy":label, **{k:r[k] for k in
                             ("steps","return","success","illegal_actions","wall_hits","wrong_dropoffs")}}
                          for label,r in (("Before: random",before_route),("After: Q-learning",learned_route),
                                          ("Exact reference",reference_route))]))
    display(pd.DataFrame([{ "step":i+1,"state":s,"decoded":env.decode(s),"action":env.action_names[a],
                           "reward":r,"next_state":ns}
                          for i,(s,a,r,ns) in enumerate(zip(learned_route["states"][:-1],learned_route["actions"],
                                                              learned_route["rewards"],learned_route["states"][1:]))]))
""")
code("""
    taxi_animation=animate_taxi(env,learned_route,interval=320)
    animation_html=taxi_animation.to_jshtml()
    display(HTML(animation_html))
    taxi_animation.save(OUTPUT/"learned_taxi_trip.gif",writer=animation.PillowWriter(fps=3))
    (OUTPUT/"learned_taxi_trip.html").write_text(animation_html,encoding="utf-8")
    plt.close(taxi_animation._fig)
    print("Animation saved:",OUTPUT/"learned_taxi_trip.gif")
""")

md("""
    ## 20. Where is the learned policy efficient?

    These matrices average the fraction of minimum-length deliveries across all 25 taxi
    positions and ten trained seeds for each pickup/destination pair. Diagonal pairs never
    occur as initial states. They expose performance across the task, beyond the demo route.
""")
code("""
    fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout="constrained")
    for ax,agent in zip(axes,("Q-learning","Q + mask")):
        frame=evaluation[(evaluation.agent==agent)&(evaluation["mode"]=="Greedy")].copy()
        frame["pickup"]=[env.decode(int(s))[2] for s in frame.start_state]
        frame["destination"]=[env.decode(int(s))[3] for s in frame.start_state]
        values=np.full((4,4),np.nan)
        grouped=frame.groupby(["pickup","destination"]).optimal_delivery.mean()
        for (p,d),value in grouped.items(): values[p,d]=value
        picture=ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=1,cmap="YlGnBu")
        ax.set_xticks(range(4),env.landmark_names); ax.set_yticks(range(4),env.landmark_names)
        ax.set(xlabel="Destination",ylabel="Passenger pickup",title=agent+": minimum-length trips")
        for p in range(4):
            for d in range(4):
                ax.text(d,p,"n/a" if p==d else f"{values[p,d]:.1%}",ha="center",va="center",
                        color="#496078" if p==d or values[p,d]<.65 else "white",fontsize=11)
    fig.colorbar(picture,ax=axes,label="Fraction of optimal deliveries",shrink=.8)
    save_figure(fig,"15_optimality_across_pickup_destination_pairs")
""")

md("""
    ## 21. Validate the full study and export the evidence

    Raw episodes, configurations, Q-tables, visits, references, package versions, and checks
    are saved. CSV episode files are compressed to keep the project manageable. The notebook
    retains the figures and discussion, so it remains useful as one standalone submission file.
""")
code("""
    experiment_checks=[]
    run_count=0
    baseline_starts={r.config.seed:r.metrics.start_state.to_numpy() for r in q_runs}
    for runs in CONDITIONS.values():
        assert len(runs)==len(SEEDS)
        for run in runs:
            frame=run.metrics
            assert len(frame)==EPISODES and frame.episode.tolist()==list(range(1,EPISODES+1))
            assert np.array_equal(frame.start_state.to_numpy(),baseline_starts[run.config.seed])
            assert (frame["return"] == -frame.steps-9*frame.illegal_actions+21*frame.success.astype(int)).all()
            assert (frame.success ^ frame.truncated).all() and frame.steps.between(1,MAX_STEPS).all()
            assert np.isfinite(run.q).all() and set(CHECKPOINTS).issubset(run.snapshots)
            assert run.visits.sum()==frame.steps.sum()
            run_count+=1
    assert (evaluation["return"] == -evaluation.steps-9*evaluation.illegal_actions+21*evaluation.success.astype(int)).all()
    for _,frame in evaluation.groupby(["agent","mode","train_seed"]):
        assert len(frame)==300 and set(frame.start_state)==set(EVAL_STARTS)
    oracle=evaluation[evaluation.agent=="Shortest-path reference"]
    assert oracle.success.all() and oracle.optimal_delivery.all() and (oracle.illegal_actions==0).all()
    masked=evaluation[evaluation.agent=="Q + mask"]
    assert (masked.illegal_actions==0).all() and (masked.wall_hits==0).all()
    experiment_checks=[
        {"check":f"{run_count} training runs: reward accounting, valid episodes, snapshots, visits", "status":"PASS"},
        {"check":"Identical episode-start sequences across all conditions for each seed", "status":"PASS"},
        {"check":"Every evaluation covers all 300 starts; reward accounting valid", "status":"PASS"},
        {"check":"Evaluation leaves Q unchanged; planning reference reaches BFS optimum", "status":"PASS"},
        {"check":"Masked training/evaluation have no excluded illegal or wall actions", "status":"PASS"}]
    experiment_check_table=pd.DataFrame(experiment_checks)
    display(experiment_check_table)
    experiment_check_table.to_csv(TABLES/"experiment_checks.csv",index=False)
    groups={"baseline":baseline_group,"alpha":alpha_group,"gamma":gamma_group,"epsilon":epsilon_group,
            "schedule":schedule_group,"masking":mask_group}
    manifest=export_evidence(groups,evaluation,diagnostics,sweep_evaluation,check_results)
    display(pd.DataFrame({"item":["Unique conditions","Independent training runs","Training episodes","Main evaluation trips","Training seconds"],
                          "value":[manifest["conditions"],manifest["training_runs"],manifest["training_episodes"],
                                    manifest["evaluation_episodes_main"],round(manifest["total_training_seconds"],2)]}))
""")

md("""
    ## 22. Measured findings and interpretation

    This discussion is calculated from the actual tables. Rankings are descriptive within
    the tested budget and seed list. Small differences relative to seed uncertainty do not
    establish a universal best parameter or a statistically significant advantage.
""")
code(r"""
    def result_row(agent,mode):
        return eval_summary[(eval_summary.agent==agent)&(eval_summary["mode"]==mode)].iloc[0]

    qg,qe=result_row("Q-learning","Greedy"),result_row("Q-learning","Exploratory")
    sg,se=result_row("SARSA","Greedy"),result_row("SARSA","Exploratory")
    mg,me=result_row("Q + mask","Greedy"),result_row("Q + mask","Exploratory")
    bt=baseline_summary.set_index("condition").loc["Q-learning"]
    early_best=alpha_summary.loc[alpha_summary.early_500_return.idxmax()]
    final_diag=diagnostics[diagnostics.episode==EPISODES]
    unmasked_illegal=bt.late_illegal_actions
    fixed,decayed=schedule_summary.iloc[0],schedule_summary.iloc[1]
    findings=[
        f"**Correctness:** all {len(check_results)} implementation check groups and {len(experiment_check_table)} complete-study checks passed. Every one of the 3,000 Taxi transitions and 500 masks matched the installed Gymnasium reference.",
        f"**Learning speed:** Q-learning mean return moved from {bt.early_500_return:.2f} in the first 500 episodes to {bt.late_return:.2f} in the last 500. In the alpha study, {early_best.condition} had the highest first-500 return ({early_best.early_500_return:.2f}); inspect late behavior separately.",
        f"**Exact benchmark:** over all 300 equally weighted initial states, the minimum route averages {OPTIMAL_MEAN_STEPS:.2f} actions and the optimal undiscounted return is {OPTIMAL_MEAN_RETURN:.2f}. The optimum is start-dependent, not one fixed trip length.",
        f"**Final Q-learning policy:** greedy success was {qg.success:.2%}, mean return {qg['return']:.3f} (SE {qg.return_se:.3f}), mean steps {qg.steps:.3f}, and minimum-length delivery frequency {qg.optimal_delivery:.2%}. Truncation was {qg.truncated:.2%}; failures remain included.",
        f"**Exploration cost:** frozen Q-learning at epsilon=.1 returned {qe['return']:.2f} with {qe.illegal_actions:.3f} illegal service attempts and {qe.wall_hits:.3f} wall no-ops per trip. Greedy return was {qg['return']:.2f}; these are different execution policies even though Q is unchanged.",
        f"**SARSA comparison:** greedy return {sg['return']:.2f}, success {sg.success:.2%}; exploratory return {se['return']:.2f} with {se.illegal_actions:.3f} illegal attempts. There were {SARSA_CAPPED_TRIPS} capped greedy trips out of {len(SEEDS)*len(EVAL_STARTS)} evaluations. The failure inspection reproduces one such trip if present; exploratory actions can escape a frozen greedy loop. Its sampled next-action target and finite noisy estimates differ from Q-learning's greedy target. Use these measurements rather than assuming a universal ranking.",
        f"**Masking extension:** unmasked Q-learning made {unmasked_illegal:.3f} illegal attempts per late training episode; masked training made zero by construction. Masked greedy success was {mg.success:.2%} and optimal-route frequency {mg.optimal_delivery:.2%}; masked exploratory return was {me['return']:.2f}. The extra validity rules are supplied knowledge.",
        f"**Exploration schedule:** fixed .3 gave late return {fixed.late_return:.2f} and {fixed.late_illegal_actions:.3f} illegal attempts; decay gave {decayed.late_return:.2f} and {decayed.late_illegal_actions:.3f}. Lower late exploration and the learned policy both affect this comparison.",
        f"**Q-value accuracy:** final mean absolute Q error over 400 valid decision states ×6 actions was {final_diag.q_mae.mean():.4f}; mean greedy optimal-route frequency was {final_diag.optimal_rate.mean():.2%}. Successful routes and complete-table accuracy are distinct measurements."
    ]
    for study in ("alpha","gamma","epsilon"):
        frame=sweep_evaluation[sweep_evaluation.study==study]
        best=frame.loc[frame["return"].idxmax()]
        tied=frame[np.isclose(frame["return"],best["return"],rtol=0,atol=1e-9)]
        labels=", ".join(tied.condition)
        ranking="shared the highest observed mean return" if len(tied)>1 else "had the highest observed mean return"
        findings.append(f"**{study} under common greedy evaluation:** {labels} {ranking} ({best['return']:.3f}); SEs across trained seed means are in the endpoint table. Success ranged {frame.success.min():.1%}–{frame.success.max():.1%}; optimal-route frequency ranged {frame.optimal_delivery.min():.1%}–{frame.optimal_delivery.max():.1%}. This separates learned policy quality from training exploration costs.")
    discussion="\n\n".join("- "+item for item in findings)
    display(Markdown(discussion))
    (OUTPUT/"measured_findings.md").write_text("# Actual experiment findings\n\n"+discussion,encoding="utf-8")
""")

md("""
    ## 23. What the results establish, and what remains limited

    **Off-policy implication:** the Q-learning backup uses a greedy future action, while its
    training behavior may still pick an illegal or inefficient random action. Frozen greedy
    versus exploratory evaluation directly measures this distinction. A learned Q-table can
    therefore support efficient exploitation while continued exploration remains costly.

    **Rule compliance:** Taxi's illegal actions are abstract service-rule violations. The
    simulator does not model speed, pedestrians, road collisions, or physical driving safety.
    Masking excludes some operations by supplied rules, while still allowing legal inefficient
    decisions. Zero masked violations are expected by construction, not a discovered safety theorem.

    **Convergence:** finite curves and Q-error measurements show empirical progress or
    stabilization. Constant alpha, a finite budget, and a positive epsilon floor alone do not
    establish theoretical convergence conditions. See [Watkins and Dayan (1992)](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).

    **Evidence limits:** only ten training seeds and a small set of parameter values are tested.
    One-factor studies miss interactions. Deterministic movement and fixed destinations simplify
    the task. The 200-action cap censors long failures; truncation is reported. The planning
    reference has knowledge the learner lacks. All-start evaluation covers the initial-state
    distribution, but a single exploratory trip per start and trained seed still has sampling noise.

    **Future experiments:** noisy movement, changing destinations, different maps, more training,
    Expected SARSA, or a separate test set of changed environments. These are extensions;
    the current experiment does not establish generalization to them.
""")

md("""
    ## 24. Present the work in a six-minute video

    | Time | Show | Explain |
    |---|---|---|
    | 0:00–0:40 | Task map and state 423 | Taxi, passenger, destination, walls, and what counts as success |
    | 0:40–1:20 | State definition and transitions | Why position alone is insufficient; 500/400/300 counts; wrong-landmark drop-off |
    | 1:20–2:00 | TD equation and hand calculation | Target 2.8, error .8, updated Q=2.24; alpha/gamma/epsilon |
    | 2:00–2:50 | Tests and baseline curves | Full Gym validation; learning progress; ten seeds; what SE means |
    | 2:50–3:40 | Parameter plots and endpoint table | One parameter changed at a time; training versus greedy performance |
    | 3:40–4:30 | Schedule and masking experiments | Decreasing exploration; supplied rules and fair interpretation |
    | 4:30–5:20 | Animation, policy slices, full-start evaluation | Pickup and delivery; context-dependent policy; exact BFS reference |
    | 5:20–6:00 | Measured findings and limitations | Actual numbers, any failures, off-policy costs, and finite convergence limits |

    Add your identity before recording. Practice explaining one transition and the TD update
    without reading the paragraph. Describe the actual findings in section 22, including any
    surprising or unsuccessful outcomes. The code, tests, and experiments are the contribution;
    the animation helps communicate it.
""")

notes=(PARTS/"research.md").read_text(encoding="utf-8")
viva=notes.split("## Viva preparation",1)[1].split("## References",1)[0].strip()
references=notes.split("## References",1)[1].strip()
md("## 25. Viva preparation\n\n"+viva)
md("## 26. References\n\n"+references+"\n\nThe Taxi task originates in Dietterich's formulation. The project applies tabular TD control and experimental analysis to this task; it does not claim to introduce a new RL algorithm.")

notebook=nbf.v4.new_notebook(cells=cells)
notebook.metadata.update({"kernelspec":{"name":"taxi_rl","display_name":"Python (taxi_rl)","language":"python"},
                          "language_info":{"name":"python","version":"3.11"},
                          "title":"Taxi Pickup and Drop-off Using Q-Learning"})
nbf.validate(notebook)
for index, cell in enumerate(cells):
    if cell.cell_type == "code":
        compile(cell.source, f"notebook_cell_{index + 1}", "exec")
target=ROOT/"Taxi_Q_Learning_Mini_Project.ipynb"
nbf.write(notebook,target)
print(f"Built {target.name}: {len(cells)} cells; {sum(c.cell_type=='code' for c in cells)} code cells")
