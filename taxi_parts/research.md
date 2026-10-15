# Taxi adaptation: research notes and experimental protocol

## Project scope

**Taxi pickup and delivery using tabular Q-learning: learning efficiency, exploration, and operational errors.**

This is a **Taxi adaptation of the teacher's reinforcement-learning sample**. It retains a custom Gridworld, explicit transitions, temporal-difference Q-learning, epsilon-greedy training, parameter studies, learning curves, and measured analysis. It does not reproduce the sample's Cliff Walking safety experiment. Taxi's errors concern dispatch and service rules; they do not model collisions, injuries, traffic, or physical road safety. Do not describe a low illegal-action count as evidence of safe real-world driving.

## Task and primary-source checks

The standard Taxi domain is a 5 × 5 grid with walls and four landmarks. Its observation combines taxi row, taxi column, passenger location (four landmarks or onboard), and destination. The six actions are south, north, east, west, pickup, and dropoff. Storage has 500 encodings: 400 nonterminal decision states, 300 allowed initial states, and four reachable successful terminal observations. A reset chooses uniformly among the initial states. [Farama Foundation, Taxi specification](https://gymnasium.farama.org/environments/toy_text/taxi/).

Ordinary actions cost −1; an illegal pickup/dropoff costs −10; successful delivery gives +20 and terminates. Crucially, an onboard passenger can be dropped at a **wrong landmark** for −1 without termination, leaving the passenger there. This is a legal but usually unhelpful action, not an illegal −10 action. Blocked movement is a −1 self-transition. The supplied mask excludes blocked movement and illegal service attempts but allows wrong-landmark dropoff. [Official Gymnasium 1.2.2 Taxi implementation](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/v1.2.2/gymnasium/envs/toy_text/taxi.py).

The current online documentation describes **Taxi-v4**, including optional rainy movement and fickle passengers. The locally available **Taxi-v3** reference is appropriate for comparison when both options are disabled. Report the local Gymnasium version and environment ID, and exhaustively compare the custom deterministic transition outcomes against that installed reference; a name comparison alone does not establish equivalence. The version change concerns optional stochastic behavior, and this project studies the dry, fixed-destination task. [Current official source and version history](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium/envs/toy_text/taxi.py).

Taxi originates in Dietterich's MAXQ work. Our learner uses a flat Q-table; citing the domain does not mean we implemented hierarchical MAXQ. [Dietterich (2000), original paper](https://arxiv.org/abs/cs/9905014).

## Learning rules and implementation safeguards

Q-learning updates one experienced entry toward `r + gamma * max(Q(next_state))`. A successful delivery must still receive an update, using target **+20 with zero continuation value**. A time limit is an administrative truncation, so the final nonterminal transition still receives its ordinary bootstrapped update. [Farama, handling time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

Use uniform random tie-breaking between maximizing actions. With zero initialization, unvisited actions can look optimistic relative to negative experienced rewards; epsilon is therefore not the only source of early action variation. Keep all six actions available in the required unmasked baseline. If SARSA is included, select the next behavior action before updating and execute that same action on the following step.

The official masking tutorial is useful for the concept of restricting exploration, exploitation, and the next-state maximum to permitted actions. Its displayed training loop conditions the entire update on neither termination nor truncation. **Do not copy that guard:** our own loop must learn from delivery and cutoff transitions. Likewise, when maximizing over a subset, map the subset index back to the original action code. [Official masking tutorial](https://gymnasium.farama.org/tutorials/training_agents/action_masking_taxi/).

## Controlled study

These are our experimental choices, rather than claims that a particular setting is universally best.

Use **10 independent seeds, 5,000 episodes per seed, a fresh zero Q-table per run**, and a common episode cap. A cap of 200 transitions matches the usual Gymnasium Taxi time limit. If the implementation uses another cap, state that explicitly and apply it to all comparisons and evaluations. Draw training initial states uniformly from the same 300-state set. Use a separate initial-state random stream from action exploration so paired seed runs can face identical starting-state sequences despite taking different routes.

| Study | Settings | Kept fixed |
|---|---|---|
| Required Q-learning baseline | α = .3, γ = .95, ε = .1 | Unmasked actions |
| Alpha sensitivity | .1, .3, .7 | Q-learning; γ = .95, ε = .1 |
| Gamma sensitivity | .5, .95, 1 | Q-learning; α = .3, ε = .1 |
| Epsilon sensitivity | .01, .1, .3 | Q-learning; α = .3, γ = .95 |
| Schedule extension | Fixed .3 versus linear .3 → .01 | Q-learning; α = .3, γ = .95 |
| Optional algorithm comparison | Q-learning versus SARSA | Same baseline α, γ, ε; both unmasked |
| Optional masking extension | Masked versus unmasked Q-learning | Same baseline α, γ, ε and initial-state sequence |

For linear decay, use one-based episode indices: .3 at episode 1, .01 at episode 4,000, then hold .01 through episode 5,000. Both compared schedules start at .3. The positive floor keeps random exploration; this finite schedule does not establish greedy-in-the-limit with infinite exploration.

Train identical baseline conditions once and reuse them in sensitivity plots. These one-factor-at-a-time experiments do not estimate interactions. Keep the planned conditions fixed before inspecting their outcomes. Label any later tuning as exploratory, and retain the original baseline evidence.

Hypotheses to evaluate, rather than predetermined conclusions:

- Larger alpha may improve early adaptation but can change variability; rank using measured early and late results.
- Small gamma discounts delayed delivery rewards heavily and may learn different or less useful finite-budget policies.
- More epsilon can improve coverage but produces extra blocked moves and illegal service attempts.
- Decaying epsilon may reduce late operational errors while retaining early exploration.
- Masking may reduce learning cost because it supplies action-validity knowledge; any advantage is not solely learned error avoidance.
- SARSA can respond differently to ongoing exploration, but this Taxi experiment does not imply the same path-risk mechanism as Cliff Walking or universal superiority of either algorithm.

## Measurements and fair evaluation

Record episode return, steps, successful delivery, truncation, illegal actions, blocked moves, legal wrong-landmark dropoffs, epsilon, seed, and initial state. Keep operational error categories separate. If T is transition count, I illegal-action count, and S successful delivery (0 or 1), the return identity is **G = −T − 9I + 21S**. A wrong-landmark dropoff already belongs to the ordinary −1 transitions.

Plot trailing 100-episode seed-level rolling means, then aggregate across seeds. Late summaries use the final 500 episodes; early summaries use the first 500. Shading and error bars may show **±1 standard error across the ten independent seed means**; label them and do not call them 95% confidence intervals. Retain raw episodes. Do not count evaluation episodes from one learned table as independent trained models.

Freeze every final Q-table and evaluate **all 300 allowed initial states**, each with equal weight, at epsilon = 0 and epsilon = .1. Use independent evaluation random streams. For exploratory evaluation, one rollout per initial state is a fixed-budget stratified estimate; additional independent repeats per state reduce Monte Carlo variation if time permits. Greedy tie-breaking can also create randomness while Q-values remain tied. Report the evaluation repeat count, cap, and policy mask setting. This is exhaustive starting-state coverage, not exhaustive enumeration of stochastic action sequences.

Report mean return, delivery rate, illegal attempts, blocked moves, wrong-landmark dropoffs, truncation, and route efficiency. Use the same **undiscounted return** for all gamma settings: their discounted training objectives differ, so discounted totals using different gamma values would not be directly comparable. Include capped failures in overall performance; if successful-only route length is shown, label its conditioning and show the success rate beside it.

For masked agents, distinguish frozen masked execution from an optional unmasked stress test. A masked agent has not learned useful values for actions never permitted during training; do not silently remove its mask and present that as its intended deployment policy. Conversely, a zero illegal-action count while masking is enabled demonstrates enforcement by supplied rules, not the agent independently discovering every rule.

## Independent optimal reference

Compute shortest movement distances by **breadth-first search on the 25-cell wall-respecting graph**, separately from the learned Q-values. For initial state s, the optimum number of transitions is

`T*(s) = distance(taxi, passenger) + 1 pickup + distance(passenger, destination) + 1 dropoff`.

The corresponding undiscounted optimum is **G*(s) = 21 − T*(s)**. This is a per-start reference; Taxi has no single fixed optimum episode length. For a successful learned rollout, report excess steps `T − T*`, return gap `G* − G`, and the fraction attaining the optimal return. Extra illegal attempts lower return by nine more units each, so route length alone is insufficient. Preserve failures separately; a capped failed episode is not a completed short route.

Cross-check BFS distances against symmetry, zero diagonal, wall landmarks, and actual reference rollouts from all initial states. An all-state dynamic-programming reference can additionally validate values for each gamma, but it must use the same reward and terminal conventions. The BFS solution is for validation and comparison, not available to the learner during training.

## Interpretation and limitations

Gamma = 1 is an undiscounted episodic comparison with a finite achievable delivery route from every valid state. It is not automatically safe from cycles: an improper policy can avoid delivery indefinitely, hence the cap and explicit failure reporting. The convergence assumptions for an undiscounted task must be checked separately; a constant learning rate and a finite episode budget do not prove them. [Watkins and Dayan (1992), Q-learning convergence paper](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).

A flattened curve demonstrates finite-budget stabilization, not proof that every Q-entry is optimal. Taxi has 3,000 stored state–action entries, but terminal/unreachable encodings must not inflate any value-error statistic. The study is deterministic and fully observed, with stochastic starting states and action choice. It does not model real passengers, road conditions, traffic laws, or physical hazards. Conclusions apply to these conditions and seeds.

## Viva preparation

**1. What is a Taxi state, and why is taxi position alone insufficient?** A state includes taxi row/column, passenger location or onboard status, and destination. The same taxi position requires different actions depending on whom it is serving and where they must go.

**2. Why are there 500 encodings, 400 decision states, and 300 initial states?** The storage count is 25 × 5 × 4. Valid nonterminal states include 300 waiting-passenger cases and 100 onboard cases. Initial states exclude onboard passengers and passenger-equals-destination cases. Four successful terminal observations are additionally reachable.

**3. Why is the state Markov?** In the dry, fixed-destination task, the encoded state and chosen action determine the next-state and reward distribution without needing earlier history. Rainy/fickle variants are outside this project's scope.

**4. What are the rewards and an easily misunderstood dropoff rule?** Ordinary actions give −1, illegal service attempts −10, and successful delivery +20. Dropping an onboard passenger at a wrong landmark is legal −1 and nonterminal; it is tracked separately.

**5. What does the Q-learning update do?** It moves Q(s,a) toward the observed reward plus gamma times the best estimated next-action value. Alpha controls the size of this correction; it learns from transitions without requiring the BFS answer.

**6. How do terminal and cutoff updates differ?** Successful delivery still updates with target +20 and zero continuation. An administrative time cutoff still updates the last nonterminal transition with its bootstrapped continuation value.

**7. What do gamma and epsilon control?** Gamma weights future rewards in learning. Epsilon selects random exploration. We use a common undiscounted evaluation metric when comparing gamma settings and retain the chosen evaluation epsilon explicitly.

**8. Why randomly break maximizing ties?** Initial Q-values are equal. Always taking the first maximum would add an arbitrary action-order preference. Uniform tie-breaking treats equally valued actions symmetrically.

**9. Why evaluate all 300 initial states and multiple training seeds?** Uniform state coverage checks every allowed starting situation. Independent trained seeds show learning variability. These are different sources of evidence; 300 evaluations of one table are not 300 trained agents.

**10. How is the optimal reference calculated?** BFS supplies wall-respecting distances. Optimal steps equal taxi-to-passenger distance, pickup, passenger-to-destination distance, and dropoff; optimal undiscounted return is 21 minus those steps.

**11. Is action masking learned safety?** It is supplied action-validity knowledge restricting selection and backups. It can enforce some operational rules, but it is neither independent discovery of those rules nor evidence of physical driving safety.

**12. Does a stable curve or gamma=1 prove convergence?** No. The run is finite with constant alpha. Gamma=1 requires appropriate episodic/properness assumptions; bad policies may loop. We report empirical performance, truncation, and the conditions tested.

## References

1. [Farama Taxi specification](https://gymnasium.farama.org/environments/toy_text/taxi/).
2. [Gymnasium 1.2.2 Taxi-v3 source](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/v1.2.2/gymnasium/envs/toy_text/taxi.py).
3. [Current Taxi-v4 source](https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium/envs/toy_text/taxi.py).
4. [Dietterich (2000), Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition](https://arxiv.org/abs/cs/9905014).
5. [Farama action-masking tutorial](https://gymnasium.farama.org/tutorials/training_agents/action_masking_taxi/), used for the masking concept, with the update correction described above.
6. [Farama handling time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
7. [Watkins and Dayan (1992), Q-learning](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).
