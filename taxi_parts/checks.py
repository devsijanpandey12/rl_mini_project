"""Meaningful correctness checks; external Gym agreement is explicitly optional."""


def run_core_checks():
    results = []

    def passed(check, evidence):
        results.append({"check": check, "status": "PASS", "evidence": evidence})

    env = TaxiEnv()
    assert len(env.initial_states) == 300 and len(env.decision_states) == 400
    for state in range(500):
        assert env.encode(*env.decode(state)) == state
    assert tuple(env.decode(0)) == (0, 0, 0, 0)
    assert tuple(env.decode(499)) == (4, 4, 4, 3)
    passed("State encoding and valid sets", "All 500 roundtrips; 300 initial and 400 decision states")

    try:
        import gymnasium as gym
        reference = gym.make("Taxi-v3").unwrapped
    except ImportError:
        results.append({"check": "Gymnasium exhaustive dynamics and masks", "status": "SKIP",
                        "evidence": "Gymnasium is not installed; custom implementation tests still execute"})
    else:
        assert set(np.flatnonzero(reference.initial_state_distrib)) == set(env.initial_states)
        count = 0
        for state in range(500):
            assert tuple(reference.decode(state)) == env.decode(state)
            assert np.array_equal(env.mask(state), reference.action_mask(state).astype(bool))
            for action in range(6):
                distribution = reference.P[state][action]
                assert len(distribution) == 1 and distribution[0][0] == 1
                _, ns, reward, done = distribution[0]
                assert env.transition(state, action) == (ns, reward, done), (state, action)
                count += 1
        reference.close()
        passed("Gymnasium exhaustive dynamics and masks", f"Gymnasium {gym.__version__}: all {count} transitions and 500 masks match")

    at_r = env.encode(0, 0, 0, 1)
    aboard_at_r = env.encode(0, 0, 4, 1)
    aboard_at_g = env.encode(0, 4, 4, 1)
    assert env.transition(at_r, 4) == (aboard_at_r, -1, False)
    assert env.transition(aboard_at_g, 5) == (env.encode(0, 4, 1, 1), 20, True)
    assert env.transition(aboard_at_r, 5) == (at_r, -1, False)
    assert env.mask(aboard_at_r)[5] and env.wrong_dropoff[aboard_at_r, 5]
    assert env.transition(at_r, 5) == (at_r, -10, False)
    assert env.transition(env.encode(0, 1, 0, 1), 2)[0] == env.encode(0, 1, 0, 1)
    assert env.transition(at_r, 1) == (at_r, -1, False)
    # Gym P does not make delivered-passenger source encodings absorbing.
    delivered = env.encode(0, 4, 1, 1)
    assert env.transition(delivered, 0) == (env.encode(1, 4, 1, 1), -1, False)
    passed("Hand-specified Taxi semantics", "Legal pickup; +20 terminal dropoff; legal −1 wrong-landmark dropoff; −10 illegal; walls; nonabsorbing P sources")

    env.reset(state=at_r)
    env.transition(aboard_at_g, 5)
    assert env.state == at_r
    assert env.step(4) == (aboard_at_r, -1, False) and env.state == aboard_at_r
    assert env.step(5) == (at_r, -1, False)
    seeds = np.random.default_rng(12)
    assert all(env.reset(rng=seeds) in env.initial_states for _ in range(100))
    for state in range(500):
        assert env.mask(state).any()
        assert np.array_equal(env.mask(state), ~(env.illegal[state] | env.wall[state]))
    passed("Stateful simulator and masks", "Pure model queries; step/reset; all 500 masks exclude only illegal actions and no-op movements")

    q = np.zeros((500, 6))
    q[0, 0] = 2
    q[1] = [1, 4, -2, 0, 0, 0]
    delta = td_update(q, 0, 0, -1, 1, False, .5, .9)
    assert np.isclose(delta, .6) and np.isclose(q[0, 0], 2.3)
    q[0, 0] = 2
    delta = td_update(q, 0, 0, -1, 1, False, .5, .9, next_action=2)
    assert np.isclose(delta, -4.8) and np.isclose(q[0, 0], -.4)
    q[1] = 10000
    q[0, 0] = 2
    delta = td_update(q, 0, 0, 20, 1, True, .5, .9)
    assert delta == 18 and q[0, 0] == 11
    passed("TD hand calculations and terminal target", "Q-learning:2→2.3; SARSA:2→−.4; terminal20 target ignores huge next Q")

    q[1] = [100, 4, 3, 200, 300, 400]
    q[0, 0] = 0
    mask = np.array([False, True, True, False, False, False])
    td_update(q, 0, 0, -1, 1, False, 1, .5, next_mask=mask)
    assert q[0, 0] == 1
    rng = np.random.default_rng(2026)
    choices = [epsilon_greedy(q[1], epsilon, rng, mask) for epsilon in (0, 1)
               for _ in range(3000)]
    assert set(choices) == {1, 2} and set(choices[:3000]) == {1}
    qrow = np.array([1., 0., 1., 0., 0., 0.])
    rng = np.random.default_rng(2026)
    ties = np.bincount([epsilon_greedy(qrow, 0, rng) for _ in range(12000)], minlength=6)
    assert ties[[1, 3, 4, 5]].sum() == 0
    assert np.max(np.abs(ties[[0, 2]] / 12000 - .5)) < .025
    rng = np.random.default_rng(2026)
    exploration = np.bincount([epsilon_greedy(qrow, 1, rng) for _ in range(12000)], minlength=6)
    assert np.max(np.abs(exploration / 12000 - 1/6)) < .025
    passed("Exploration, ties, and masked backups", "Uniform unmasked exploration/ties; masked selection and target exclude high-valued forbidden actions")

    invalid = [dict(alpha=0), dict(alpha=2), dict(gamma=-1), dict(epsilon=2),
               dict(episodes=0), dict(max_steps=0), dict(seed=-1), dict(alpha=np.nan),
               dict(schedule="bad"), dict(algorithm="bad"), dict(action_masking=1),
               dict(schedule="linear", epsilon=.1, epsilon_min=.2)]
    for kwargs in invalid:
        try:
            Config(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid configuration accepted: {kwargs}")
    for state, action in [(-1, 0), (500, 0), (0, -1), (0, 6)]:
        try:
            env.transition(state, action)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid state/action accepted")
    passed("Input validation", "12 invalid configs and 4 invalid state/action pairs rejected")

    config = Config(episodes=50, max_steps=40, seed=17)
    a = train_agent(TaxiEnv(), config, snapshot_episodes=(0, 10, 50))
    b = train_agent(TaxiEnv(), config, snapshot_episodes=(0, 10, 50))
    assert np.array_equal(a.q, b.q) and np.array_equal(a.visits, b.visits)
    pd.testing.assert_frame_equal(a.metrics, b.metrics)
    assert set(a.snapshots) == {0, 10, 50}
    assert np.array_equal(a.snapshots[0], np.zeros((500, 6)))
    assert np.array_equal(a.snapshots[50], a.q)
    assert all(not np.shares_memory(snapshot, a.q) for snapshot in a.snapshots.values())
    assert a.visits.sum() == a.metrics["steps"].sum()
    assert (a.metrics["coverage"].diff().dropna() >= 0).all()
    assert np.isclose(a.metrics["coverage"].iloc[-1], np.count_nonzero(a.visits) / 2400)
    changed = train_agent(TaxiEnv(), Config(algorithm="sarsa", epsilon=.8,
                           action_masking=True, episodes=50, max_steps=40, seed=17))
    assert np.array_equal(a.metrics["start_state"], changed.metrics["start_state"])
    assert (changed.metrics["illegal_actions"] == 0).all() and (changed.metrics["wall_hits"] == 0).all()
    passed("Reproducibility, starts, and visit counts", "Identical seeded runs; independent snapshots; paired start sequences across algorithms/masks; exact visitation accounting")

    for run in (a, changed):
        metrics = run.metrics
        expected_return = -metrics["steps"] - 9 * metrics["illegal_actions"] + 21 * metrics["success"].astype(int)
        assert (metrics["return"] == expected_return).all()
        assert ((metrics["success"].astype(int) + metrics["truncated"].astype(int)) == 1).all()
    passed("Training return and outcome accounting", "Return=−steps−9×illegal+21×success; success/truncation complementary")

    # A one-step collector still bootstraps from a previously learned nonterminal
    # state. Manual reference replays exactly the documented independent RNGs.
    tiny_config = Config(alpha=1, gamma=.9, epsilon=1, episodes=10000, max_steps=1, seed=5)
    tiny = train_agent(TaxiEnv(), tiny_config)
    seed_starts, seed_actions = np.random.SeedSequence(5).spawn(2)
    starts_rng, actions_rng = np.random.default_rng(seed_starts), np.random.default_rng(seed_actions)
    manual = np.zeros((500, 6))
    terminal_like = np.zeros((500, 6))
    for _ in range(tiny_config.episodes):
        state = int(starts_rng.choice(env.initial_states))
        actions_rng.random()  # epsilon-greedy first draws its exploration coin.
        action = int(actions_rng.integers(6))
        ns, reward, done = env.transition(state, action)
        manual[state, action] = reward + (0 if done else .9 * manual[ns].max())
        terminal_like[state, action] = reward
    assert np.allclose(tiny.q, manual)
    assert not np.allclose(tiny.q, terminal_like), "Cutoffs incorrectly treated as terminal"
    # An explicit hand calculation supplements the independent replay.
    probe = np.zeros((500, 6)); probe[1] = 3
    td_update(probe, 0, 0, -1, 1, False, 1, .9)
    assert np.isclose(probe[0, 0], 1.7)
    assert tiny.metrics["truncated"].all()
    passed("Administrative cutoff bootstrapping", "10000 one-step episodes match independent replay and differ from terminal-cutoff bug; target includes .9×3")

    # Full-exploration SARSA has a directly reproducible action stream. This
    # checks that the next action used in its target is the action executed.
    sarsa_config = Config(algorithm="sarsa", alpha=.5, gamma=.9, epsilon=1,
                          episodes=10, max_steps=15, seed=73)
    sarsa_run = train_agent(TaxiEnv(), sarsa_config)
    starts_seed, actions_seed = np.random.SeedSequence(73).spawn(2)
    srng, arng = np.random.default_rng(starts_seed), np.random.default_rng(actions_seed)
    manual_sarsa = np.zeros((500, 6))
    for _ in range(sarsa_config.episodes):
        state = int(srng.choice(env.initial_states))
        arng.random()
        action = int(arng.integers(6))
        for _ in range(sarsa_config.max_steps):
            ns, reward, done = env.transition(state, action)
            if not done:
                arng.random()
                next_action = int(arng.integers(6))
            target = reward if done else reward + .9 * manual_sarsa[ns, next_action]
            manual_sarsa[state, action] += .5 * (target - manual_sarsa[state, action])
            if done:
                break
            state, action = ns, next_action
    assert np.array_equal(sarsa_run.q, manual_sarsa)
    passed("SARSA carries its sampled next action", "10 complete seeded episode replays match independent on-policy updates exactly")

    q_star, iterations = value_iteration(TaxiEnv(), gamma=.95)
    frozen = q_star.copy()
    optimal = evaluate_policy(TaxiEnv(), q_star, env.initial_states, seed=2026)
    expected_steps = np.array([shortest_steps(int(state)) for state in env.initial_states])
    assert optimal["success"].all() and (optimal["illegal_actions"] == 0).all()
    assert np.array_equal(optimal["steps"].to_numpy(), expected_steps)
    assert np.array_equal(optimal["return"].to_numpy(), 21 - expected_steps)
    assert np.array_equal(q_star, frozen)
    # At gamma=1 the exact state value is 21 minus shortest episode length.
    undiscounted, _ = value_iteration(TaxiEnv(), gamma=1)
    assert np.allclose(undiscounted[env.initial_states].max(axis=1), 21 - expected_steps)
    sample_route = rollout(TaxiEnv(), q_star, int(env.initial_states[0]), seed=10)
    assert len(sample_route["states"]) == len(sample_route["actions"]) + 1
    assert len(sample_route["rewards"]) == sample_route["steps"]
    passed("Exact BFS reference and frozen evaluation", f"All 300 starts: Q* achieves BFS minimum steps and 21−steps return; VI {iterations} iterations; Q unchanged")

    schedule = Config(schedule="linear", epsilon=.5, epsilon_min=.05, decay_episodes=101)
    assert np.isclose(schedule.epsilon_at(1), .5)
    assert np.isclose(schedule.epsilon_at(51), .275)
    assert np.isclose(schedule.epsilon_at(101), .05)
    assert np.isclose(schedule.epsilon_at(1000), .05)
    passed("Exploration schedule", "Linear .50→.05 across 101 episodes then fixed floor")
    return pd.DataFrame(results)
