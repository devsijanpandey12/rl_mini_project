"""A transparent, deterministic Taxi model and reproducible tabular control."""
from dataclasses import dataclass
from collections import deque
from functools import lru_cache
import numbers
import numpy as np
import pandas as pd


class TaxiEnv:
    """Canonical dry Taxi-v3, implemented without Gymnasium.

    A state encodes taxi row, taxi column, passenger location (0..3 or 4=aboard),
    and destination (0..3). Source-state transitions match Gymnasium even for
    encodings that represent an already delivered passenger. Episodes stop on
    a successful dropoff transition; those encodings are not made absorbing.
    """
    n_states, n_actions, rows, cols = 500, 6, 5, 5
    landmarks = ((0, 0), (0, 4), (4, 0), (4, 3))
    landmark_names = ("R", "G", "Y", "B")
    MAP = ("+---------+", "|R: | : :G|", "| : | : : |", "| : : : : |",
           "| | : | : |", "|Y| : |B: |", "+---------+")
    action_names = ("south", "north", "east", "west", "pickup", "dropoff")

    def __init__(self):
        self.initial_states = np.array([s for s in range(500)
            if self.decode(s)[2] < 4 and self.decode(s)[2] != self.decode(s)[3]], dtype=int)
        self.decision_states = np.array([s for s in range(500)
            if self.decode(s)[2] != self.decode(s)[3]], dtype=int)
        self.P_next = np.empty((500, 6), dtype=np.int64)
        self.P_reward = np.empty((500, 6), dtype=float)
        self.P_done = np.empty((500, 6), dtype=bool)
        self.action_masks = np.zeros((500, 6), dtype=bool)
        self.wall = np.zeros((500, 6), dtype=bool)
        self.wrong_dropoff = np.zeros((500, 6), dtype=bool)
        for state in range(500):
            row, col, passenger, destination = self.decode(state)
            for action in range(6):
                nr, nc, npass = row, col, passenger
                reward, done = -1.0, False
                if action == 0:
                    nr = min(row + 1, 4)
                elif action == 1:
                    nr = max(row - 1, 0)
                elif action == 2:
                    if self.MAP[row + 1][2 * col + 2] == ":":
                        nc = min(col + 1, 4)
                elif action == 3:
                    if self.MAP[row + 1][2 * col] == ":":
                        nc = max(col - 1, 0)
                elif action == 4:
                    if passenger < 4 and (row, col) == self.landmarks[passenger]:
                        npass = 4
                    else:
                        reward = -10.0
                elif passenger == 4 and (row, col) == self.landmarks[destination]:
                    npass, reward, done = destination, 20.0, True
                elif passenger == 4 and (row, col) in self.landmarks:
                    npass = self.landmarks.index((row, col))
                    self.wrong_dropoff[state, action] = True
                else:
                    reward = -10.0
                next_state = self.encode(nr, nc, npass, destination)
                self.P_next[state, action] = next_state
                self.P_reward[state, action] = reward
                self.P_done[state, action] = done
                self.wall[state, action] = action < 4 and next_state == state
                # Gym's mask excludes no-op movements and penalized operations.
                # A legal wrong-landmark dropoff remains available.
                self.action_masks[state, action] = reward != -10 and not self.wall[state, action]
        self.illegal = self.P_reward == -10
        self.next_state, self.reward, self.terminated = self.P_next, self.P_reward, self.P_done
        self._rng = np.random.default_rng(0)
        self.state = int(self.initial_states[0])

    @staticmethod
    def encode(row, col, passenger, destination):
        for name, value, bound in (("row", row, 5), ("col", col, 5),
                                   ("passenger", passenger, 5), ("destination", destination, 4)):
            if not isinstance(value, numbers.Integral) or not 0 <= value < bound:
                raise ValueError(f"{name} must be an integer in [0, {bound - 1}]")
        return int(((row * 5 + col) * 5 + passenger) * 4 + destination)

    @staticmethod
    def decode(state):
        if not isinstance(state, numbers.Integral) or not 0 <= state < 500:
            raise ValueError("state must be an integer in [0, 499]")
        state = int(state)
        destination, state = state % 4, state // 4
        passenger, position = state % 5, state // 5
        return position // 5, position % 5, passenger, destination

    def transition(self, state, action):
        self.decode(state)
        if not isinstance(action, numbers.Integral) or not 0 <= action < 6:
            raise ValueError("action must be an integer in [0, 5]")
        return (int(self.P_next[state, action]), float(self.P_reward[state, action]),
                bool(self.P_done[state, action]))

    def mask(self, state):
        self.decode(state)
        return self.action_masks[state].copy()

    def reset(self, rng=None, state=None):
        if state is None:
            generator = self._rng if rng is None else rng
            self.state = int(generator.choice(self.initial_states))
        else:
            self.decode(state)
            self.state = int(state)
        return self.state

    def step(self, action):
        next_state, reward, done = self.transition(self.state, action)
        self.state = next_state
        return next_state, reward, done


@dataclass(frozen=True)
class Config:
    alpha: float = 0.3
    gamma: float = 0.95
    epsilon: float = 0.1
    episodes: int = 5000
    max_steps: int = 200
    seed: int = 0
    schedule: str = "constant"
    epsilon_min: float = 0.01
    decay_episodes: int = 4000
    action_masking: bool = False
    algorithm: str = "q_learning"

    def __post_init__(self):
        if self.algorithm not in ("q_learning", "sarsa"):
            raise ValueError("algorithm must be 'q_learning' or 'sarsa'")
        if self.schedule not in ("constant", "linear"):
            raise ValueError("schedule must be 'constant' or 'linear'")
        for name in ("alpha", "gamma", "epsilon", "epsilon_min"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not 0 <= self.epsilon <= 1 or not 0 <= self.epsilon_min <= 1:
            raise ValueError("epsilon and epsilon_min must be in [0, 1]")
        if self.schedule == "linear" and self.epsilon_min > self.epsilon:
            raise ValueError("linear decay requires epsilon_min <= epsilon")
        for name in ("episodes", "max_steps", "decay_episodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, numbers.Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.action_masking, bool):
            raise ValueError("action_masking must be a bool")

    def epsilon_at(self, episode):
        if self.schedule == "constant":
            return float(self.epsilon)
        fraction = min(1.0, max(0.0, (episode - 1) / max(1, self.decay_episodes - 1)))
        return float(self.epsilon + fraction * (self.epsilon_min - self.epsilon))


@dataclass
class TrainingRun:
    q: np.ndarray
    metrics: pd.DataFrame
    snapshots: dict
    visits: np.ndarray
    config: Config


def epsilon_greedy(q_row, epsilon, rng, mask=None):
    """Uniform random exploration and uniform tie-breaking over allowed actions."""
    if mask is not None and not np.any(mask):
        raise ValueError("action mask must allow at least one action")
    if rng.random() < epsilon:
        if mask is None:
            return int(rng.integers(len(q_row)))
        allowed = np.flatnonzero(mask)
        return int(allowed[rng.integers(len(allowed))])
    best_value = q_row.max() if mask is None else q_row[mask].max()
    best = np.flatnonzero(q_row == best_value if mask is None else (q_row == best_value) & mask)
    return int(best[0]) if len(best) == 1 else int(best[rng.integers(len(best))])


def td_update(q, s, a, r, ns, done, alpha, gamma, next_action=None, next_mask=None):
    """Apply Q-learning's max target or SARSA's actual-next-action target."""
    if done:
        target = float(r)
    elif next_action is not None:
        target = r + gamma * q[ns, next_action]
    else:
        if next_mask is not None and not np.any(next_mask):
            raise ValueError("next-state mask must allow at least one action")
        continuation = q[ns].max() if next_mask is None else q[ns, next_mask].max()
        target = r + gamma * continuation
    delta = float(target - q[s, a])
    q[s, a] += alpha * delta
    return delta


def train_agent(env, config, snapshot_episodes=()):
    """Train from zero, with separate RNGs for starts and action decisions.

    This preserves identical episode-start sequences across parameter settings.
    Administrative truncations bootstrap normally; true delivery transitions do
    not. Coverage is visited decision-state/action pairs divided by 400*6, for
    a common denominator across masked and unmasked experiments.
    """
    start_seed, action_seed = np.random.SeedSequence(config.seed).spawn(2)
    start_rng, action_rng = np.random.default_rng(start_seed), np.random.default_rng(action_seed)
    q = np.zeros((env.n_states, env.n_actions), dtype=float)
    visits = np.zeros_like(q, dtype=np.int64)
    requested = set(snapshot_episodes)
    snapshots = {0: q.copy()} if 0 in requested else {}
    records, visited_pairs = [], 0
    masked, sarsa = config.action_masking, config.algorithm == "sarsa"
    masks, next_table = env.action_masks, env.P_next
    reward_table, done_table = env.P_reward, env.P_done
    illegal_table, wall_table, wrong_table = env.illegal, env.wall, env.wrong_dropoff
    choose, update = epsilon_greedy, td_update
    for episode in range(1, config.episodes + 1):
        epsilon = config.epsilon_at(episode)
        state = env.reset(rng=start_rng)
        start_state = state
        q_before = q.copy()
        total_return, abs_td = 0.0, 0.0
        illegal_actions, wall_hits, wrong_dropoffs = 0, 0, 0
        done = False
        if sarsa:
            action = choose(q[state], epsilon, action_rng, masks[state] if masked else None)
        for step in range(1, config.max_steps + 1):
            if not sarsa:
                action = choose(q[state], epsilon, action_rng, masks[state] if masked else None)
            ns = int(next_table[state, action])
            reward, done = reward_table[state, action], bool(done_table[state, action])
            next_mask = masks[ns] if masked else None
            next_action = choose(q[ns], epsilon, action_rng, next_mask) if sarsa and not done else None
            if visits[state, action] == 0:
                visited_pairs += 1
            visits[state, action] += 1
            abs_td += abs(update(q, state, action, reward, ns, done,
                                 config.alpha, config.gamma, next_action, next_mask))
            total_return += reward
            illegal_actions += int(illegal_table[state, action])
            wall_hits += int(wall_table[state, action])
            wrong_dropoffs += int(wrong_table[state, action])
            state = ns
            if done:
                break
            if sarsa:
                action = next_action
        env.state = state
        records.append((episode, start_state, total_return, step, illegal_actions,
                        wall_hits, wrong_dropoffs, done, not done, epsilon,
                        abs_td / step, float(np.max(np.abs(q - q_before))),
                        visited_pairs / (len(env.decision_states) * env.n_actions)))
        if episode in requested:
            snapshots[episode] = q.copy()
    metrics = pd.DataFrame(records, columns=["episode", "start_state", "return", "steps",
        "illegal_actions", "wall_hits", "wrong_dropoffs", "success", "truncated", "epsilon",
        "mean_abs_td", "q_max_change", "coverage"])
    return TrainingRun(q=q, metrics=metrics, snapshots=snapshots, visits=visits, config=config)


def _validate_evaluation(env, q, start_state, epsilon, max_steps, masking, policy):
    env.decode(start_state)
    if np.shape(q) != (500, 6) or not np.all(np.isfinite(q)):
        raise ValueError("q must be a finite (500, 6) array")
    if not isinstance(epsilon, numbers.Real) or not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if not isinstance(max_steps, numbers.Integral) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not isinstance(masking, bool):
        raise ValueError("masking must be a bool")
    if policy not in ("learned", "random", "heuristic"):
        raise ValueError("policy must be 'learned', 'random', or 'heuristic'")


def _heuristic_action(env, state, rng):
    """A local Manhattan rule, not a planner; walls can make it inefficient."""
    row, col, passenger, destination = env.decode(state)
    target = env.landmarks[destination if passenger == 4 else passenger]
    if (row, col) == target:
        return 5 if passenger == 4 else 4
    tr, tc = target
    directions = []
    if row < tr:
        directions.append(0)
    elif row > tr:
        directions.append(1)
    if col < tc:
        directions.append(2)
    elif col > tc:
        directions.append(3)
    allowed = [a for a in directions if env.action_masks[state, a]]
    if allowed:
        return int(allowed[rng.integers(len(allowed))])
    # A blocked local goal direction triggers a random legal movement.
    movements = np.flatnonzero(env.action_masks[state, :4])
    return int(movements[rng.integers(len(movements))])


def rollout(env, q, start_state, epsilon=0, max_steps=200, seed=0, masking=False,
            policy="learned"):
    """Record one frozen-policy delivery attempt from a specified state."""
    _validate_evaluation(env, q, start_state, epsilon, max_steps, masking, policy)
    rng = np.random.default_rng(seed)
    state = env.reset(state=start_state)
    states, actions, rewards = [state], [], []
    illegal_actions, wall_hits, wrong_dropoffs, done = 0, 0, 0, False
    for _ in range(max_steps):
        if policy == "heuristic":
            action = _heuristic_action(env, state, rng)
        else:
            action = epsilon_greedy(q[state], 1 if policy == "random" else epsilon,
                                    rng, env.action_masks[state] if masking else None)
        illegal_actions += int(env.illegal[state, action])
        wall_hits += int(env.wall[state, action])
        wrong_dropoffs += int(env.wrong_dropoff[state, action])
        state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        if done:
            break
    return {"start_state": int(start_state), "states": states, "actions": actions,
            "rewards": rewards, "return": float(sum(rewards)), "steps": len(actions),
            "illegal_actions": illegal_actions, "wall_hits": wall_hits,
            "wrong_dropoffs": wrong_dropoffs, "success": done, "truncated": not done}


def evaluate_policy(env, q, start_states, epsilon=0, max_steps=200, seed=0,
                    masking=False, policy="learned"):
    """Frozen evaluation over explicit start states with independent RNG streams."""
    starts = list(start_states)
    if not starts:
        raise ValueError("start_states cannot be empty")
    children = np.random.SeedSequence(seed).spawn(len(starts))
    records = []
    for episode, (state, child_seed) in enumerate(zip(starts, children), start=1):
        result = rollout(env, q, start_state=state, epsilon=epsilon, max_steps=max_steps,
                         seed=child_seed, masking=masking, policy=policy)
        records.append({"episode": episode, **{key: value for key, value in result.items()
                        if key not in ("states", "actions", "rewards")}})
    return pd.DataFrame(records)


def value_iteration(env, gamma=.95, tolerance=1e-10, max_iterations=10000):
    """Independent model-based Q* reference; training never uses this solution."""
    if not isinstance(gamma, numbers.Real) or not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive")
    values = np.zeros(env.n_states)
    active = env.decision_states
    for iteration in range(1, max_iterations + 1):
        candidates = env.P_reward + gamma * values[env.P_next] * (~env.P_done)
        next_values = np.zeros_like(values)
        next_values[active] = candidates[active].max(axis=1)
        error = np.max(np.abs(next_values[active] - values[active]))
        values = next_values
        if error < tolerance:
            q_star = np.zeros((500, 6))
            q_star[active] = (env.P_reward + gamma * values[env.P_next] * (~env.P_done))[active]
            return q_star, iteration
    raise RuntimeError("Value iteration failed to converge")


@lru_cache(maxsize=1)
def _grid_distances():
    """BFS distances using map walls directly, independent of transition arrays."""
    distances = np.full((25, 25), -1, dtype=int)
    for origin in range(25):
        distances[origin, origin] = 0
        queue = deque([origin])
        while queue:
            position = queue.popleft()
            row, col = divmod(position, 5)
            neighbors = []
            if row < 4:
                neighbors.append(position + 5)
            if row > 0:
                neighbors.append(position - 5)
            if col < 4 and TaxiEnv.MAP[row + 1][2 * col + 2] == ":":
                neighbors.append(position + 1)
            if col > 0 and TaxiEnv.MAP[row + 1][2 * col] == ":":
                neighbors.append(position - 1)
            for neighbor in neighbors:
                if distances[origin, neighbor] < 0:
                    distances[origin, neighbor] = distances[origin, position] + 1
                    queue.append(neighbor)
    if np.any(distances < 0):
        raise AssertionError("Taxi road graph unexpectedly disconnected")
    return distances


def shortest_steps(state):
    """Exact minimum actions to deliver from any valid decision state.

    Waiting passengers require travel to pickup, pickup, travel to destination,
    and dropoff. Aboard passengers require travel and dropoff only.
    """
    row, col, passenger, destination = TaxiEnv.decode(state)
    distances = _grid_distances()
    taxi_position = row * 5 + col
    dr, dc = TaxiEnv.landmarks[destination]
    destination_position = dr * 5 + dc
    if passenger == 4:
        return int(distances[taxi_position, destination_position] + 1)
    pr, pc = TaxiEnv.landmarks[passenger]
    pickup_position = pr * 5 + pc
    return int(distances[taxi_position, pickup_position] +
               distances[pickup_position, destination_position] + 2)
