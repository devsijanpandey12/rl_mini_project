# Taxi Pickup and Drop-off Using Q-Learning

**Reinforcement Learning Mini Project**

| Student Details | |
| :--- | :--- |
| **Name** | Sijan Pandey |
| **Roll No.** | 11 |
| **Programme** | B.Tech in Artificial Intelligence |
| **Semester** | 8th Semester |

## 1. Project Overview

This project trains a simulated taxi to locate a passenger, pick them up, and deliver them to the requested destination while minimizing unnecessary actions and penalties. It applies **tabular Q-learning**, a Temporal-Difference reinforcement learning algorithm, to a custom implementation of the classic Taxi environment.

The study covers environment construction, agent implementation, training, controlled experiments, evaluation, visualization, and interpretation of results. **SARSA** and a **random policy** provide comparisons, while an exact shortest-path solution provides a reference for measuring route efficiency.

## 2. Objectives

- Construct the Taxi gridworld and explicitly define its transition dynamics and rewards.
- Implement Q-learning with an epsilon-greedy exploration policy.
- Investigate how learning rate, discount factor, and exploration affect learning.
- Compare constant and decaying exploration, Q-learning and SARSA, and training with and without action masking.
- Evaluate delivery success, return, route length, illegal service attempts, and shortest-route performance.
- Explain the exploration–exploitation trade-off, observed failures, and the limitations of the simulation.

## 3. Environment and Task Definition

| Component | Specification |
| :--- | :--- |
| Map | A deterministic 5 × 5 grid with internal walls and four passenger landmarks |
| State | Taxi row, taxi column, passenger location or onboard status, and destination |
| State space | 500 encoded states; 300 allowed initial states |
| Actions | South, North, East, West, Pickup, Drop-off |
| Ordinary action reward | −1 |
| Illegal pickup/drop-off reward | −10 |
| Successful delivery reward | +20 and episode termination |
| Episode limit | 200 actions |

The environment was implemented from scratch and checked against **Taxi-v3 in Gymnasium 1.2.3**. All **3,000 state–action transitions** and **500 action masks** matched the reference implementation. Attempting to move into a wall leaves the taxi in place and costs one step; it does not represent a physical collision.

## 4. Methodology and Work Completed

1. **Environment implementation:** State encoding/decoding, walls, passenger interactions, rewards, transitions, and episode handling.
2. **Agent implementation:** Q-tables, epsilon-greedy action selection, random tie-breaking, Q-learning updates, and SARSA updates.
3. **Controlled training:** Independent random seeds, matched starting-state sequences across conditions, and separate random generators for starts and actions.
4. **Evaluation:** Frozen Q-tables evaluated from all 300 allowed initial states, using both greedy and exploratory policies.
5. **Reference solutions:** Breadth-first search and value iteration to measure optimality. These solutions were used for evaluation and were not supplied to the learning agents.
6. **Analysis and visualization:** Learning curves, parameter comparisons, value and policy maps, route comparisons, checkpoint evaluation, and an animation of an actual learned trip.
7. **Verification:** Environment checks, learning-update tests, experiment integrity checks, and a reproduced SARSA failure trace.

Q-learning uses the update:

```text
Q(s, a) ← Q(s, a) + α [r + γ max Q(s′, a′) − Q(s, a)]
```

The maximum is over next-state actions. The bootstrap term is zero after a successful terminal delivery. A time-limit cutoff is handled separately from termination. The notebook includes a worked numerical example explaining the update.

## 5. Experimental Design

| Experiment | Settings Compared |
| :--- | :--- |
| Learning rate, α | 0.1, 0.3, 0.7 |
| Discount factor, γ | 0.5, 0.95, 1.0 |
| Exploration rate, ε | 0.01, 0.1, 0.3 |
| Exploration schedule | Constant ε = 0.3 versus linear decay from 0.3 to 0.01 |
| Action masking | Standard Q-learning versus Q-learning with supplied valid-action masks |
| Control algorithm | Q-learning versus SARSA |

The baseline uses **α = 0.3, γ = 0.95, and ε = 0.1**. The study contains **10 unique conditions**, each trained for **5,000 episodes across 10 independent seeds**: **100 runs and 500,000 training episodes** in total. Parameter sweeps vary one factor at a time.

Learning curves use a trailing 100-episode average. Shaded uncertainty bands show **±1 standard error across independent seeds**, rather than confidence intervals. Evaluation is performed without further learning.

## 6. Key Results

The following results use **greedy evaluation (ε = 0)** over every allowed initial state. Learned-agent results are averaged across 10 trained seeds. The exact reference is computed directly.

| Policy | Mean Return | Mean Actions per Trip | Successful Delivery | Shortest-Route Delivery* |
| :--- | ---: | ---: | ---: | ---: |
| Q-learning | 7.900 | 13.100 | 100.00% | 98.50% |
| Q-learning with action masking | 7.917 | 13.083 | 100.00% | 99.33% |
| SARSA | −22.262 | 40.238 | 85.60% | 73.80% |
| Exact shortest-path reference | 7.930 | 13.070 | 100.00% | 100.00% |

*Shortest-route delivery is the percentage of all evaluated trips that completed using the minimum possible number of actions.*

Q-learning achieved reliable delivery and routes close to the exact reference. Action masking slightly improved route optimality. Under exploratory evaluation at ε = 0.1, masking also removed illegal service attempts by construction because the validity rules were supplied to the agent.

SARSA did not consistently learn a successful greedy policy within this training budget: **14.4% of greedy evaluation trips reached the action limit**. The notebook reproduces a failure in which the learned policy repeatedly selects a blocked movement. This is a measured result for these settings and this finite budget, rather than a general claim that SARSA is inferior.

The full notebook discusses the measured effects of α, γ, ε, exploration decay, and masking using the saved experimental results.

## 7. Visualizations and Validation

The completed notebook includes **15 static figures**, an animated learned trip, and written interpretation of the results. Visuals cover training return, delivery success, illegal actions, learning speed, parameter sensitivity, policy/value maps, and route optimality.

All **25 code cells** were executed successfully. The saved validation includes **13 core check groups and 5 study-integrity check groups**, for **18 passing groups**. Raw episode data, final Q-tables, visit counts, configurations, and execution records are retained for inspection.

## 8. Submission Files

| File or Folder | Contents |
| :--- | :--- |
| [Taxi_Q_Learning_Mini_Project.ipynb](Taxi_Q_Learning_Mini_Project.ipynb) | Main self-contained notebook: explanations, complete code, saved outputs, experiments, and conclusions |
| [Taxi_Project_Preview.html](Taxi_Project_Preview.html) | Browser-readable notebook with graphs and animation |
| `results/figures/` | High-resolution graphs |
| `results/tables/` | Experimental data, summaries, transition checks, and failure analysis |
| `results/trained_q_and_visits.npz` | Trained Q-tables and state–action visit counts |
| [Learned trip animation](results/learned_taxi_trip.html) | Standalone playback of the learned demonstration trip |
| `results/run_manifest.json` | Settings, package versions, and execution environment |
| `results/deliverable_verification.json` | Saved deliverable verification report |
| `environment.yml` / `environment-lock-win-64.txt` | Conda environment specification and exact Windows package record |

The main notebook contains all code required to run the project. The additional `taxi_parts/` and `_build/` folders retain supporting development and reproduction scripts.

## 9. How to Review and Run

**For review:** Open `Taxi_Project_Preview.html` in a browser, or open the notebook to inspect its saved outputs. No training run is required to view the submitted results.

**To reproduce:** Double-click `Launch_Taxi_Project.bat`, or open Anaconda Prompt in this project folder and run:

```bat
conda activate taxi_rl
python -m jupyterlab Taxi_Q_Learning_Mini_Project.ipynb
```

Select the **Python (taxi_rl)** kernel and choose **Restart Kernel and Run All Cells**.

To create the environment on another computer:

```bat
conda env create -f environment.yml
conda activate taxi_rl
python -m ipykernel install --user --name taxi_rl --display-name "Python (taxi_rl)"
```

The environment pins Gymnasium 1.2.3 to preserve the Taxi-v3 validation reference. The notebook explains the version choice.

## 10. Conclusion and Limitations

The project demonstrates how a tabular reinforcement learning agent can learn a passenger pickup-and-delivery task and how exploration and parameter choices influence its behavior. Evaluating all allowed starting states, comparing with an exact reference, and examining unsuccessful policies provide evidence beyond a single successful demonstration.

The results apply to a small, deterministic, stationary simulator. They do not establish real-road driving safety or performance on unseen maps. Action masking uses supplied environment knowledge, parameter sweeps do not capture every interaction, and finite learning curves do not prove theoretical convergence.

## 11. References

- Sutton, R. S., and Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd edition. [Book website](http://incompleteideas.net/book/the-book-2nd.html).
- Watkins, C. J. C. H., and Dayan, P. (1992). *Q-learning*. [Paper](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).
- Farama Foundation. [Taxi environment documentation](https://gymnasium.farama.org/environments/toy_text/taxi/). The project uses the pinned Taxi-v3 reference described above.

Additional references and detailed explanations are included in the notebook.
