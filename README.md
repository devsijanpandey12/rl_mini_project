# Taxi Pickup and Drop-off Using Q-Learning

The single main deliverable is **Taxi_Q_Learning_Mini_Project.ipynb**. All environment,
algorithm, evaluation, plotting, and test code is embedded in it. The notebook does not
depend on the development files in `taxi_parts/` or `_build/`.

## Open

Double-click **Launch_Taxi_Project.bat**, or run from this folder in Anaconda Prompt:

```bat
conda activate taxi_rl
python -m jupyterlab Taxi_Q_Learning_Mini_Project.ipynb
```

Select **Python (taxi_rl)**. Use **Restart Kernel and Run All Cells** to reproduce the study.
The notebook is delivered with saved executed outputs. **Taxi_Project_Preview.html** is a
browser-readable copy with figures and animation.

## What is implemented

- Custom deterministic Taxi-v3-compatible 5 x 5 environment and explicit transitions.
- All 3,000 transitions and 500 masks checked against Gymnasium 1.2.3.
- Tabular Q-learning, epsilon-greedy exploration, and a SARSA comparison.
- Controlled alpha, gamma, epsilon, exploration-decay, and supplied-mask experiments.
- Ten independent seeds and 5,000 episodes per condition.
- Evaluation over every one of the 300 allowed initial states, with frozen Q-tables.
- Exact BFS/value-iteration references; training does not receive those answers.
- Learning curves, parameter graphs, policy/value maps, actual route animation, tests,
  computed discussion, references, a six-minute video outline, and viva questions.

## Evidence files

`results/figures/` contains high-resolution PNGs for your video or slides. `results/tables/`
contains compressed raw episodes, configurations, endpoint summaries, and checks.
`results/trained_q_and_visits.npz` preserves final Q-tables and visit counts.
The run manifest records the exact settings, package versions, and Python interpreter.
The GIF and standalone HTML player show the actual learned demonstration trip.

## Recreate the environment

```bat
conda env create -f environment.yml
conda activate taxi_rl
python -m ipykernel install --user --name taxi_rl --display-name "Python (taxi_rl)"
```

The environment pins Gymnasium 1.2.3 / Taxi-v3 for reproducible validation. The current
online documentation describes v4; the notebook explains the version choice.

## Prepare your submission

Add your name, roll number, and class in the first cell. Review section 22's measured
findings, practice the hand-worked update, and use section 24's six-minute video outline.
Explain any failures or nonoptimal routes honestly. Video deadline: **16 October 2026**.
The notebook and animation are ready for a screen recording; uploading the recorded video
to the teacher's designated destination is a separate step.

## Build sources

`python _build/build_notebook.py` rebuilds the notebook and clears saved outputs.
`python _build/execute_notebook.py` executes it using the taxi_rl kernel and regenerates HTML.
The development sources are provided for reproducibility; they are not required to run
the self-contained notebook.
`python _build/verify_project.py` independently checks the saved notebook and evidence files.
