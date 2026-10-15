"""Independent checks of the delivered notebook and exported experiment evidence."""
from pathlib import Path
import ast
import json
import nbformat
import numpy as np
import pandas as pd
from PIL import Image, ImageStat

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"results"
notebook=nbformat.read(ROOT/"Taxi_Q_Learning_Mini_Project.ipynb",as_version=4)
nbformat.validate(notebook)
code_cells=[c for c in notebook.cells if c.cell_type=="code"]
assert len(code_cells)==25
assert [c.execution_count for c in code_cells]==list(range(1,26))
assert not [o for c in code_cells for o in c.outputs if o.output_type=="error"]
for c in code_cells:
    tree=ast.parse(c.source)
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):
            assert not (node.module or "").startswith("taxi_parts")
        if isinstance(node,ast.Import):
            assert not any(a.name.startswith("taxi_parts") for a in node.names)
assert any(c.source.strip()==(ROOT/"taxi_parts"/"visuals.py").read_text(encoding="utf-8").strip() for c in code_cells)
assert any(c.source.strip()==(ROOT/"taxi_parts"/"experiment_helpers.py").read_text(encoding="utf-8").strip() for c in code_cells)

manifest=json.loads((RESULTS/"run_manifest.json").read_text(encoding="utf-8"))
assert manifest["conditions"]==10 and manifest["training_runs"]==100
assert manifest["training_episodes"]==500_000
assert Path(manifest["python_executable"]).parent.name=="taxi_rl"
training=pd.read_csv(RESULTS/"tables"/"training_episodes.csv.gz")
assert len(training)==500_000
sizes=training.groupby(["condition_id","seed"]).size()
assert len(sizes)==100 and (sizes==5000).all()
assert (training["return"]==-training.steps-9*training.illegal_actions+21*training.success.astype(int)).all()
assert (training.success ^ training.truncated).all()
assert training.steps.between(1,200).all()
baseline=training[training.condition_id==0]
for (_,seed),frame in training.groupby(["condition_id","seed"]):
    assert np.array_equal(frame.start_state.to_numpy(),baseline[baseline.seed==seed].start_state.to_numpy())

arrays=np.load(RESULTS/"trained_q_and_visits.npz")
for (condition,seed),frame in training.groupby(["condition_id","seed"]):
    q=arrays[f"q_condition_{condition}_seed_{seed}"]
    visits=arrays[f"visits_condition_{condition}_seed_{seed}"]
    assert q.shape==(500,6) and np.isfinite(q).all()
    assert visits.sum()==frame.steps.sum()

evaluation=pd.read_csv(RESULTS/"tables"/"evaluation_episodes.csv.gz")
assert len(evaluation)==21_300
assert (evaluation["return"]==-evaluation.steps-9*evaluation.illegal_actions+21*evaluation.success.astype(int)).all()
assert (evaluation["return"]<=evaluation.optimal_return+1e-9).all()
for _,frame in evaluation.groupby(["agent","mode","train_seed"]):
    assert len(frame)==300 and frame.start_state.nunique()==300
qg=evaluation[(evaluation.agent=="Q-learning")&(evaluation["mode"]=="Greedy")]
assert len(qg)==3000 and qg.success.all()
oracle=evaluation[evaluation.agent=="Shortest-path reference"]
assert oracle.optimal_delivery.all() and np.isclose(oracle["return"].mean(),7.93)
assert np.isclose(oracle.steps.mean(),13.07)
masked=evaluation[evaluation.agent=="Q + mask"]
assert (masked.illegal_actions==0).all() and (masked.wall_hits==0).all()

checks=pd.read_csv(RESULTS/"tables"/"implementation_checks.csv")
study_checks=pd.read_csv(RESULTS/"tables"/"experiment_checks.csv")
assert len(checks)==13 and (checks.status=="PASS").all()
assert len(study_checks)==5 and (study_checks.status=="PASS").all()
transitions=pd.read_csv(RESULTS/"tables"/"complete_transition_model.csv")
assert len(transitions)==3000 and not transitions.duplicated(["state","action"]).any()
assert (transitions.probability==1).all()
reference=pd.read_csv(RESULTS/"tables"/"exact_reference_all_starts.csv")
assert len(reference)==300 and np.isclose(reference.optimal_steps.mean(),13.07)

figures=list((RESULTS/"figures").glob("*.png"))
assert len(figures)==15
for file in figures:
    with Image.open(file) as picture:
        assert picture.width>=800 and picture.height>=300
        assert max(ImageStat.Stat(picture.convert("RGB")).stddev)>10
with Image.open(RESULTS/"learned_taxi_trip.gif") as picture:
    assert picture.n_frames>=14
    picture.seek(picture.n_frames-1)
    picture.load()
assert (ROOT/"Taxi_Project_Preview.html").stat().st_size>1_000_000

audit={"status":"PASS","executed_code_cells":len(code_cells),"self_contained_notebook":True,
       "training_runs":len(sizes),"training_episodes":len(training),
       "main_evaluation_trips":len(evaluation),"initial_states_per_evaluation":300,
       "implementation_and_study_check_groups":18,"verified_static_figures":len(figures),
       "custom_transition_rows":len(transitions),"exact_reference_starts":len(reference)}
(RESULTS/"deliverable_verification.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
print(json.dumps(audit,indent=2))
