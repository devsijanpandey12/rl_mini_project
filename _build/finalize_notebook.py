"""Clarify editorial SE/evaluation labels without changing experiment results."""
from pathlib import Path
import base64
import io
import json
import os
import sys
import nbformat
from nbconvert import HTMLExporter

root=Path(__file__).resolve().parents[1]
path=root/"Taxi_Q_Learning_Mini_Project.ipynb"
notebook=nbformat.read(path,as_version=4)
old="individual seed SEs are in the endpoint table"
new="SEs across trained seed means are in the endpoint table"
visual_source=(root/"taxi_parts"/"visuals.py").read_text(encoding="utf-8").strip()
for cell in notebook.cells:
    cell.source=cell.source.replace(old,new)
    if cell.source.startswith('"""Seed-aware plots'):
        cell.source=visual_source
    for output in cell.get("outputs",[]):
        if isinstance(output.get("text"),str):
            output["text"]=output["text"].replace(old,new)
        for mime,value in output.get("data",{}).items():
            if mime.startswith("text/") and isinstance(value,str):
                output["data"][mime]=value.replace(old,new)
    if cell.cell_type=="code":
        compile(cell.source,"notebook_cell","exec")

# This is an editorial title change; rerender this plot from unchanged raw data.
if os.name=="nt":
    os.environ["PATH"]=str(Path(sys.prefix)/"Library"/"bin")+os.pathsep+os.environ.get("PATH","")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
scope={}
exec(compile(visual_source,"visuals.py","exec"),scope)
evaluation=pd.read_csv(root/"results"/"tables"/"evaluation_episodes.csv.gz")
learned=evaluation[evaluation.agent.isin(["Q-learning","SARSA","Q + mask"])]
figure=scope["plot_evaluation"](learned)
figure.savefig(root/"results"/"figures"/"09_frozen_learned_policy_evaluation.png",dpi=180,bbox_inches="tight",facecolor=figure.get_facecolor())
buffer=io.BytesIO()
figure.savefig(buffer,format="png",dpi=110,bbox_inches="tight",facecolor=figure.get_facecolor())
plt.close(figure)
for cell in notebook.cells:
    if cell.cell_type=="code" and cell.source.startswith("learned_group ="):
        output=next(o for o in cell.outputs if "image/png" in o.get("data",{}))
        output["data"]["image/png"]=base64.b64encode(buffer.getvalue()).decode("ascii")
        break
else:
    raise AssertionError("Frozen evaluation output was not found.")
nbformat.validate(notebook)
nbformat.write(notebook,path)
findings=root/"results"/"measured_findings.md"
findings.write_text(findings.read_text(encoding="utf-8").replace(old,new),encoding="utf-8")
html,_=HTMLExporter().from_notebook_node(notebook)
(root/"Taxi_Project_Preview.html").write_text(html,encoding="utf-8")
audit_path=root/"results"/"notebook_execution_audit.json"
audit=json.loads(audit_path.read_text(encoding="utf-8"))
audit["editorial_clarification"]="SE and evaluation-title wording clarified; one figure rerendered from unchanged raw evaluation data. Numeric results unchanged."
audit_path.write_text(json.dumps(audit,indent=2),encoding="utf-8")
print("Final notebook and HTML preview saved; numeric experiment results unchanged.")
