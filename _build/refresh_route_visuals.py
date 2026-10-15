"""Refresh route legends using saved Q-tables; preserve all measured experiments.

Synchronize the changed notebook cells with build_notebook.py, rerun only the
reference/route/animation presentation cells, and replace their saved media.
Training and evaluation tables are neither recomputed nor edited.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import base64
import hashlib
import io
import json
import os
import sys
import tempfile
import textwrap

ROOT = Path(__file__).resolve().parents[1]
if os.name == "nt":
    library_bin = Path(sys.prefix) / "Library" / "bin"
    os.environ["PATH"] = str(library_bin) + os.pathsep + os.environ.get("PATH", "")
    _dll_handle = os.add_dll_directory(str(library_bin))
sys.path.insert(0, str(ROOT))
# Keep Matplotlib's temporary frame files inside this writable project folder.
# The resolved guard also confines TemporaryDirectory's automatic cleanup here.
render_temp = (ROOT / "_build" / "render_temp").resolve()
assert ROOT.resolve() in render_temp.parents
render_temp.mkdir(exist_ok=True)
tempfile.tempdir = str(render_temp)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import nbformat
from nbconvert import HTMLExporter
import numpy as np
import pandas as pd

from taxi_parts.core import TaxiEnv, rollout, shortest_steps, value_iteration
from taxi_parts.visuals import draw_taxi_map, plot_policy_slices, animate_taxi

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "figure.dpi": 110, "savefig.dpi": 180,
                     "animation.embed_limit": 60})
path = ROOT / "Taxi_Q_Learning_Mini_Project.ipynb"
notebook = nbformat.read(path, as_version=4)
evidence_paths = [*sorted((ROOT / "results" / "tables").glob("*")),
                  ROOT / "results" / "trained_q_and_visits.npz",
                  ROOT / "results" / "run_manifest.json"]


def evidence_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in evidence_paths if p.is_file()}


before_hashes = evidence_hashes()
# Extract literal notebook cells without invoking the builder (which clears outputs).
builder = ast.parse((ROOT / "_build" / "build_notebook.py").read_text(encoding="utf-8"))
declared = {}
for node in builder.body:
    if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in {"md", "code"} and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and isinstance(node.value.args[0].value, str)):
        source = textwrap.dedent(node.value.args[0].value).strip()
        declared[source.splitlines()[0]] = source

markers = ["## 9. Visualize the task and inspect transitions",
           "## 10. Baseline learning: Q-learning and SARSA",
           "## 19. Inspect the policy and watch a real trip",
           "env = TaxiEnv()",
           'save_figure(plot_policy_slices(env,q_runs[0].q,passenger=0,destination=3),"13_conditioned_policy_and_values")']
matched = {}
for cell in notebook.cells:
    first_line = cell.source.splitlines()[0] if cell.source else ""
    if first_line in markers:
        cell.source = declared[first_line]
        matched[first_line] = cell
    if cell.source.startswith('"""Seed-aware plots'):
        cell.source = (ROOT / "taxi_parts" / "visuals.py").read_text(encoding="utf-8").strip()
assert set(matched) == set(markers)

rendered = {}


def save_figure(figure, name):
    # The policy-slice figure shares a cell with the route but has not changed.
    if name not in {"01_task_map_and_reference", "14_before_and_after_actual_trips"}:
        plt.close(figure)
        return
    figure.savefig(ROOT / "results" / "figures" / f"{name}.png",
                   dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight",
                   facecolor=figure.get_facecolor())
    rendered[name] = {"image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                      "text/plain": repr(figure)}
    # Validate the on-image legend agrees with the rendered line styles.
    for ax in figure.axes:
        legend = ax.get_legend()
        assert legend is not None
        assert [line.get_linestyle() for line in legend.get_lines()] == ["--", "-", "-"]
        assert "not on board" in legend.get_texts()[0].get_text()
        assert "passenger on board" in legend.get_texts()[1].get_text()
    plt.close(figure)


arrays = np.load(ROOT / "results" / "trained_q_and_visits.npz")
q = arrays["q_condition_0_seed_11"]
scope = {"np": np, "pd": pd, "plt": plt, "TaxiEnv": TaxiEnv,
         "rollout": rollout, "shortest_steps": shortest_steps,
         "value_iteration": value_iteration, "draw_taxi_map": draw_taxi_map,
         "plot_policy_slices": plot_policy_slices, "animate_taxi": animate_taxi,
         "save_figure": save_figure, "display": lambda *args: None,
         "HTML": lambda value: value, "animation": animation,
         "BASE": SimpleNamespace(gamma=.95), "q_runs": [SimpleNamespace(q=q)],
         "EVAL_STARTS": TaxiEnv().initial_states, "MAX_STEPS": 200,
         "OUTPUT": ROOT / "results"}

for marker in markers[3:]:
    exec(compile(matched[marker].source, "refreshed_route_cell", "exec"), scope)
assert (scope["learned_route"]["steps"], scope["learned_route"]["return"],
        scope["learned_route"]["success"]) == (14, 7, True)
assert (scope["before_route"]["steps"], scope["before_route"]["return"],
        scope["before_route"]["success"]) == (35, -134, False)

reference_cell, trip_cell = (matched[m] for m in markers[3:])
reference_images = [o for o in reference_cell.outputs if "image/png" in o.get("data", {})]
trip_images = [o for o in trip_cell.outputs if "image/png" in o.get("data", {})]
assert len(reference_images) == 1 and len(trip_images) == 2
reference_images[0].data = rendered["01_task_map_and_reference"]
trip_images[1].data = rendered["14_before_and_after_actual_trips"]

animation_cell = next(c for c in notebook.cells
                      if c.cell_type == "code" and c.source.startswith("taxi_animation="))
exec(compile(animation_cell.source, "refreshed_animation_cell", "exec"), scope)
animation_output = next(o for o in animation_cell.outputs if "text/html" in o.get("data", {}))
animation_output.data["text/html"] = scope["animation_html"]

for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"notebook_cell_{index + 1}", "exec")
nbformat.validate(notebook)
nbformat.write(notebook, path)
html, _ = HTMLExporter().from_notebook_node(notebook)
(ROOT / "Taxi_Project_Preview.html").write_text(html, encoding="utf-8")
assert evidence_hashes() == before_hashes, "Measured experiment evidence changed"

audit_path = ROOT / "results" / "notebook_execution_audit.json"
audit = json.loads(audit_path.read_text(encoding="utf-8"))
audit["visual_clarification"] = (
    "Route plots and animation rerendered from unchanged saved Q-tables; orange dashed "
    "means passenger not on board, teal solid means on board; wall legend added. "
    "Notebook code/Markdown and HTML synchronized. Experiment files unchanged."
)
audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
print("Route figures, notebook, animation, and HTML refreshed; experiment evidence unchanged.")
