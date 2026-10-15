"""Synchronize submission details without rerunning or clearing saved results."""
from pathlib import Path
import ast
import hashlib
import json
import textwrap
import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "Taxi_Q_Learning_Mini_Project.ipynb"
notebook = nbformat.read(path, as_version=4)
before = hashlib.sha256(json.dumps(
    [dict(c) for c in notebook.cells if c.cell_type == "code"],
    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

headers = {"# Taxi Pickup and Drop-off Using Q-Learning",
           "## 24. Present the work in a six-minute video"}
replacements = {}
builder = ast.parse((ROOT / "_build" / "build_notebook.py").read_text(encoding="utf-8"))
for node in builder.body:
    if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == "md"
            and node.value.args and isinstance(node.value.args[0], ast.Constant)
            and isinstance(node.value.args[0].value, str)):
        source = textwrap.dedent(node.value.args[0].value).strip()
        if source.splitlines()[0] in headers:
            replacements[source.splitlines()[0]] = source
assert set(replacements) == headers
updated = set()
for cell in notebook.cells:
    header = cell.source.splitlines()[0] if cell.source else ""
    if cell.cell_type == "markdown" and header in replacements:
        cell.source = replacements[header]
        updated.add(header)
assert updated == headers
notebook.metadata["authors"] = [{"name": "Sijan Pandey"}]

after = hashlib.sha256(json.dumps(
    [dict(c) for c in notebook.cells if c.cell_type == "code"],
    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
assert before == after, "Code cells or saved results changed"
assert not any("Add your name" in c.source or "Add your identity" in c.source
               for c in notebook.cells)
nbformat.validate(notebook)
nbformat.write(notebook, path)
html, _ = HTMLExporter().from_notebook_node(notebook)
(ROOT / "Taxi_Project_Preview.html").write_text(html, encoding="utf-8")
assert "Sijan Pandey" in html
assert "Add your name, roll number, and class" not in html
print("Student details updated in notebook and HTML; all code and saved outputs preserved.")
