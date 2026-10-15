"""Execute the complete notebook and export a browser-readable copy."""
from pathlib import Path
from time import perf_counter
import asyncio
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
if os.name=="nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    prefix=Path(sys.prefix)
    os.environ["PATH"]=os.pathsep.join(str(prefix/p) for p in ("","Library/bin","Scripts"))+os.pathsep+os.environ.get("PATH","")

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

path=ROOT/"Taxi_Q_Learning_Mini_Project.ipynb"
notebook=nbformat.read(path,as_version=4)
started=perf_counter()


def announce(cell,cell_index,**kwargs):
    if cell.cell_type=="code":
        print(f"Cell {cell_index+1}/{len(notebook.cells)}: {cell.source.splitlines()[0][:90]}",flush=True)


client=NotebookClient(notebook,timeout=1200,kernel_name="taxi_rl",
                      resources={"metadata":{"path":str(ROOT)}},on_cell_start=announce)
try:
    client.execute()
finally:
    nbformat.write(notebook,path)
nbformat.validate(notebook)
code_cells=[c for c in notebook.cells if c.cell_type=="code"]
errors=[o for c in code_cells for o in c.outputs if o.output_type=="error"]
assert not errors,errors
assert all(c.execution_count is not None for c in code_cells)
html,_=HTMLExporter().from_notebook_node(notebook)
(ROOT/"Taxi_Project_Preview.html").write_text(html,encoding="utf-8")
audit={"notebook":path.name,"code_cells_executed":len(code_cells),"error_outputs":len(errors),
       "execution_seconds":round(perf_counter()-started,2),"kernel":"taxi_rl","nbformat_valid":True}
(ROOT/"results"/"notebook_execution_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
print(json.dumps(audit,indent=2),flush=True)
