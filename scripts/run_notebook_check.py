# -*- coding: utf-8 -*-
"""Execute the notebook to verify it runs. Produces a copy with outputs."""
import json, sys, os, time, traceback

NB_IN  = r"E:\DSAI\DSAI4207-new\full-work-though.ipynb"
NB_OUT = r"E:\DSAI\DSAI4207-new\_executed_check.ipynb"

nb = json.load(open(NB_IN, encoding="utf-8"))

# Flip stage switches so the verification run does not download the 4B model or train for hours.
# RUN_DATA stays True so the dataset-prep cells are genuinely exercised.
DISABLE = ["RUN_DOWNLOAD", "RUN_TRAIN", "RUN_EVAL", "RUN_MERGE", "RUN_GGUF", "RUN_SETUP"]
patched = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = c["source"]
    if isinstance(src, list):
        src = "".join(src)
    lines = src.split("\n")
    for i, line in enumerate(lines):
        for name in DISABLE:
            if line.strip().startswith(name) and "    = " in line and "True" in line:
                lines[i] = line.replace("True", "False", 1)
                patched += 1
    c["source"] = "\n".join(lines)
print("patched switches:", patched)

import nbformat
from nbclient import NotebookClient

nbf = nbformat.from_dict(nb)
# normalise: ensure string sources + ids
for i, c in enumerate(nbf.cells):
    if isinstance(c.get("source"), list):
        c["source"] = "".join(c["source"])
    if not c.get("id"):
        c["id"] = f"cell-{i:03d}"

client = NotebookClient(nbf, timeout=3600, kernel_name="dsai4207",
                        allow_errors=True,
                        resources={"metadata": {"path": r"E:\DSAI\DSAI4207-new"}})
t0 = time.time()
try:
    client.execute()
    print("EXECUTION FINISHED in %.1f min" % ((time.time()-t0)/60))
except Exception as e:
    print("EXECUTION RAISED:", type(e).__name__, str(e)[:600])
    traceback.print_exc()

nbformat.write(nbf, NB_OUT)
print("wrote", NB_OUT)

errs = 0
for i, c in enumerate(nbf.cells):
    if c.get("cell_type") != "code":
        continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            errs += 1
            print("\n" + "=" * 70)
            print(f"CELL {i} ERROR: {o.get('ename')}: {str(o.get('evalue'))[:400]}")
            tb = o.get("traceback", [])
            if tb:
                print("\n".join(tb[-8:]))
print(f"\nTOTAL ERROR OUTPUTS: {errs}")
