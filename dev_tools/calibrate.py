"""After a successful oracle run: score it with checks.py and log it as human_1 in human_scores.json.
Usage (inside the oracle container): python calibrate.py /path/to/task
Then set pass_criteria thresholds at or below the printed values.
"""
import importlib.util, json, sys
from pathlib import Path
task = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("checks", task / "tests" / "checks.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
res = m.evaluate()
for c in res["criteria"]:
    print("PASS" if c["passed"] else "FAIL", c["id"], c["detail"])
print("metrics:", res["metrics"])
p = task / "tests" / "human_scores.json"
hs = json.loads(p.read_text())
hs["attempts"] = [{
    "id": "human_1",
    "architecture": "XGBoost multi:softprob, GPU hist, depth 10, eta 0.08",
    "description": "Engineered distance/soil/wilderness features; early stopping on a held-out 45-55th percentile elevation band; refit on all training rows.",
    "metrics": res["metrics"],
}]
p.write_text(json.dumps(hs, indent=2))
print("wrote", p)
