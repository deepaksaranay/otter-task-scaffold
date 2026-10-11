#!/bin/bash
set -e
APP_DIR="${OTTER_APP_DIR:-/app}"
export OTTER_APP_DIR="$APP_DIR"
cd "$APP_DIR"
mkdir -p "$APP_DIR/outputs" "$APP_DIR/workdir"

python << 'PYEOF'
import os
import numpy as np, pandas as pd, xgboost as xgb

SEED = 1337
APP = os.environ["OTTER_APP_DIR"]
IN, OUT = f"{APP}/task_inputs", f"{APP}/outputs"
tr = pd.read_csv(f"{IN}/train.csv")
te = pd.read_csv(f"{IN}/test_features.csv")

def features(d):
    f = d.drop(columns=["Id", "Cover_Type"], errors="ignore").copy()
    soil = [c for c in f.columns if c.startswith("Soil_Type")]
    wild = [c for c in f.columns if c.startswith("Wilderness_Area")]
    f["soil_id"] = f[soil].values.argmax(1)
    f["wild_id"] = f[wild].values.argmax(1)
    f["hydro_dist"] = np.hypot(f.Horizontal_Distance_To_Hydrology, f.Vertical_Distance_To_Hydrology)
    f["elev_minus_vdh"] = f.Elevation - f.Vertical_Distance_To_Hydrology
    f["fire_plus_road"] = f.Horizontal_Distance_To_Fire_Points + f.Horizontal_Distance_To_Roadways
    f["fire_minus_road"] = f.Horizontal_Distance_To_Fire_Points - f.Horizontal_Distance_To_Roadways
    f["hydro_plus_fire"] = f.Horizontal_Distance_To_Hydrology + f.Horizontal_Distance_To_Fire_Points
    return f

# Validation = a contiguous mid-elevation band of the training data, mimicking the unseen band.
lo, hi = tr.Elevation.quantile([0.45, 0.55])
is_val = tr.Elevation.between(lo, hi)
fit, val = tr[~is_val], tr[is_val]
val[["Id"]].to_csv(f"{OUT}/val_ids.csv", index=False)

params = dict(objective="multi:softprob", num_class=7, device="cuda", tree_method="hist",
              eta=0.08, max_depth=10, min_child_weight=2, subsample=0.8, colsample_bytree=0.7,
              reg_lambda=2.0, eval_metric="mlogloss", seed=SEED)

dfit = xgb.DMatrix(features(fit), label=fit.Cover_Type - 1)
dval = xgb.DMatrix(features(val), label=val.Cover_Type - 1)
m = xgb.train(params, dfit, 3000, evals=[(dval, "val")], early_stopping_rounds=60, verbose_eval=False)
best = m.best_iteration + 1
pv = m.predict(dval, iteration_range=(0, best)).argmax(1) + 1
pd.DataFrame({"Id": val.Id, "Cover_Type": pv}).to_csv(f"{OUT}/val_predictions.csv", index=False)

# Final model on all labelled rows, slightly more rounds because there is more data.
dall = xgb.DMatrix(features(tr), label=tr.Cover_Type - 1)
final = xgb.train(params, dall, int(best * 1.1))
pt = final.predict(xgb.DMatrix(features(te))).argmax(1) + 1
pd.DataFrame({"Id": te.Id, "Cover_Type": pt}).to_csv(f"{OUT}/test_predictions.csv", index=False)
print("val macro rounds:", best)
PYEOF

echo "Task completed successfully"
