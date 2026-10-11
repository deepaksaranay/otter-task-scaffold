"""Build the real task data from UCI Covertype. Run once on your machine, then upload the zip.
Usage: python prepare_data.py --task-dir covertype-elevation-gap-shift
Needs: pip install pandas numpy requests
"""
import argparse, gzip, io, zipfile
from pathlib import Path
import numpy as np, pandas as pd, requests

URL = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
NAMES = (["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
          "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
          "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
         + [f"Wilderness_Area{i}" for i in range(1, 5)]
         + [f"Soil_Type{i}" for i in range(1, 41)] + ["Cover_Type"])
SEED, TRAIN_ROWS = 20261011, 200_000

ap = argparse.ArgumentParser(); ap.add_argument("--task-dir", required=True)
root = Path(ap.parse_args().task_dir)

z = zipfile.ZipFile(io.BytesIO(requests.get(URL, timeout=300).content))
df = pd.read_csv(io.BytesIO(gzip.decompress(z.read("covtype.data.gz"))), header=None, names=NAMES)
print("rows:", len(df))

lo, hi = df.Elevation.quantile([0.45, 0.55])
band = df.Elevation.between(lo, hi)
rng = np.random.RandomState(SEED)
test = df[band].sample(frac=1, random_state=SEED).reset_index(drop=True)
train = df[~band].sample(n=TRAIN_ROWS, random_state=SEED).reset_index(drop=True)
train.insert(0, "Id", np.arange(len(train)))
test.insert(0, "Id", 1_000_000 + np.arange(len(test)))
assert train.Elevation.between(lo, hi).sum() == 0

ti = root / "environment" / "task_inputs"
train.to_csv(ti / "train.csv", index=False)
test.drop(columns="Cover_Type").to_csv(ti / "test_features.csv", index=False)
test[["Id", "Cover_Type"]].to_csv(root / "tests" / "test_labels.csv", index=False)
print("train", len(train), "test", len(test), "band", lo, hi)
print("test class counts:\n", test.Cover_Type.value_counts().sort_index())
