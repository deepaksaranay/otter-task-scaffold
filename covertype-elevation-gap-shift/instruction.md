Forest Service cartographic data labels 30 m x 30 m cells with one of seven forest cover types. You have a labelled table and an unlabelled table in `/app/task_inputs/`: `train.csv` and `test_features.csv` (columns are described in `data_description.md`).

Every test cell falls in one band of the elevation range that is completely absent from the training data, so ordinary random cross-validation will not tell you how a model behaves there. Build a classifier that holds up across that gap, using the GPU where it helps. Use only the provided files.

Write these to `/app/outputs/`:

- `val_ids.csv`: one column `Id`. A validation subset of `train.csv` that you hold out to imitate the unseen band. It must contain at least 5% of the training rows, and every training row whose elevation lies inside the range of these rows must be in it, with training rows present on both sides of that range.
- `val_predictions.csv`: columns `Id`, `Cover_Type`, one row per validation id, predicted by a model that never trained on those rows.
- `test_predictions.csv`: columns `Id`, `Cover_Type` (integers 1-7), one row per row of `test_features.csv`.

Pass requires macro-F1 on the test cells >= 0.80 and the lowest per-class recall (over classes with at least 50 test cells) >= 0.50.
