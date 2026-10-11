# Finish and submit covertype-elevation-gap-shift

1. Data: `python dev_tools/prepare_data.py --task-dir covertype-elevation-gap-shift`
   (creates train.csv, test_features.csv, tests/test_labels.csv from the real UCI data).
2. Copy the official skeleton's `tests/rubric_score.py` and `tests/pass_fail.py` into `tests/` (unedited). Compare my `checks.py` return shape with the skeleton's and adjust if it differs.
3. Dockerfile: pin the base-image digest and the `libgomp1` version (TODO lines).
4. Oracle on GPU: `stb harbor run -a oracle -e modal -p ./covertype-elevation-gap-shift`. Read and rewrite `solve.sh` so the approach is yours.
5. `python dev_tools/calibrate.py covertype-elevation-gap-shift` (log the oracle in human_scores.json), then set thresholds in `pass_criteria` at or below the oracle values, and update `instruction.md` to the same numbers.
6. Run Gemini 3.5 Flash; adjust difficulty. Break the oracle on purpose to confirm tests fail.
7. `zip -r covertype-elevation-gap-shift.zip covertype-elevation-gap-shift/`, then upload with Tabular ML > classification.
