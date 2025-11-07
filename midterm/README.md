# Midterm Checkpoint — Song Popularity (CS 4741/4641)

Goal
----
Predict song popularity (log1p playcount) from Million Song Dataset (MSD) metadata and Echo Nest Taste Profile signals. Compare a linear baseline with a tree model using a temporal split (train ≤ 2008, test > 2008).

Links
-----
- GitHub Pages (report): https://github.gatech.edu/pages/amohammad38/ml_projectpage.github.io/
- Repository: this repo (private; mentor added as collaborator)

Repository structure
--------------------
- midterm/run_midterm.py — trains Elastic Net and Histogram Gradient Boosting on midterm/data/working/merged.csv; saves plots and metrics
- midterm/make_merged_from_msd.py — builds midterm/data/working/merged.csv from raw MSD files (track_metadata.db, unique_tracks.txt, train_triplets.txt)
- midterm/data/working/merged.csv — merged dataset ready for modeling (committed)
- midterm/figs/ — output figures and metrics_summary.txt
- midterm/requirements.txt — Python dependencies
- data/raw/ — place large raw inputs here if rebuilding merged.csv (not tracked by git)
- _config.yml — GitHub Pages configuration

Quick start (use committed dataset)
-----------------------------------
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r midterm/requirements.txt

python midterm/run_midterm.py \
  --data midterm/data/working/merged.csv \
  --year_col year \
  --playcount_col playcount \
  --numeric_cols duration,year,artist_familiarity,artist_hotttnesss \
  --train_end_year 2008

open midterm/figs/reg_pred_actual_scatter.png
open midterm/figs/reg_resid_hist.png
cat midterm/figs/metrics_summary.txt
