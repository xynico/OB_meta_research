# meta_OB_research

## Structure

```text
post/
  posts.csv
results/
  llm/
    <model>/
      evidence.csv
      ratings.csv
  human/
    human_ratings.csv
analysis/
  common.py
  run_sem.py
  run_path_a.py
  run_path_b.py
  compute_rstar.py
  validate_measurement.py
  run_resampling.py
  fit_sem.R
  fit_cfa.R
  requirements.txt
```

## Data

- `post/posts.csv`: 5,715 anonymized posts linked by `post_id`.
- `results/llm/<model>/evidence.csv`: RID, CCB, imputation auxiliaries, and five-construct coverage indicators for each model.
- `results/llm/<model>/ratings.csv`: item-level and construct-level ratings for each model.
- `results/human/human_ratings.csv`: ratings from 23 anonymized raters.

## Analysis

- `run_sem.py`: latent-variable SEM.
- `run_path_a.py`: adjusted composite-score path analysis.
- `run_path_b.py`: pairwise composite-score path analysis.
- `compute_rstar.py`: relationship-pattern concordance.
- `validate_measurement.py`: human-rating agreement, Cronbach's alpha, and CFA.
- `run_resampling.py`: fixed-seed sample-size analysis.

Run scripts from the repository root with Python 3.12, the packages in `analysis/requirements.txt`, R 4.4.3, and `lavaan` 0.7-2.
