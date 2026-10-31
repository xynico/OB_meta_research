# Data structure

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

## Files

- `post/posts.csv`: 5,715 posts identified by `post_id` (`P0001`–`P5715`). It contains the post title and post text only.
- `results/llm/<model>/evidence.csv`: five-construct relevant information density (`rid`), construct coverage breadth (`ccb`), and construct-coverage indicators.
- `results/llm/<model>/ratings.csv`: item-level and overall scores for remote-work intensity, autonomy, stress, job satisfaction, and leader–member exchange.
- `results/human/human_ratings.csv`: ratings from 23 anonymized raters (`S01`–`S23`) for 63 materials (`M001`–`M063`) representing 62 unique posts.

All files are linked by `post_id`. Blank cells represent missing ratings.

## Analysis

The scripts require Python 3.12 with the packages in `analysis/requirements.txt`, and R 4.4 with `lavaan`.
Run commands from the repository root.

Install the dependencies:

```bash
python -m pip install -r analysis/requirements.txt
Rscript -e 'install.packages("lavaan", repos="https://cloud.r-project.org")'
```

Run the complete 0.00-0.90 threshold grid for latent SEM, Path A, and Path B:

```bash
python analysis/run_sem.py
python analysis/run_path_a.py
python analysis/run_path_b.py
python analysis/compute_rstar.py output/sem.csv output/path_a.csv output/path_b.csv
```

Use `--models`, `--methods`, `--start`, `--stop`, and `--step` to run a subset. For example:

```bash
python analysis/run_sem.py --models qwen3-32b --methods mean --start 0.4 --stop 0.4
```

Reproduce the measurement validation:

```bash
python analysis/validate_measurement.py
```

Run the fixed-seed sample-size analysis. The default sample sizes are 50, 100, 500, 1000, 2000, 3000, 4000, and 5000, with 20 repetitions and the full threshold grid. Samples are nested within each repetition and shared across models. The script writes threshold-level SEM/Path A/Path B `r*` results and a second `_measurement.csv` file containing Cronbach's alpha and CFA results:

```bash
python analysis/run_resampling.py --models qwen3-32b --jobs 4
```

Mean and MICE-PMM are implemented independently. MICE-PMM uses five imputations by default. The full-grid analyses use seed `20261013`; resampling uses master seed `20261016`.
