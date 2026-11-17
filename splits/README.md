# Dataset split manifests

The paper uses nested cross-validation rather than one permanent train/test
split. These files preserve the exact split used by this reproduction.

- `outer_fold_assignments.csv`: one row per source encounter. For outer fold
  `k`, `outer_test_fold == k` is the held-out test set and all other rows are
  that fold's training set.
- `inner_fold_assignments.csv`: one row per encounter in each outer training
  set. For outer fold `k` and inner fold `j`,
  `inner_validation_fold == j` is validation and the other rows are inner
  training data.
- `split_summary.json`: seeds, fold sizes, positive counts and the source CSV
  SHA-256.

`source_row_index` is zero-based and refers to the data rows of
`data/raw/diabetic_data.csv`, excluding the CSV header. The source dataset is
not committed; obtain it with:

```bash
conda run -p ./.conda python scripts/download_data.py
```

Regenerate and verify the manifests with:

```bash
conda run -p ./.conda python scripts/export_splits.py
```

The outer seed is 42. Inner seeds are 43, 44, 45, 46 and 47 for outer folds
1–5 respectively, matching `reproduce.py`.

