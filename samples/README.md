# Samples

A small, diverse set of images (~100) for looking at the data and the models' behaviour, and for
figures in our own report.

```
samples/
  raw/            the images as downloaded, plus manifest.csv
  predictions/    the same images with model predictions drawn on, plus predictions.csv
```

`raw/manifest.csv` uses the same columns as `data/processed/<dataset>/manifest.csv` (`image_id`, `path`,
`dataset`, `brand`, `era`, `group_id`, `split`, ...) so predictions can be joined on `image_id`.
`predictions/predictions.csv` follows `outputs/results/<dataset>/predictions/*.csv`
(`image_id`, `<task>_true`, `<task>_pred`, `<task>_conf`), with one more column, `model`.

Images are not committed (Fashion-Gen is research-only and the Europeana runway photos are In Copyright);
only the CSVs and this README are tracked.
