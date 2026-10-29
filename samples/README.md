# Samples

Five Fashion-Gen **test** images (never seen in training), for looking at the data and the model's
behaviour, and for figures in our own report.

```
samples/
  raw/            the images, plus manifest.csv (brand, season, category, MSRP, product name)
  predictions/    the same images with the model's output drawn on, plus predictions.csv
```

The model is given the garment category and predicts brand, season and MSRP. In the annotated
images green means correct (for MSRP: within 25% of the true price) and red means wrong.
`predictions/predictions.csv` has `image_id`, `model`, `category` (the input), and `<task>_true`,
`<task>_pred`, `<task>_conf` per task. Regenerate with `scripts/predict.py` and a checkpoint, e.g.
`--inputs-csv` with each file's category.

Images are not committed (Fashion-Gen is research-only); only the CSVs and this README are tracked.
