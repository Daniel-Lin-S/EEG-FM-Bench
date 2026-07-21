# miniROCKET baseline

This baseline uses the upstream multivariate miniROCKET implementation at <https://github.com/angus924/minirocket>. The upstream source is GPL-3.0, so it is not copied into this Apache-2.0 repository.

Clone miniROCKET outside this repository and install the baseline extras:

```bash
git clone https://github.com/angus924/minirocket /path/to/minirocket
pip install -r requirements/feature_extractors.txt
```

Set the clone root in the baseline configuration:

```yaml
model:
  minirocket_source_path: /path/to/minirocket
```

The trainer loads
`/path/to/minirocket/code/minirocket_multivariate.py`, uses raw `float32` EEG with shape `(trials, channels, timepoints)`, and fits miniROCKET only on the training split. It then selects the Ridge regularization strength using the fixed validation split and reports the selected classifier on test data.

`minirocket_num_features` defaults to `10000` and `minirocket_max_dilations_per_kernel` defaults to `32`. Both are configurable under `model`.
