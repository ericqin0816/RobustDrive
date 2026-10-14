# Data and provenance

Real recordings belong in `data/raw/driving_log.csv` with `data/raw/IMG/`.
See [the collection plan](../docs/data_collection.md). Keep an unmodified local
backup. Never commit full datasets or simulator binaries.

Run `python scripts/inspect_dataset.py --data data/raw` to inspect usable rows,
camera counts, missing/corrupt images, steering/speed distributions, and nearly
straight examples (`abs(steering) < 0.05`). Side images may be absent; evaluation
requires center images. Steering outside [-1, 1] is invalid.

Moved Windows/POSIX paths resolve through the recording's `IMG/` folder. Use
`--image-root PATH` if images are elsewhere. Repeated references to an identical
image path are rejected. Preserve row order and camera grouping when cleaning.

`data/sample/` contains generated synthetic images and a `dataset_metadata.json`
marker. The loader carries synthetic provenance into runs even if a config claims
real data. The generator will not overwrite an existing CSV. Synthetic performance
is software evidence only.

Preprocessing is done in memory. `data/processed/` is optional derivative storage.
Data fingerprints include CSV bytes and readable image bytes, so changes require
a new experiment. Move the whole recording without rewriting the CSV when possible.

Future model distribution can use selected GitHub Release assets, Git LFS, or a
model-hosting service. None is configured or required now.
