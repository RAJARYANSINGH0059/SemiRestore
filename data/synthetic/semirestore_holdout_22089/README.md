SemiRestore synthetic holdout dataset

This dataset metadata describes 64 held-out pairs: 2 architectures (DRAM and FinFET) × 4 difficulty levels × 8 samples per combination. Search images are 120 × 120 grayscale pixels; reference images are 100 × 100.

Provenance: all examples are procedurally generated SEM-like scenes. They are not real SEM microscope captures and were not downloaded from REFICS, Carinthia, Zenodo, EUDAT, NFFA-EUROPE, Bath, or other external dataset providers. The nine external sources remain registry references; the downloader is a scaffold.

The manifest preserves seed, architecture, difficulty, ground-truth and predicted coordinates, error, confidence, route, and the relative reference/search image paths. The PNG image-pair binaries are not included in this Git commit; they remain in the local semirestore_dataset.zip release artifact.