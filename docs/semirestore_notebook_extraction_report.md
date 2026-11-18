# SemiRestore notebook extraction and evaluation report

## Source and extraction
The uploaded SemiRestore.ipynb embeds a Base64 ZIP project package. The package was extracted and rebranded as semirestore. The notebook's final evaluation workflow was run locally.

## Dataset summary
- 64 held-out pairs: 2 architectures × 4 difficulty levels × 8 samples
- 128 grayscale PNGs regenerated locally (one reference and one search image per pair)
- 64-row manifest including seeds, true/predicted coordinates, localization error, route, confidence, and image paths
- Unit/smoke tests: 15/15 passed

## Evaluation metrics
- Mean localization error: 11.5085 px
- Median localization error: 9.9259 px
- Accuracy within 5 px: 34.38%
- Mean pipeline latency: 140.0291 ms
- Verifier ROC-AUC: 0.9737419; average precision: 0.5342083
- Verifier recall at threshold 0.5: 25.0% (7 TP, 21 FN)
- OOD: 16 synthetic pairs; mean error 19.3556 px; accuracy within 5 px: 0%

## Provenance boundary
The reconstructed images are procedural SEM-like DRAM/FinFET scenes, not real microscope captures. The nine public SEM entries are registry references only. No third-party real SEM image dataset was downloaded. These results are pilot-scale synthetic evidence, not industrial accuracy claims.

## Commit scope
This Git commit adds the compressed 64-row dataset manifest, dataset provenance README, and a report. The 128 PNG image-pair binaries and full project/notebook ZIP are not included in this commit; they remain in the local release artifacts.