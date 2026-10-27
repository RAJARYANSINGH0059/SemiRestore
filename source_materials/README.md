# Source materials and provenance

The repository was assembled from files supplied by the project owner in the conversation.

- `NOVA_original_uploaded.ipynb`: the original uploaded Colab orchestration notebook, preserved as supplied.
- `NOVA_original_backup.ipynb`: a second preserved copy used during engineering.
- `original_archives/NOVA-Loc-source.zip`: source-code bundle from the user.
- `original_archives/NOVA-Loc-unified-dataset-factory.zip`: factory bundle from the user.
- `../docs/supplied/RESULTS.md`, `experimental_protocol.md`, `math.md`, and `engineering_acceptance_requirements.txt`: supplied results, research protocol, mathematical formulation, and engineering acceptance request.
- `../docs/assets/wafer_inspection_workflow.jpg`: compressed view of the supplied workflow diagram. The supplied presentation decks were visually audited; their key requirements are summarized in `docs/proposal.md`, `docs/round1_slides.md`, `docs/NOVA_Loc_Test_Report.md`, and the root README.

The 50 MB `NOVA-Loc-prototype.zip` was not duplicated in this repository because its generated PNG image pairs can be rebuilt by `src/generator.py` from the retained ground-truth CSVs and fixed seed. To prepare the pairs: `python scripts/prepare_synthetic_dataset.py`.