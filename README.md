# AMM rotor inverse-design research data

This snapshot accompanies the AMM manuscript on specification-driven rotor inverse design and numerical verification using ROSS 2.3.0.

- data/verification: frozen verification slots, aggregate results, seed comparisons and diversity metrics. Seed 42 covers eight specifications and Seed 7 covers five.
- data/perturbation: the original 50-design, six-scenario study.
- data/design_selection: full-precision A/B/C inputs, scenario results, Figure 6 identities, mode shapes and elastic energy.
- data/stiffness_scan: the separate A/B/C bearing-stiffness sweep, refined boundary and branch tracking, outside the original 300 evaluations.

Previously public inputs matching frozen SHA-256 hashes are listed below. The training CSV was checked after decompression. Use the manuscript specification mapping for test_specs.csv. Previous exploratory results remain in Git history.

- `S1/04_versions/v4.7/dataset_v4.7/dataset_v4.7_all.csv.gz`
- `data/specifications/test_specs.csv`
- `S2/spec_design_v2/outputs/v2i/model.pt`
- `S2/spec_design_v2/outputs/v2m/model.pt`
- `S2/surrogate_v47/outputs/mlp_v47_best/models/multi_seed42.pt`
- `S2/surrogate_v47/outputs/mlp_v47_best/models/multi_seed43.pt`
- `S2/surrogate_v47/outputs/mlp_v47_best/models/multi_seed44.pt`
- `S2/surrogate_v47/outputs/mlp_v47_best/models/multi_seed45.pt`
- `S2/surrogate_v47/outputs/mlp_v47_best/models/multi_seed46.pt`
- `S2/surrogate_v47/outputs/mlp_v47_best/summary_mlp.json`
