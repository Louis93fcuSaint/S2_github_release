# AMM rotor inverse-design research data

This snapshot accompanies the AMM manuscript on specification-driven rotor inverse design and numerical verification using ROSS 2.3.0.

- data/training: compressed rotor geometry and critical-speed training data.
- data/models: point-target DDPM, interval and one-sided-bound DDPM, and the five-member multi-output MLP ensemble.
- data/verification: frozen verification slots, aggregate results, seed comparisons and diversity metrics. Seed 42 covers eight specifications and Seed 7 covers five.
- data/perturbation: the original 50-design, six-scenario study.
- data/design_selection: full-precision A/B/C inputs, scenario results, Figure 6 identities, mode shapes and elastic energy.
- data/stiffness_scan: the separate A/B/C bearing-stiffness sweep, refined boundary and branch tracking, outside the original 300 evaluations.

Previously public inputs matching frozen SHA-256 hashes are listed below. The training CSV was checked after decompression. Use the manuscript specification mapping for test_specs.csv. Previous exploratory results remain in Git history.

- `data/training/rotor_critical_speed_dataset.csv.gz`
- `data/specifications/test_specs.csv`
- `data/models/ddpm/interval_and_bound_ddpm.pt`
- `data/models/ddpm/point_target_ddpm.pt`
- `data/models/surrogate/multi_output_mlp_seed42.pt`
- `data/models/surrogate/multi_output_mlp_seed43.pt`
- `data/models/surrogate/multi_output_mlp_seed44.pt`
- `data/models/surrogate/multi_output_mlp_seed45.pt`
- `data/models/surrogate/multi_output_mlp_seed46.pt`
- `data/models/surrogate/multi_output_mlp_summary.json`

The archived protocol and evidence tables retain the original run and record identifiers so they can be joined to the frozen results. In those records, the point-target DDPM is v2m and the interval and bound DDPM is v2i. S1–S8 in the manuscript are specification identifiers.
