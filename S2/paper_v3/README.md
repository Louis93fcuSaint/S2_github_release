# Paper v3: code for the revised manuscript

This folder contains the code used for the revised Chapter 4 and for the
specification-conditioned rotor design pipeline described in the manuscript.
It is a code and reproducibility snapshot. The full raw dataset, the
candidate pools, and the per-slot ROSS records are not redistributed here.

## 1. Repository layout

| Path | Contents |
|---|---|
| `S2/spec_design/` | v1 latent DDPM: material plus first-order point target, decoder, ROSS guard |
| `S2/spec_design_v2/` | v2 point/interval conditioned DDPM (`v2m`, `v2i`), constraint decoder, surrogate guidance, `s2_design.py` entry point |
| `S2/surrogate_v47/` | 47-feature critical-speed surrogate: training, inference, five-member ensemble |
| `S2/DDPM_v47/` | diffusion modules shared by the latent DDPM |
| `S2/CVAE_baseline/` | conditional VAE baseline used in the DDPM/CVAE comparison |
| `S2/paper_v3/pascal_revision/scripts/` | frozen TST protocol, pool generation, shortlists, ROSS acceptance, final surrogate CV, figures |
| `S2/paper_v3/pascal_revision/tst_runs/00_protocol/` | frozen specification and evaluation protocol (JSON, addendum, hashes) |
| `S2/paper_v3/pascal_revision/results_snapshot/` | small CSV/JSON summaries quoted in the revision |
| `S2/paper_v3/research_generators_specification_study/eaai_comparative_study/audit_scripts/` | shared comparison pipeline: pool generation, shortlists, ROSS runner, accounting, summaries |
| `S2/paper_v3/research_generators_specification_study/eaai_comparative_study/spec_candidates/` | calibration/test specification candidates and their hash list |
| `S2/paper_v3/research_generators_specification_study/10_reproducibility/` | first-batch analyses: inventory, specification difficulty, four-way re-aggregation, diversity, screening |
| `S2/paper_v3/chapitre4_revision/scripts/` | disk-edge geometry screen, accepted-set diversity, Chapter 4 verification and document build |
| `S2/paper_v3/chapitre4_revision/results_snapshot/` | `tst_geometry_acceptance.csv`, `accepted_set_diversity.csv` |

## 2. Pipeline

1. **Frozen specifications.** `tst_runs/00_protocol/tst_protocol_frozen.json` and
   `research_generators_specification_study/eaai_comparative_study/spec_candidates/test_specs.csv` define the
   specifications before any candidate is generated.
2. **Candidate generation.** `pascal_revision/scripts/run_tst_pools.py` produces
   5,000 attempts per specification, method, and sampling seed. The interval
   DDPM (`v2i`), the point DDPM (`v2m`), the CVAE, and retrieval+jitter share
   the frozen specification list but not their training data or checkpoints.
3. **Shortlists.** `build_tst_shortlists.py` draws a uniform sample of 100
   attempts and builds a 200-design shortlist. The shortlist is produced by the
   common surrogate gate followed by uniform selection among predicted in-band
   unique designs and a deterministic out-of-band fallback ordered by
   `(violation, candidate_id)`. It is neither a top-200 score ranking nor a
   farthest-point sample.
4. **ROSS acceptance.** `run_tst_ross_full.py` re-computes every submitted
   design with ROSS 1.6.1. Denominators stay at 100 and 200; errors, timeouts,
   and missing records count as non-delivery. Results are cached only under an
   exact design, material, unit, solver-version, and acceptance-protocol key.
5. **Geometry screen.** `chapitre4_revision/scripts/compute_geometry_diversity.py`
   applies an axial disk-edge check: the distance from a bearing centre to the
   nearest disk end face must be at least 10 mm. This is a one-dimensional
   check, not a full assembly or manufacturing check. H4 itself is the
   centre-to-centre distance rule enforced by the decoder and is therefore
   passed by every decoded design.
6. **Diversity.** The same script reports configuration-family coverage and
   within-family distances on designs that pass both ROSS and the disk-edge
   screen. Distances use active canonical parameters and scales fitted on the
   frozen reference library. TST07 has no common sample across methods and is
   reported without a distance estimate.

## 3. Protocol and results quoted in the revision

The Chapter 4 comparison uses five frozen specifications (TST01, TST02, TST04,
TST07, TST09), three methods, and sampling seed 42. Each cell has 5,000
generation attempts, a 100-design uniform sample, and a 200-design shortlist.

| Quantity | DDPM | CVAE | Retrieval+jitter |
|---|---:|---:|---:|
| Raw-sample ROSS satisfaction (macro) | 70/500 = 14.0% | 157/500 = 31.4% | 299/500 = 59.8% |
| Shortlist ROSS satisfaction (macro) | 592/1,000 = 59.2% | 678/1,000 = 67.8% | 832/1,000 = 83.2% |
| Joint ROSS and disk-edge acceptance (macro) | 196/1,000 = 19.6% | 235/1,000 = 23.5% | 270/1,000 = 27.0% |

The final surrogate was evaluated with a fixed-configuration five-fold
GroupKFold procedure (groups: material, disk count, bearing count). The
unadjusted fold-mean MAPE is 6.044%; excluding the predefined degenerate
`cs1 < 100 rpm` labels gives 4.579%. The value 4.44% is a descriptive result
from the model-selection set and is not an independent generalization estimate.

The 4,500 logical ROSS slots in the TST subset contain 4,442 unique designs;
165 unique designs reused stored results, and 4,277 required a new solve.

## 4. Running the code

Environment used for the archived runs: Python 3.8.18, numpy 1.24.4,
pandas 2.0.3, scipy 1.10.1, scikit-learn 1.3.2, matplotlib 3.7.5,
PyTorch 2.4.1, ROSS 1.6.1.

The scripts were written for the original Windows workspace and contain
absolute paths of the form `.../S2/...`. Adjust the `BASE`, `S2`, or `PROJECT`
constants at the top of each script if the checkout is placed elsewhere.

Typical order:

```bash
python paper_v3/pascal_revision/scripts/run_final_surrogate_cv.py \
    --out results/surrogate_cv_final --folds 0 1 2 3 4 --threads 6
python paper_v3/pascal_revision/scripts/run_tst_pools.py
python paper_v3/pascal_revision/scripts/build_tst_shortlists.py
python paper_v3/pascal_revision/scripts/run_tst_ross_full.py
python paper_v3/pascal_revision/scripts/analyze_tst_subset_s42.py
python paper_v3/chapitre4_revision/scripts/compute_geometry_diversity.py
python paper_v3/chapitre4_revision/scripts/verify_chapter4.py
```

Candidate generation and ROSS acceptance are the expensive steps. The archived
subset used 12 ROSS workers and about 14,543 s of batch wall time for 4,277 new
solves. Reproduction requires the frozen dataset and checkpoints described
below.

## 5. Checkpoints and data

Included in this repository:

- interval DDPM `v2i`: `S2/spec_design_v2/outputs/v2i/model.pt`
- point DDPM `v2m`: `S2/spec_design_v2/outputs/v2m/model.pt`
- five surrogate members: `S2/surrogate_v47/outputs/mlp_v47_best/models/`
- CVAE baseline: `S2/CVAE_baseline/outputs/cvae_main/model_best.pt` with
  `training_log.json`

Not included: the 199,815-record source dataset, generated candidate pools,
per-slot ROSS tables, and the 5,000-candidate dumps. These are large derived
artifacts and are available from the author on request. The small summaries in
`results_snapshot/` are sufficient to check every number quoted in Chapter 4.

## 6. Notes on interpretation

- ROSS re-computation is numerical verification, not a physical experiment.
- The H1--H6 parameterization and the 10 mm disk-edge screen are not a complete
  assembly or manufacturing feasibility check.
- The TST comparison uses one sampling seed; macro averages are descriptive and
  are not accompanied by significance tests.
- The historical six-specification, three-seed experiment used a different
  shortlist protocol and is not pooled with the TST results.
