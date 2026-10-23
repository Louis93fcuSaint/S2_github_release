# TST ROSS subset protocol addendum (seed 42)

Created: 2026-10-22 (Asia/Shanghai)
Status: frozen before the subset ROSS run

## Relation to the full protocol

The original TST_v1 protocol and its 48-pool / 14,400-logical-slot design remain unchanged. This addendum does not replace or reinterpret that protocol. It records a time-bounded, explicitly partial run for the 2026-10-23 discussion.

The full run was stopped after the QA repeats and 165 cache entries because 14,205 unique designs at roughly 48-66 s per design with 14 workers would require well over 12 h. The subset is not evidence for the complete eight-specification, two-seed comparison.

## Frozen subset

- Specifications: TST01, TST02, TST04, TST07, TST09 (5 specifications)
- Methods: interval-conditioned DDPM, CVAE, retrieval+jitter (3 methods)
- Sampling seed: 42 only
- Lists per pool: 100 uniformly sampled raw-pool candidates + 200 shortlist candidates
- Logical slots: 5 x 3 x 300 = 4,500
- Denominators remain 100 and 200. No renormalisation for invalid generation, solve failure, timeout, duplicate designs, or missing shortlist entries.
- Same frozen surrogates, shortlists, ROSS 1.6.1 protocol, cache key, and acceptance evaluator as the full protocol.
- Cached ROSS values from the stopped full run are reused only when the cache key (design vector, material, disk count, bearing count, solver protocol) matches exactly.

## Specification coverage

| ID | Material | Family | Orders | Condition type | Reference support |
|---|---|---|---|---|---|
| TST01 | Steel | free | 1 | one-sided q95 | high (3,319 feasible) |
| TST02 | Aluminum | 3d-3b | 1 | interval | medium (337) |
| TST04 | Steel | 6d-4b | 1,2 | point (joint) | low (14) |
| TST07 | Steel | free | 1,2,3 | point (joint) | low (10) |
| TST09 | Titanium | free | 1,2 | interval (joint) | medium (555) |

The omitted specifications are TST03, TST05, TST06, and TST08. Their omission is a compute-budget decision fixed before this subset run, not a selection based on ROSS outcomes.

## Interpretation limit

Results from this subset may be reported only as a descriptive five-specification, one-sampling-seed partial validation. Specification is the comparison unit; sampling seed 42 is a sampling replicate, not an independent training replicate. No claim of complete method ranking, statistical significance, or generalisation to unseen specifications follows from this subset.

## Script and protocol hashes

- scripts/run_tst_ross.py: f9c3421eeceadad3299c845ed7419353a270d7c0b6bbfbb3f1e8b16bdf0044d7
- tst_protocol_frozen.json: d44db35faa9a5af60e1bd96d945603ebbee95ec3360a9bbef0265c01e4f464c3
- test_specs.csv: c5a59e43f2c846ffcbb73b0c78f1c001e2e7ed5c930a91bb4ace3d5bb0a348d5

## Reproduction command

Set TST_ROSS_WORKERS=12, TST_SPEC_FILTER=TST01,TST02,TST04,TST07,TST09 and TST_SEED_FILTER=42, then run `python scripts/run_tst_ross.py`.
