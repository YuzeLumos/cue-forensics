# CUE: final joint-moment CPU release candidate

This candidate packages the final `JointWithinCore` used for **Evidence Is All
You Need—But Not Forever: Open-World Continual Image Forensics**. It preserves
the original estimator mathematics and the evidence controller. New wrappers
provide stable imports, explicit inputs, fresh output directories, and portable
checksums. The old `CueFDEvidenceScorer` experiment entry is not used.

This is an author-review candidate. The authors have not yet selected an
original software license; see `NOTICE` before public redistribution.

## Install and run

Use Python 3.11 or later. The verified runtime is Python 3.13.5 with the versions
in `requirements-tested.txt`. Only CPU numerical dependencies are required.

```text
python -m pip install ".[test]"
python -m cue_joint --help
python -m cue_joint demo --output-dir runs/synthetic
python -m cue_joint inspect --state runs/synthetic/state.json
python -m cue_joint score --state runs/synthetic/state.json --input runs/synthetic/query.npz --output-dir runs/rescored
python -m cue_joint frozen-demo --assets-dir assets --output-dir runs/frozen
python -m pytest -q
```

After installation the package works from another working directory. Pass an
explicit path to this repository's `assets/` when using frozen artifacts.
Assets remain companion files; they are not silently copied into the wheel.
Every output command requires a **new** output directory. A repeated output
path is rejected without changing the completed run. Choose another directory
for a repeat; original research records are never deleted or overwritten.

`demo` uses synthetic 256-dimensional projected features and anchor scores.
`frozen-demo` restores the real H0/H32 world-0, seed-20260918, K=4 snapshots,
checks both fitted heads against the saved formal head array, and scores four
synthetic **encoded** feature rows. Its successful result is a serialization
and numerical consistency check, not reproduction of paper accuracy.

## Paper → implementation → artifacts

| Scope | Implementation | Shipped artifact |
|---|---|---|
| Final joint feature–anchor moments, current support, historical within-class scatter | `src/cue_joint/core.py` | Schema `cue-joint-within-core-v1` |
| Window removal/rebuild, empty-current anchor, transactional transitions | `JointWithinCore` and `window` command | Core JSON produced by any run |
| Immutable evidence bank and lifecycle controller | `src/cue_joint/lifecycle.py`, compatibility `cuefd/methods/evidence_lifecycle_v2.py` | Two real DINO H0/H32 snapshots |
| Complete lifecycle restore bound to its exact base | `src/cue_joint/snapshot.py` | `assets/dino/base.pkl`, `integrated_H0_K4.snapshot.json`, `integrated_H32_K4.snapshot.json` |
| Six-benchmark CF configuration | `cf-replay` | `assets/cf/base.pkl`, `geometry.npz`, six final `states/*_core.json` |
| Saved formal H0/H32 fitted heads | `frozen-demo` consistency check | `assets/dino/heads.npz`, `HEAD_INDEX.csv` |

The `cuefd` namespace is a small frozen compatibility subset for the original
pickled projection/base and controller. Its legacy classes do not replace the
final `cue_joint.core.JointWithinCore` solver. `SOURCE_MANIFEST.json` records each
copied file's logical source root, source-relative name, original SHA-256,
candidate SHA-256 and any import-only edits. `core.py` is byte-identical to its
formal research source. Original protocols are under `provenance/`.

## Feature and state interfaces

All NPZ files load with `allow_pickle=False`. Use numerical arrays and Unicode
sample IDs, not Python object arrays.

| Command | Input arrays | Behavior |
|---|---|---|
| `init --input config.npz` | `metric[256,256]`, scalar `kappa`, scalar `lambda_prior`, optional orthonormal `basis[256,r]` | Explicit frozen configuration, empty current/history |
| `score --state state.json --input query.npz` | `projected_features[N,256]`, `anchor_scores[N]` | Final-core score using already standardized/projected features |
| `window add-history/replace-current/advance --input support.npz` | Above two arrays plus `labels[N]`, `sample_ids[N]` | Balanced binary support, unique provenance IDs |
| `window forget --window-id ID` | Existing state only | Remove that retained current/history window and rebuild |
| `cf-replay --benchmark NAME --input cf.npz --assets-dir assets` | `features[N,384]` | **Official CF pre-logit features**, saved benchmark-specific final head |
| `dino-replay --input dino.npz --assets-dir assets` | `features[N,1102]` | Original encoded-feature representation, two saved H0/H32 K=4 lifecycle snapshots |

Every writing command also takes `--output-dir`. Window commands additionally
require `--state` and `--window-id`. `advance` requires one explicit choice:
`--archive-current` or `--discard-current`. The wrapper stages a new solved core;
failure leaves the input state and completed outputs unchanged.

For example, replace the current synthetic support, then remove it:

```text
python -m cue_joint window replace-current --state runs/synthetic/state.json --input runs/synthetic/support.npz --window-id current0 --output-dir runs/replaced
python -m cue_joint window forget --state runs/replaced/state.json --window-id current0 --output-dir runs/forgotten
```

Forgetting is **retained core-statistics removal**. It does not erase historical
controller audit posteriors or ledgers, retrain an encoder, or establish that the
entire system never encountered a sample. The lifecycle adapter documents this
boundary and preserves atomic core/controller updates.

## What full paper reproduction still needs

This candidate includes working model-state inference and support-update tools,
not a new image-encoding or 384-fit benchmark execution claim.

- Pixel-to-feature reproduction requires the exact external CommunityForensics
  encoder/checkpoint and official 384-dimensional pre-logit preprocessing, or
  the original DINO/forensic 1102-dimensional encoding recipe. Encoder weights,
  image datasets, image decoders and their feature caches are not bundled.
- Recomputing the six reported benchmark metrics requires the original
  support/query identities, exclusions, labels and matching encoded rows.
  `cf-replay` scores supplied feature rows using the existing fitted head; it
  does not select new supports or independently regenerate the reported AUROC.
- Recomputing the complete H0/H32 budget curves requires the original four
  worlds, six support-order seeds, eight budgets, feature/identity allocation,
  histories, sensor calibration and bootstrap groups. Only the two stated K=4
  complete snapshots are shipped. The full `heads.npz` is retained for provenance
  and checking; `dino-replay` explicitly uses the two complete snapshots.
- Previously reported numbers in the shipped CSV/JSON files are historical
  research records. Local release verification uses synthetic probes and checks
  saved-head equality; it does not independently verify those performance claims.

Pickle files are loaded only after a matching checked-in source-manifest hash.
Treat pickles from other sources as executable Python input. The compatibility
runtime is pinned because scikit-learn pickle restoration is version-sensitive.
The manuscript's statistical and feature-space assumptions remain applicable;
the wrapper adds no new generalization, unlearning or performance guarantees.

## Verification

`tests/` retains the original moment/controller unit tests and adds final-core
deletion/rebuild, atomic failure, install/import, relocated-asset, repeat-output,
real checkpoint/head equality, tamper rejection and six saved-state scoring
checks. `VERIFY.json` records the exact local verification and its limits.
