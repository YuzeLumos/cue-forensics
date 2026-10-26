# CF anchor, shared joint-within adaptation: fixed standard replay

2026-09-19. Declared before fitting this base and scoring this candidate on the six existing standards. This is a same-update-rule compatibility evaluation on exposed benchmarks, not new confirmation and not a hyperparameter search.

## One fixed configuration

Use the original Table1 CommunityForensics ViT-S/16 checkpoint and its384-dimensional pre-logit representation. Preserve its official linear raw logit as the frozen anchor; do not retrain or replace this anchor with a new classifier. The zero-label score is exactly this anchor's raw logit (old probability output has the same ranking, not necessarily identical numeric score). Apply the official head to original features, never to newly standardized features.

Use the exact existing8000 anchor-training and2000 calibration row allocation. Fit StandardScaler and PCA256(seed20260824) only on the8000 features. Build the original two-class native initial ledger and initial A/W geometry/head on the same8000, cap256 and geometry_lambda10. Freeze kappa by the original freeze_kappa routine on the2000 calibration logits and initial-head scores. Neither calibration labels nor benchmark queries select the anchor, coordinates or a hyperparameter. Preserve all nondegeneracy and positive-kappa checks; failure stops this candidate without a replacement chosen from query scores.

Recompute the base-preserving precision from these CF base statistics using the existing recipe. With normalized initial direction qB, raw second moment RB, covariance CB, D0=trace(RB), kB=qB^T RB qB, beta=D0/(D0+kB), set MB=I-beta qBqB^T, mB=kB/(D0+kB), vB=qB^T CB qB. Set rho_ref=(10+jitter)*vB/mB and lambda_prior=rho_ref/16. The fixed transferred setting is the dimensionless factor1/16, not the DINO numeric lambda393. All CF base values are derived before benchmark evaluation.

Use the unmodified JointWithinCore from joint_within_integration_20260919. J=0 controlled contributions, R=I256. History/current begin empty for each benchmark. Fit one balanced current window with the original16 real+16 fake support, then score its fixed query. Six fits total. K0 and K16 use the same CF anchor. No selected readout, prototype/kernel mixture, per-benchmark rule, temperature search, support order search, fine-tuning, or outcome-based fallback. This standard episode evaluates adaptation with the same solver; it does not by itself establish additional-history benefit.

## Inputs and comparisons

Complete only8271 missing original CF feature rows; reuse1729 existing rows exactly. Bind image manifest, official checkpoint, decoder/transform and fixed32 encoding anchors. Preserve previous caches. The resulting10000-row compact archive supplies only base fitting/calibration.

Reuse the existing3264-row standard feature archive and original fixed support identities. Preserve all previous query exclusions; add any deterministic known identity/byte/pixel collisions with the full new10000-row base. Exclude known support/query aliases, including all available cross-benchmark links. Bind masks before scoring and apply identical masks to all references. Do not search for a more favorable query subset. Metadata audit documents coverage limits. If a support identity was already in base, retain the original support allocation and explicitly report repeated/current-new label counts rather than falsely calling every support label new.

Compare with all22 existing released-checkpoint scores, official CF anchor, and legacy CF-CUE update scores on the same retained queries. Old CF-CUE uses a different update rule and is a separate reference. Keep all baselines, not only the best ones. The comparison matches evaluation examples, not encoder pretraining or adaptation label budgets; report these differences.

## Validation and outputs

Before scoring bind code, base, protocol, allocations, feature caches, official-head arrays and query masks. Verify official-head logits against the cached original base logits within the existing2e-4 absolute+relative tolerance; cached base features stay bitwise unchanged. Verify empty-current scores equal the anchor, serialized-state reload equality, solve residuals, and one direct ridge equivalence when history is empty. Do not rerun unrelated integration tests.

Save fitted base, six current heads/states, query scores, all benchmark AUROC/AP/FPR95 and ranks against the22 saved released checkpoints, fixed-anchor and legacy-update differences, and six-benchmark macro means. No averaging or ensembling across independently chosen algorithms. Report how many benchmarks meet/exceed the best compared released AUROC; do not equate this automatically with field-wide SOTA. Keep the original result names and old numbers separate.

For adapted-versus-own-anchor uncertainty use1000 paired identity/known-alias group bootstrap draws with seed20260919, identical weights across the two methods and linked queries. Report conditional query uncertainty, not uncertainty over the one fixed support allocation. No resampling result changes the model.

Save training/calibration/current labels separately, numeric and serialized state sizes (shared encoder/base distinguished), fitting and scoring time. Stop after this fixed replay whether positive or negative. No manuscript edits or inheritance of old headline scores.
