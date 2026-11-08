# Remaining Work Before Journal / Conference Submission

Analysis-only package is complete for outlining. The items below are **still required** before a credible submission. None of these were executed as part of this prep step.

---

## Blockers (publication integrity)

- [ ] **External untouched holdout** locked *before* inference, **or** explicit venue-acceptable framing that all test results are an exploratory reused benchmark (already drafted; authors must decide venue tolerance).
- [ ] **Author-curated related-work citations** (none invented here).
- [ ] **Final claim audit** against `05_UNSUPPORTED_CLAIMS.md` after prose drafting.
- [ ] **Seed-42 unpaired** language present in every place E1−UNIFORM causal claims appear.
- [ ] **Micro and macro both reported** in main results with the weighting explanation.
- [ ] **Full-21 primary population** retained; manuscript-16 / Rule-A 18 only as secondary subgroups.
- [ ] **V1 vs V2** distinction stated; no mixed-protocol score tables.

## Strongly recommended before submission

- [ ] Plot Figure E (temporal weights) from `TABLE_S4` for main or supplement.
- [ ] Select final 6–8 qualitative panels from `08_QUALITATIVE_CASE_INDEX.md`; verify PNG/PDF readability at column width.
- [ ] Add empty-target supplementary paragraph + `TABLE_S1` to supplement.
- [ ] Document compute (GPU type, wall time) from run logs if required by venue.
- [ ] License / data-access statement for TS-SatFire / VIIRS / BA / FirePred sources.
- [ ] Code + config + SHA256 checklist in a reproducibility appendix.
- [ ] Copy-edit abstract numbers against `00_NUMERICAL_VERIFICATION.md` one final time.
- [ ] Decide whether finer ENV-* ablations (if any historical single-seed runs exist) are cited as prior exploratory work or omitted (multi-seed matrix does not include them).

## Optional (not required for an honest methods paper)

- [ ] Controlled V1↔V2 retrain matrix (expensive; only if claiming protocol-correction gains).
- [ ] Additional seeds beyond five.
- [ ] Perimeter-aware or area-weighted metrics (would need new frozen eval—do not retune thresholds).
- [ ] Human expert qualitative rating study.

## Explicitly out of scope for “remaining prep” (do not do casually)

- Retraining, checkpoint edits, threshold retuning on test
- Dropping difficult fires from headline tables
- Constructing a “holdout” from Phase-1 missing train IDs or zero-window 2021 folders
- Git commit/push of this package unless authors request it
