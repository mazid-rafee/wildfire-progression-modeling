# Claims Lacking Sufficient Experimental Support

Do **not** invent comparisons, citations, geographic areas, or additional results. The list below flags statements that are **not** backed by the frozen V2 20-cell matrix / validation audit.

---

## Hard unsupported (omit or heavily hedge)

| Claim | Why unsupported |
|---|---|
| Untouched / blind external holdout performance | Explicitly not completed; in-corpus candidates ineligible |
| Definitive statistical significance for real-world deployment | CIs are within a reused 21-fire exploratory set |
| E1 is temporal attention / transformer / input-adaptive | Fusion weights are globally learned, content-agnostic |
| V2 improves over V1 by X F1 points | No controlled V1↔V2 multi-seed retrain matrix in this package |
| SpaSE TS=6 mean-pool multi-seed comparison | Archival / prior audit only; not a 5-seed headline arm |
| Physical area skill in km² | Only pixel counts on ≈0.00337° EPSG:4326 grid |
| Georeferenced cartographic accuracy of qualitative maps | Image-coordinate center crops |
| Regional superiority (e.g., “works in the Rockies / fails in the Southwest” as a general law) | Only event-level observations on listed US_2021 IDs |
| ENV-group necessity (wind / terrain / veg / meteo / drought individually) | Multi-seed matrix includes ENV-0 aggregate mask only, not the finer ENV-* arms |
| Operational early-warning readiness | No latency, coverage, or agency-workflow evaluation |
| Seed-42 E1−UNIFORM as identical-init causal effect | Unpaired initialization |
| Dropping the five difficult fires yields the “true” model ranking | Subgroups are secondary; full-21 is primary |
| Positive-target-only macro replaces full-21 macro | Supplementary only; empty-target convention must stay disclosed |
| Correlations of F1 with fire size / progression are causal | Prespecified descriptive associations only |
| Competing published SOTA numbers | No external method reimplementation under V2 locks in this package |

---

## Supported with required caveats

| Claim | Caveat |
|---|---|
| E1 > TS2 / ENV0 / UNIFORM on micro F1 (5 seeds) | Exploratory reused test; 21 fire clusters |
| Learnable fusion beats frozen-uniform | Primary evidence = paired seeds 7/17/29/101 |
| Micro ≫ macro for E1 | Weighting + five BA-sparse events; not a bug |
| ENV-0 can beat E1 on some windows | Qualitative/window-level; average still E1 > ENV-0 |
| Newest-frame weight dominance in E1 | Observed in frozen run summaries; mechanism is correlative with gains |

---

## Citation / geography inventiveness ban

Do not add:
- Paper titles, venues, or years not verified by the authors
- Fire names / NFIRS identifiers beyond the `US_2021_*` IDs in the manifests
- States or regions as experimental strata unless derived from those IDs and labeled descriptive
- Extra metrics (AUROC, frechet, perimeter IoU, etc.) not present in frozen eval outputs
