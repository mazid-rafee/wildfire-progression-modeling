# Same-Source Later-Year vs Genuinely External Evaluation

These are **different claims**. Do not market one as the other.

---

## 1. Same-source later-year test

**Definition:** Events processed with the **same sensor stack, BA semantics, FirePred recipe, crop, and channel map** as TS-SatFire / V2, but with **ignition years after the training cutoff** (train/val ≤2020; development test = 2021).

| Property | Implication |
|---|---|
| Distribution shift | Mainly **temporal** (climate year, fuel dryness, management) |
| Label process | Ideally identical BA band-8 rules |
| Model I/O | Frozen 27-ch V2 weights apply directly |
| Independence | Requires events **never used** in development and preferably never inspected |
| On-disk 2022 AF clips | Same-source **layout**, but AF-test contamination + n≈12 → **not** a credible primary holdout |

**Honest paper phrasing:**  
*“Later-year evaluation under the same data-generation protocol.”*

---

## 2. Genuinely external-dataset evaluation

**Definition:** A corpus built by **different authors / sensors / label definitions / grids**, e.g. Mesogeos (Mediterranean EFFIS/MODIS drivers) or SeasFire (seasonal 0.25°).

| Property | Implication |
|---|---|
| Distribution shift | Temporal **and** geographic **and** sensor/label |
| Model I/O | Frozen V2 weights **do not** consume native tensors |
| Required work | New adapter **and usually new training**, or a severely degraded proxy |
| Claim strength | Stronger “external validity” **if** the task is re-defined fairly |
| Risk | Hidden incomparability (resolution, daily vs 8-day, BA definition) |

**Honest paper phrasing:**  
*“Cross-dataset evaluation,”* not “V2 holdout F1 = …”

---

## 3. Mapping current options

| Option | Bucket | Use in V2 paper? |
|---|---|---|
| Reconstructed 2022+ CONUS cubes (Class A) | Same-source later-year | **Yes — preferred independent holdout** |
| TS-SatFire 2022 named AF clips | Same-source, contaminated, tiny | Appendix smoke check only |
| Missing-13 / zero-window 2021 | Neither | Never |
| Mesogeos / SeasFire | External dataset | Separate study / not frozen V2 metrics |

---

## 4. What would undermine either claim

- Calling AF-contaminated 2022 clips “untouched external holdout”  
- Reporting Mesogeos scores as if they were V2 micro F1 on BA progression  
- Building later-year cubes then **dropping hard fires after inference**  
- Mixing `mask_zero` training with `include_as_forecast` holdout inputs  
