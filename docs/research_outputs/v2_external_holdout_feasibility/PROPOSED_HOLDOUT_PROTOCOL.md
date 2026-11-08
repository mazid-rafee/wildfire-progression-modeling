# Proposed Locked Holdout Protocol (For Approval — Do Not Run Yet)

This protocol is a **proposal**. No inference may occur until authors approve a frozen `HOLDOUT_MANIFEST.json` with content hashes.

---

## 0. Scope classes (choose one; do not conflate)

| Class | Definition | Paper claim allowed |
|---|---|---|
| **A. Blind reconstructed later-year** | 2022+ CONUS events newly cubed; IDs never in exclusion registry; never inspected for model scores | “Independent later-year holdout under V2 locks” |
| **B. Contaminated same-source mini-set** | TS-SatFire 2022 AF named clips | At most “same-source smoke check; not independent” |
| **C. External foreign dataset** | Mesogeos / SeasFire / etc. | Only with a **new** model or heavy adapter — out of scope for frozen V2 weights |

**Primary recommendation:** pursue **Class A**. Class B is optional appendix only. Class C is a separate paper.

---

## 1. Exclusion registry (hard rejects)

Reject any event that appears in `artifacts/EXCLUSION_REGISTRY.json`:

- Phase 4B train (82) / val (8) / test24 (21) fire IDs  
- Official US_2021 roster including 3 zero-window folders  
- Phase-1 missing-13 train-roster IDs  
- All 17 TS-SatFire AF named showcase fire folder names  
- Any ID listed in prior audit decision sets used to design V2 eligibility  

Additional rejects:

- Events whose ROIs substantially overlap a registry fire (propose: centroid < 25 km **and** temporal overlap within 30 days — finalize before lock)  
- Events previously opened for qualitative debugging of V2 models  

---

## 2. Deterministic eligibility (Class A cubes)

Mirror locked V2 matched TS=6 rules:

1. Sorted VIIRS_Day files with parseable `YYYY-MM-DD_VIIRS_Day.tif`  
2. For label index \(L\): \(L \ge 6\) and \(L < N\) → need **≥ 7** day files  
3. Inputs = days \([L-6,\ldots,L-1]\); **label day L excluded from X**  
4. BA band present (VIIRS_Day band 8) on last input and label day  
5. Shared V2 geo-align + center crop; \(H,W \ge 256\)  
6. **BA-labelled-both:** any burned pixel in crop on BA at \(L-1\) and \(L\)  
7. Target = \(\mathrm{clip}(BA_L - BA_{L-1}, 0, 1)\)  
8. Calendar gaps: **record** `max_gap`, `n_gaps>1`; default **do not filter** (match Phase 4A) unless a pre-registered gap cap is approved before lock  
9. FirePred missing → zero-fill (document); for primary full-channel arms require FirePred present on all six input days  
10. Forecast policy: **`mask_zero`** on channels 22–25  
11. Normalization: frozen `outputs/v2_protocol/fp_stats_matched_v2.npz` + SAT constants — **never refit**

### Missing data
- Missing Night → zero night channels (existing V2 behavior)  
- Missing FirePred on any required input day → **exclude window** from primary full-channel eval; optional ENV-0 secondary pool may keep with disclosure  
- Undersized raster → exclude  

### Empty BA progression
- Windows with BA-labelled-both but \(TP+FN=0\) (no positive target pixels) **remain** in the population  
- Report F1 with `1e-8` convention; provide empty-target false-alarm supplement (same policy as manuscript prep)  
- **Do not** drop empty-target fires after seeing model scores  

### Event exclusions after lock
- Only for documented **data corruption** (unreadable GeoTIFF, CRS failure) with a signed erratum  
- Never for low F1, sparse BA, or geographic inconvenience  

---

## 3. Metrics and models (frozen)

- Arms: E1, ENV-0, TS2, UNIFORM — all five seeds’ frozen checkpoints  
- Thresholds: **reuse** each cell’s validation-selected threshold from multi-seed runs — **no retune**, including no val re-sweep on holdout  
- Metrics: micro F1/IoU/P/R; unweighted per-fire macro F1; nested fire-cluster bootstrap if \(n_{\text{fires}} \ge 10\)  
- Primary population: all BA-eligible windows/fires passing §2  
- Secondary: positive-target-only macro; ENV-0-only if FirePred incomplete  

---

## 4. Manifest lock format (before any inference)

Create `HOLDOUT_MANIFEST.json` containing:

```text
protocol_id, class (A|B), created_utc, creator
exclusion_registry_hash
event_list[]: fire_id, source_catalog_id, bbox, date_start, date_end, cube_root_uri, cube_sha256
window_list[]: target_key, fire_id, label_idx, label_date, input_dates, eligibility_flags
stats_path + sha256 (fp_stats_matched_v2.npz)
forecast_policy: mask_zero
checkpoint_table: arm, seed, ckpt_sha256, selected_threshold
statement: "No model outputs were computed before this lock."
```

After lock, only a dedicated eval script may load the manifest; results write under `outputs/v2_external_holdout_eval/` (new; not yet created).

---

## 5. Adapter design (implementation after approval)

Keep frozen:

- Model weights, architectures, thresholds  
- V2 geo-align / directional-aug code paths for train parity (eval: no aug)  
- Channel map and `mask_zero`  

Add (new files only):

- Holdout cube root config (writable path **≠** DATA_ROOT)  
- Manifest schema validator  
- `build_holdout_rows(manifest) → rows`  
- `scripts/v2_holdout_eval.py` wrapping `preload_matched_windows_v2` + `evaluate_windows` without population asserts  

---

## 6. Ready-to-lock checklist

- [ ] Class A event list chosen from MTBS/FEDS **without** model scores  
- [ ] Cubes built and SHA256 recorded  
- [ ] Eligibility pass tabulated  
- [ ] Exclusion registry diff empty  
- [ ] Manifest signed off by authors  
- [ ] **Then** run inference  

**Current status:** checklist incomplete — **not ready to lock.**
