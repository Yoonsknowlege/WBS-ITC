# Denominator construction for patent mapping in classification-poor emerging domains: A framework-first WBS–ITC approach with a lunar ISRU construction illustration

## Overview

This repository provides the reproduction package for the manuscript:

> Lee, T.S. & Lee, Y. (2026). "Denominator construction for patent mapping in classification-poor emerging domains: A framework-first WBS–ITC approach with a lunar ISRU construction illustration." *Scientometrics* (under review).

The package enables exact reconstruction of the core downstream numeric outputs (Tables 3–5, Figs. 2–5, and Online Resource 1 Tables S6a and S8–S10) from the released adjudicated dataset. It does not provide deterministic replay of the upstream retrieval, screening, or consensus-adjudication workflow.

---

## Repository structure

```
./
├── data/
│   ├── phase2_453_families.json      # Released dataset (453 families)
│   ├── decision_ledger.csv           # Family-level adjudication record
│   ├── itc_codebook.json            # Operative CPC/keyword rule set (15 domains)
│   └── lens_query.txt               # Retrieval query provenance
│
├── scripts/
│   ├── isru_data.py                  # Analytical constants and codebook
│   ├── generate_all_figures.py       # Regenerate Figs. 2–5
│   ├── recompute_tables_from_json.py # Recompute Tables 3–5 from JSON
│   ├── sensitivity_checks.py         # Sensitivity analyses
│   ├── anchor_sensitivity.py         # Systematic anchor-removal (Table S8)
│   ├── baseline_comparison.py        # Rule-layer baseline (Table S9)
│   ├── concordance_diagnostic.py     # Rule-to-adjudicated concordance (Table S6a)
│   ├── wbs_layer_concordance.py      # WBS-layer κ and Krippendorff's α
│   └── nonseed_bridging.py           # Non-seed CPC bridging diagnostic (Table S10)
│
├── figures/                          # Generated figure outputs
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Recompute core tables from the released JSON

```bash
python scripts/recompute_tables_from_json.py
```

Independently derives from `phase2_453_families.json`:
- Table 3: ITC domain portfolio tag counts (15 domains)
- Table 4: CPC bridging centrality top-10
- Table 5: Jaccard similarity top pairs (15 × 15 matrix)
- Figure 3 data: filing-year × WBS-layer tag aggregation
- Figure 4 data: CPC co-classification matrix for top 25 codes

### 3. Regenerate figures

```bash
python scripts/generate_all_figures.py
```

Figure scripts read audited constants hard-coded in `scripts/isru_data.py` (portfolio counts, filing-year distributions, Jaccard matrix, CPC co-occurrence matrix). These constants are cross-checked against the released JSON by `recompute_tables_from_json.py`, which independently recomputes the same values and compares them to the hard-coded reference. If any value drifts, the cross-check will report a mismatch.

### 4. Run sensitivity checks

```bash
python scripts/sensitivity_checks.py
```

Reproduces: leave-out analysis, classified-only CPC bridging comparison, CN vs non-CN jurisdictional sensitivity (Supplementary Table S2), shared-anchor adjustments.

### 5. Run systematic anchor sensitivity (Supplementary Table S8)

```bash
python scripts/anchor_sensitivity.py
```

Reproduces the systematic CPC anchor-removal test for all top-10 Jaccard pairs. Panel A uses pre-specified theory-guided CPC anchors (domain-relevant, not simply the most frequent prefix); Panel B uses the auto-detected most-frequent prefix for complementary comparison. For each pair, removes intersection families carrying the tested anchor and reports the adjusted Jaccard (J_adj) and sensitivity loss (Δ).

### 6. Run rule-layer baseline comparison (Supplementary Table S9)

```bash
python scripts/baseline_comparison.py
```

Computes domain-tag-level precision and recall for:
- CPC-rule only: precision 81.8%, recall 42.9%
- Keyword-rule only: precision 62.1%, recall 26.4%
- CPC ∪ Keyword (rule union): precision 68.8%, recall 53.0%
- Adjudication-only tags: 374 of 795 (47.0%)

### 7. Run concordance diagnostic (Supplementary Table S6a)

```bash
python scripts/concordance_diagnostic.py
```

Reproduces the per-domain and pooled rule-to-adjudicated concordance diagnostic (Cohen's κ, PABAK) on the 443 × 15 family–domain matrix. Pooled κ = 0.519, PABAK = 0.810.

### 8. Run WBS-layer concordance and Krippendorff's α

```bash
python scripts/wbs_layer_concordance.py
```

Reproduces: WBS-layer-level Cohen's κ (Materials = 0.257, Manufacturing = 0.590, Robotics = 0.269, Structures & Systems = 0.617) and Krippendorff's α (nominal) = 0.519 over the 443 × 15 binary family–domain matrix.

### 9. Run non-seed CPC bridging diagnostic (Supplementary Table S10)

```bash
python scripts/nonseed_bridging.py
```

Computes CPC bridging centrality on the 443-active families, flags each code as seed or non-seed (against `itc_codebook.json`), and reports: (a) the full top-10 with seed-anchor flag showing 6/10 codes are retrieval seed anchors, and (b) the non-seed-only top-10 providing the most conservative cross-domain connectivity evidence. Non-seed codes E02D, H02J, Y02E, and E21B occupy overall ranks #7–#10.

---

## Data description

The dataset (`phase2_453_families.json`) contains 453 released patent families. From this total, nine false-match records and one audit-only near-threshold record are excluded, yielding **443 analytically active families** (373 ITC-classified + 70 codebook-external). The 70 codebook-external families (labeled `domain_external` in the CSV and scripts for backward compatibility) are retained as transfer-relevant boundary evidence but receive no assignment within the current 15-domain ITC codebook.

### Dataset scope summary

| Scope | N |
|---|---|
| Released families (JSON) | 453 |
| Excluded (9 false-match + 1 audit-only) | 10 |
| Analytically active | 443 |
| ITC-classified | 373 |
| Codebook-external / domain-external (active) | 70 |

### Family-level traceability

Each family record includes a `lens_id` field (e.g., `"172-275-795-866-748"`). The corresponding Lens.org patent page can be accessed at `https://www.lens.org/lens/patent/{lens_id}`, enabling claim-level auditability for every entry in the dataset.

### Decision ledger

`data/decision_ledger.csv` provides a family-level adjudication record for all 453 released families. Each row includes: final ITC assignments, CPC-rule-layer hit domains, keyword-rule-layer hit domains, screening status, analytical-set membership flag, and `adjudication_action` (keep / add / suppress / add+suppress / add_only / suppress_all). The action column summarizes, for each family, whether adjudication confirmed the rule-union output, added domains, suppressed domains, or both. This ledger allows third parties to trace which tagging route (CPC anchor, keyword, or adjudication) contributed to each family's domain assignment; per-family rationale notes are not included.

---

## WBS–ITC framework

The Work-Breakdown-Structure-based Integrated Technology Classification (WBS–ITC) codebook was specified prior to empirical analysis, drawing on agency frameworks (ISECG, NASA, KICT); no post-hoc adjustment of WBS layers or ITC domains was performed after observing empirical distributions.

- **4 WBS layers:** Materials, Manufacturing, Robotics, Structures & Systems
- **15 ITC domains** defined using CPC anchor prefixes and keyword stems
- Multi-domain assignment allowed
- Full operative rule set: `data/itc_codebook.json` (the complete CPC-anchor and keyword-stem list used in the deterministic first stage)
- Programmatic implementation: `scripts/isru_data.py` (ITC_RULES dictionary)

**Relationship to manuscript Table 2:** Table 2 in the manuscript is a summary presentation of the operative rule set. The authoritative machine-readable version is `data/itc_codebook.json`, which contains the complete CPC anchor prefixes and keyword stems for all 15 domains. Minor editorial differences (e.g., truncated keyword lists in Table 2 for space) are resolved by treating this file as the definitive reference.

---

## Reproducibility scope

**Directly recomputable from the released data:**
- Core numeric tables and figures in the manuscript (Tables 3–5, Figs. 2–5)
- Concordance diagnostic (Supplementary Table S6a)
- WBS-layer κ and Krippendorff's α (§3.5 in-text values)
- Systematic anchor sensitivity (Supplementary Table S8)
- Rule-layer baseline comparison (Supplementary Table S9)
- Non-seed CPC bridging diagnostic (Supplementary Table S10)

**Partially recomputable:** Table 6 (domain overview) contains interpretive summaries (dominant CPC, core keywords) that combine computed statistics with editorial characterization; the numeric components are derivable from the JSON but the interpretive labels are not script-generated.

**Not reproduced by this repository:**
- Initial retrieval universe from Lens.org (query provenance in `lens_query.txt`)
- Intermediate screening decisions (consensus-based author adjudication)

---

## License

CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/

---

## Citation

See `CITATION.cff` or the associated manuscript (under review).

---

## Contact

Corresponding author: Yoonsun Lee (yoonsunlee@hanyang.ac.kr)
