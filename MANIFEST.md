# MANIFEST — Expected Reference Outputs

This file lists the expected numeric outputs of each reconstruction script
against the released adjudicated mapping layer in `data/`. Use this as a
sanity-check reference when running the package on a clean environment.

All values correspond to the released adjudicated mapping layer archived
under version DOI [10.5281/zenodo.20442140](https://doi.org/10.5281/zenodo.20442140)
(June 2026) and are derived from `data/decision_ledger.csv`,
`data/phase2_453_families.json`, and `data/itc_codebook.json`.

---

## Headline counts

| Quantity | Value | Source |
|---|---|---|
| Released dataset | 453 families | `data/phase2_453_families.json` |
| False matches (excluded from active layer) | 9 | `decision_ledger.csv` (`screening_category = false_match`) |
| Audit-only below-threshold (excluded from active layer) | 1 | `decision_ledger.csv` (`screening_category = audit_only`) |
| Analytically active layer | 443 families | 453 − 9 − 1 |
| ITC-classified subset | 373 families | `decision_ledger.csv` (`is_classified = True`) |
| Active domain-external families | 70 | `decision_ledger.csv` (`screening_category = domain_external_active`) |
| Adjudicated family–domain tags | 795 | sum of `itc_codes` over classified families |
| Diagnostic cells | 6,645 | 443 × 15 |
| Decision ledger additions | 374 | FN cells reported by `concordance_diagnostic.py` (adjudicated ITC = 1 and rule-union = 0 over the 443 × 15 active-layer diagnostic matrix) |
| Decision ledger suppressions | 256 | FP cells reported by `concordance_diagnostic.py` (rule-union = 1 and adjudicated ITC = 0 over the 443 × 15 active-layer diagnostic matrix) |
| Net tag change | +118 | 374 − 256, with 677 rule tags → 795 final tags |

Note. The `adjudication_action` column in `decision_ledger.csv` is a
family-level summary label; the cell-level addition and suppression
counts above are derived by comparing each family's `itc_codes` with
`cpc_rule_domains` or `keyword_rule_domains` (rule union) over the 443 × 15 diagnostic
matrix.

## scripts/recompute_tables_from_json.py

Regenerates the main analytical accounting and rule-to-adjudicated diagnostic outputs and cross-checks against released JSON.

Expected: ITC-domain tag counts summing to 795; rule-to-adjudicated
diagnostic summary with CPC-or-keyword recovery of 421/795 = 53.0%;
top-Jaccard pairs led by
2-1 ↔ 2-4 (J = 0.341), 4-1 ↔ 4-4 (J = 0.294),
4-2 ↔ 4-4 (J = 0.260).

## scripts/concordance_diagnostic.py

Online Resource 1 Table S6a (per-domain rule-to-adjudicated concordance).
Expected: pooled Cohen's κ = 0.519, PABAK = 0.810, Krippendorff's α
(15-label) = 0.519. These are reported as procedural diagnostics, not
inter-coder reliability estimates.

## scripts/baseline_comparison.py

Online Resource 1 Table S9 (rule-layer recovery against the adjudicated
operational reference layer).

Expected key row (rule union, CPC-or-keyword):
- TP = 421, FP = 191
- Rule-positive retention = 68.8 %
- Adjudicated-tag recovery = 53.0 %
- Adjudicated-only tags (FN) = 374 (47.0 %)

Note: 256 (suppressions) and 191 (S9 false positives) are not
interchangeable. The 256 value comes from the active-layer 6,645-cell
diagnostic in Table S6a. The 191 value is the rule-union false-positive
count within the 373-family ITC-classified subset reported in Table S9.

## scripts/anchor_sensitivity.py

Online Resource 1 Table S8 (single-anchor sensitivity for top-10 Jaccard
pairs). Expected: the raw top pair 2-1 ↔ 2-4 is strongly attenuated after
B33Y removal (J = 0.341 → 0.073), whereas selected structural and
material-process pairs retain larger residual overlaps; for example,
4-1 ↔ 4-4 retains residual J ≈ 0.161, 4-2 ↔ 4-4 retains J ≈ 0.143, and
1-3 ↔ 2-3 shows the smallest attenuation among the top-10.

## scripts/nonseed_bridging.py

Online Resource 1 Table S10 Panel A (full top-10 CPC bridging centrality
with Stage-1 retrieval seed and ITC codebook anchor flags reported
separately) and Panel B (non-codebook-anchor top-10).

## scripts/sensitivity_checks.py

Online Resource 1 Table S13 (rescue-route sensitivity).

Expected accounting:
- With rescue-route classified families: 443 active families,
  373 ITC-classified families
- Excluding 9 classified rescue-route families: 434 active families,
  364 ITC-classified families

The audit-only rescue record is excluded from both computations.

## scripts/seed_only_baseline.py

Online Resource 1 Table S14 (within-layer comparison of deterministic
first-pass and adjudicated mapping outputs on the 443-family active layer).

Expected total tag counts:

| Layer | N classified families | Total tags |
|---|---|---|
| CPC-rule only | 332 | 452 |
| CPC-or-keyword (rule union) | 393 | 677 |
| Adjudicated (released) | 373 | 795 |

Expected adjudicated WBS-layer tag-share distribution:

| WBS layer | Tags | Share |
|---|---|---|
| WBS-1 Materials | 223 | 28.1 % |
| WBS-2 Manufacturing | 253 | 31.8 % |
| WBS-3 Robotics | 37 | 4.7 % |
| WBS-4 Structures | 282 | 35.5 % |

Expected leading Jaccard adjacencies (adjudicated layer):
- 2-1 ↔ 2-4: J = 0.341 (CPC-rule only baseline J = 0.124, 2.75× larger)
- 4-1 ↔ 4-4: J = 0.294 (CPC-rule only baseline J = 0.127, 2.31× larger)

## scripts/expanded_baseline.py

Online Resource 1 Table S16 (expanded within-layer operational benchmarking
against deterministic and restrictive delineation variants, on the 443-family
active layer; adjudicated layer = reference).

Expected per-variant values:

| Layer | N classified | Total tags | Tag recovery vs 795 | Rule retention | Top-6 overlap |
|---|---:|---:|---:|---:|---:|
| CPC-rule only | 332 | 452 | 42.9 % | 75.4 % | 4/6 |
| Keyword-rule only | 271 | 377 | 26.4 % | 55.7 % | 0/6 |
| CPC-or-keyword (rule union) | 393 | 677 | 53.0 % | 62.2 % | 4/6 |
| Explicit-lunar subset (rule-union restricted) | 249 | 445 | 39.4 % | 70.3 % | 3/6 |
| Adjudicated (released) | 373 | 795 | ref | ref | 6/6 (ref) |

Note: Rule retention in S16 is computed over variant-positive tags on the
443-family active layer and is **not** numerically identical to the S9
retention values, which are restricted to the 373-family ITC-classified
subset (S9 rule-union retention = 68.8 %; S16 rule-union retention =
62.2 %).

## scripts/build_audit_sample.py and scripts/score_audit.py

Online Resource 1 Table S15 (reviewer-executable blind audit protocol).

`build_audit_sample.py` writes a blind 80-family sample
(`data/audit_sample.csv`) and an author-only answer key
(`data/audit_sample_key.csv`) using `RNG_SEED = 20260513`. Expected
stratum composition:

| Stratum | Sampling basis | Target n | Eligible pool |
|---|---|---:|---:|
| Rule-retained positives | rule hit retained after adjudication, no in-family suppression | 18 | 190 |
| Adjudication additions | at least one adjudication-only tag in the family | 22 | 241 |
| Rule suppressions | at least one rule-positive tag suppressed at adjudication | 18 | 158 |
| Domain-external active | active layer, no ITC assignment | 15 | 70 |
| False / audit-only | excluded from active layer (9 false matches + 1 audit-only) | 7 | 10 |
| **Total** | | **80** | |

`score_audit.py` computes, against the released ITC assignments: family
tag-set exact agreement, mean family-level Jaccard, and family-domain
cell-level precision, recall, F1, and Cohen's kappa, with per-stratum
breakdown.

**No author-internal audit results are reported in the manuscript.** The
protocol is provided so independent execution by a reviewer or third party is
procedurally possible; any resulting agreement score is a bounded
reliability check of the released operational layer, not external
gold-standard validation.

## scripts/generate_all_figures.py

Body Figs. 2–4 and Online Resource 1 Fig. S1. Requires Matplotlib and
seaborn; on first run, set `MPLCONFIGDIR=.mplconfig` to avoid font cache
issues. Body Fig. 2 is a two-panel figure with separate y-axis scales:
Panel A reports the 15 ITC-domain adjudicated tag counts (N = 373
ITC-classified families; 795 tags; multi-tagging permitted); Panel B
reports a single grey bar showing the 70 active domain-external
families (count, not tags). The Online Resource 1 Table S7a extension-
direction partition of the 70 domain-external families is documented
in the supplement only and is **not** shown in body Fig. 2.

---

## Numeric components of Table S12 (Online Resource 1)

The following Table S12 values are regenerated by
`scripts/table_s12_profiles.py` from `data/decision_ledger.csv` and
`data/phase2_453_families.json`.

| Pair | J | Intersection size | Dominant CPC top-3 | 2020-or-later share |
|---|---:|---:|---|---:|
| 2-1 ↔ 2-4 | 0.341 | 46 | B33Y, B29C, B22F | 89 % |
| 4-1 ↔ 4-4 | 0.294 | 48 | E04B, B64G, E04H | 83 % |
| 4-2 ↔ 4-4 | 0.260 | 19 | B64G, E04B, B32B | 68 % |
| 2-1 ↔ 2-2 | 0.202 | 23 | B22F, B33Y, B29C | 91 % |
| 1-3 ↔ 2-1 | 0.172 | 30 | B33Y, C04B, E04B | 90 % |
| 1-3 ↔ 2-3 | 0.168 | 20 | C04B, B28B, G01N | 100 % |

Dominant-CPC top-3 prefixes are counted by CPC assignment occurrences
within each intersection set, with Y* cooperative-classification tagging
schemes (Y02, Y04, Y10) excluded so the displayed top-3 reflects
substantive technical classes only. The interpretive keyword summaries
reported in the "core technology keywords" column of Table S12 are
descriptive author summaries and are **not** script-generated.

## File integrity (SHA256)

Verify package integrity against this snapshot. Hashes are computed over the canonical repository content with LF line endings, as exported by `git archive` and GitHub/Zenodo release archives (this repository ships a `.gitattributes` that enforces LF for text files). To recompute on a Unix-like system, run `sha256sum <path>` (or `shasum -a 256 <path>`) and compare to the value below; on Windows, compare against an LF checkout (e.g. the files inside the released archive) rather than a CRLF working copy.

| File | Size (bytes) | SHA256 |
|---|---:|---|
| `data/phase2_453_families.json` | 461554 | `d477ddd2e895661325399c033bf0f631e617dec19ebe833b78d0c0fb201251e4` |
| `data/decision_ledger.csv` | 79314 | `82b8f7ef93b4611513d6a339783b9c319d05297e202e0735b60e94d8ea249572` |
| `data/decision_ledger_long.csv` | 424184 | `38b96c211630210a4939ef42289351dc21a42e6949baa505ed428a06ced0d876` |
| `data/itc_codebook.json` | 5623 | `63227e68f8bbbea95b4069472ed6556a1add7bb25118c210e9c4b85ffe9fed34` |
| `data/lens_query.txt` | 4867 | `1a02f1ae1a4e0ca3446e2ef195b2abfeab2830e5907095360fbbfc8909adc8c8` |
| `scripts/recompute_tables_from_json.py` | 16497 | `b79dc0d56d5e077cb278b2996773d75a479c4a6ce7f241b7df89f96f277fb8e0` |
| `scripts/concordance_diagnostic.py` | 5184 | `02ccef5b355e45107027e2257f672c2d7e6be6e201f2492e0f1b764f6e79b94d` |
| `scripts/baseline_comparison.py` | 3947 | `9e5492ee6eec42e33a8a15148e05397e62717760be74f2b8aa3ea10b9859e310` |
| `scripts/anchor_sensitivity.py` | 8385 | `449407e69ed51c05db6103104d5ce549b2d808cf4a0aec2b052db4fe7a7136d6` |
| `scripts/nonseed_bridging.py` | 7652 | `f0a5524760cc6e52e9d527c6864364945dfac7fd13fff9137d26d2dc29c9e33a` |
| `scripts/sensitivity_checks.py` | 17728 | `c1685e50a0e8ebf7a8bf6ac58c373625274c64c6190a03fd96ea88816b00f8be` |
| `scripts/seed_only_baseline.py` | 6415 | `00c89ed61cf7402303df6130181206afe35d195fd96baefbf18baa57074757d2` |
| `scripts/expanded_baseline.py` | 9367 | `a17bb5c7b284b37132053758467274d0309363cc1e9b483bd1f5b84affd30ef9` |
| `scripts/table_s12_profiles.py` | 6832 | `6a5052ed15e4062e4edc00f2123e802e3961515b46cb58ee70d488c3557e9a1d` |
| `scripts/domain_external_partition.py` | 5538 | `aa01b06cc00896813b6f5e0dfa15c9438f3c915203e1ec90a1262daea81e7a54` |
| `scripts/build_long_form_ledger.py` | 5377 | `8fd844fc5f28f02a66ae8c0c3564aa0237ed9f01ba7cc0479c5673919600a463` |
| `scripts/build_audit_sample.py` | 6519 | `61d4674a90235816e9f740ebb84feca1d4ba093090b3c61d8c788f0ef2b3f349` |
| `scripts/score_audit.py` | 5933 | `620e01eabe13db788cb2595bdc56219db27da59070297ad340999d68ad755084` |
| `scripts/generate_all_figures.py` | 10957 | `cb8da5c5377e0431e9415d32611dfac4538c111b510767fb820fefc5e165023e` |
| `scripts/wbs_layer_concordance.py` | 9214 | `8a3eed16f5c8d9930cdd2fa99fa2345be6a2be73b5d9226c9ca9751190a06729` |
| `scripts/isru_data.py` | 17124 | `3b9463bf35214bfbb9c582ed5e29f4dcc8eebd7945ce458cd50e8b9917fe5eed` |
| `figures/fig2_itc_portfolio_bar.png` | 434211 | `c6a95bdd0a78dc70f39a37ae936c7432f3882dc3e98e8559eaa81e526a269763` |
| `figures/fig3_wbs_filing_year.png` | 247058 | `6506960acf755606947436ad9e79fa877e78588d19fdb6b72c494dd186ec9901` |
| `figures/fig4_itc_jaccard_matrix.png` | 646268 | `39c318b7fb14e3018596dd48a3349d327434016824b0f4552d8a48ce6c180a6a` |
| `figures/figS1_cpc_coclass_heatmap.png` | 356161 | `1937f4f681dced7dc2c5499a3617f678677b8c692f5994c6076418a9f29babd8` |
| `requirements.txt` | 51 | `4dff5cc6ebc133e8a62cc9d49a72facc24cbb3dc9013ab29db66ec157a801812` |
| `requirements-lock.txt` | 408 | `62455136246b10defb0f1e220816c51c9197db3be975bb84a8f9df61941e96f0` |
| `environment.yml` | 152 | `d826bd242a0eec3c60ab02fdf92686ac0e570e6986f2a250bd4fe2f96948e3af` |
| `run_all.sh` | 1543 | `003a6a4544aa2d67b280a6fc2173c61fb7e0e833733f1772998150d468a3daa7` |
| `README.md` | 13764 | `d4c11c133e83e08956eee0ab5ff185e92f5df1dc36fb17a90fa41e34a961a841` |
| `CITATION.cff` | 2340 | `b22ea9d599980baaec94ca552d4612caaba68da600f29bbfd14cbcb091c2c904` |
| `LICENSE` | 272 | `ed8e5cc4b7c082357eb6ceccb99aa8aad5a84092b6bd0fb93f62ca19aaf960dd` |
