# MANIFEST - Expected Reference Outputs

This file lists the expected numeric outputs of each reconstruction script in
`scripts/`, computed from the released dataset in `data/`. Use it as a
sanity-check reference when running the package on a clean environment.

All values correspond to the submitted manuscript snapshot (reproduction
package v0.1.3). Zenodo version DOI 10.5281/zenodo.21124281; concept DOI
10.5281/zenodo.20993229 (see `CITATION.cff`). Values are derived from `data/assignment_table.csv`,
`data/phase2_453_families.json`, and `data/itc_codebook.json`.

The quantities compare the deterministic rule-hit layer (CPC / title-abstract)
with the claim-reviewed WBS-ITC assignment as within-corpus
agreement / divergence, not as classifier-performance metrics.

---

## Headline counts

| Quantity | Value | Source |
|---|---|---|
| Released dataset | 453 families | `data/phase2_453_families.json` |
| False matches (excluded from active layer) | 9 | `assignment_table.csv` (`status_category = false_match`) |
| Below-threshold record (excluded from active layer) | 1 | `assignment_table.csv` (`status_category = screening_log_only`) |
| Analytically active layer | 443 families | 453 - 9 - 1 |
| ITC-classified subset | 373 families | `assignment_table.csv` (`is_classified = True`) |
| Active domain-external families | 70 | `assignment_table.csv` (`status_category = domain_external_active`) |
| Claim-reviewed WBS-ITC tags | 795 | sum of `itc_codes` over classified families |
| Diagnostic cells | 6,645 | 443 x 15 |
| Assignment-only additions | 374 | cells with assignment = 1 and rule-union = 0 over the 443 x 15 active-layer matrix |
| Rule-only (not retained) | 256 | cells with rule-union = 1 and assignment = 0 over the 443 x 15 active-layer matrix |
| Net tag change | +118 | 374 - 256 (677 rule-positive tags -> 795 assignment tags) |

Note. The `assignment_hit` column in the released data is the claim-reviewed WBS-ITC assignment indicator; the cell-level
addition / not-retained counts above are derived by comparing each family's
`itc_codes` with the rule union (`cpc_rule_domains` or `keyword_rule_domains`)
over the 443 x 15 matrix.

## scripts/recompute_tables_from_json.py

Regenerates the analytical accounting and rule-hit/assignment agreement
outputs and cross-checks them against the released JSON. Expected: ITC-domain
tag counts summing to 795; CPC-or-keyword agreement of 421/795 = 53.0%; top
Jaccard pairs led by 2-1 <-> 2-4 (J = 0.341), 4-1 <-> 4-4 (J = 0.294),
4-2 <-> 4-4 (J = 0.260).

## scripts/concordance_diagnostic.py

Online Resource 1 Table S6a (per-domain rule-hit/assignment agreement over the
443 x 15 = 6,645 cells). Expected pooled: 677 rule-positive tags -> 795
assignment tags; 374 added, 256 not retained, net +118. Removal-heavy domains
include Process Monitoring (2-4); addition-heavy domains include Habitat
Structures (4-1).

## scripts/baseline_comparison.py

Online Resource 1 Table S9 (rule-hit/assignment agreement within the 373-family
ITC-classified subset). Expected key row (CPC-or-keyword rule union):
shared 421, rule-only 191, rule-positive 612, assignment-tag agreement 53.0%,
rule-positive agreement 68.8%, assignment-only 374 (47.0%). CPC-only agreement
42.9%; keyword-only agreement 26.4%. These are internal layer-comparison
counts on the 373-family subset and are not interchangeable with the 256
not-retained count in the 6,645-cell Table S6a accounting.

## scripts/anchor_sensitivity.py

Online Resource 1 Table S8 (single-anchor sensitivity for the top-10 Jaccard
pairs). Expected: the top pair 2-1 <-> 2-4 is strongly attenuated after B33Y
removal (J = 0.341 -> 0.073), whereas 4-1 <-> 4-4 retains residual J ~ 0.161,
4-2 <-> 4-4 retains J ~ 0.143, and 1-3 <-> 2-3 shows the smallest attenuation.

## scripts/nonseed_bridging.py

Online Resource 1 Table S10 Panel A (top-10 CPC bridging centrality with
Stage-1 retrieval seed and ITC anchor flags) and Panel B (non-anchor top-10).

## scripts/sensitivity_checks.py

Online Resource 1 Table S13 (rescue-route sensitivity). With rescue-route
families: 443 active, 373 ITC-classified. Excluding 9 classified rescue-route
families: 434 active, 364 ITC-classified. Writes `sensitivity_notes.md`.

## scripts/seed_only_baseline.py

Online Resource 1 Table S14 (within-corpus comparison of deterministic
first-pass and claim-reviewed assignment outputs on the 443-family active
layer). Expected total tags: CPC-rule only 452; CPC-or-keyword 677; released
assignment 795. Assignment WBS-layer shares: WBS-1 28.1%, WBS-2 31.8%,
WBS-3 4.7%, WBS-4 35.5%. Leading Jaccard adjacencies: 2-1 <-> 2-4 J = 0.341,
4-1 <-> 4-4 J = 0.294.

## scripts/expanded_baseline.py

Online Resource 1 Table S16 (expanded within-corpus comparison against
deterministic and restrictive variants; released assignment = reference).
Expected assignment-tag overlap: CPC-rule only 42.9%, keyword-only 26.4%,
rule union 53.0%, explicit-lunar subset 39.4%. Top-6 overlap: 4/6, 0/6, 4/6,
3/6, 6/6 (ref).

## scripts/table_s12_profiles.py

Online Resource 1 Table S12 numeric components: adjacency pair, Jaccard,
intersection size, dominant CPC top-3 prefixes within the intersection, and
2020-or-later share. The interpretive keyword summaries in Table S12 are
documented in the manuscript but are not script-generated.

| Pair | J | Intersection | Dominant CPC top-3 | 2020+ share |
|---|---:|---:|---|---:|
| 2-1 <-> 2-4 | 0.341 | 46 | B33Y, B29C, B22F | 89% |
| 4-1 <-> 4-4 | 0.294 | 48 | E04B, B64G, E04H | 83% |
| 4-2 <-> 4-4 | 0.260 | 19 | B64G, E04B, B32B | 68% |
| 2-1 <-> 2-2 | 0.202 | 23 | B22F, B33Y, B29C | 91% |
| 1-3 <-> 2-1 | 0.172 | 30 | B33Y, C04B, E04B | 90% |
| 1-3 <-> 2-3 | 0.168 | 20 | C04B, B28B, G01N | 100% |

## scripts/domain_external_partition.py

Coarse CPC-prefix diagnostic for the 70 active domain-external families
(Online Resource 1 Table S7a-related; not an exact reproduction).

## scripts/build_long_form_table.py

Generates `data/assignment_table_long.csv`, the family-domain family-domain assignment table
(453 x 15 full-release cells; active-layer diagnostics use the 443 x 15 subset).

## scripts/generate_all_figures.py

Body Figs. 2-4 and Online Resource 1 Fig. S1. Requires Matplotlib and seaborn;
on first run set `MPLCONFIGDIR=.mplconfig` to avoid font-cache issues. Body
Fig. 2 Panel A reports the 15 ITC-domain assignment tag counts (N = 373
classified families; 795 tags); Panel B reports the 70 active domain-external
families (count, not tags).

## scripts/reproduce_boundary_adjacency.py

Body Fig. 5 (inherited CPC-only vs constructed assignment co-tagging adjacency,
and rank trajectories of the leading pairs) and Table 9 (leading reordering
pairs); written to `tables/table9_boundary_adjacency.csv`. Recomputed from
`data/assignment_table_long.csv`. Headline values: Spearman rho = 0.41 and
Kendall tau = 0.31 over 105 domain pairs; 45 / 70 / 83 active co-tagging pairs
under the CPC-only / rule-union / assignment layers.

## File integrity (SHA256)

Check package integrity against this snapshot. To recompute on a Unix-like
system, run `sha256sum <path>` (or `shasum -a 256 <path>`) and compare.

_Note on figures: the SHA256 table covers the distributed release snapshot. Regenerated PNG files may have different byte-level hashes across environments because Matplotlib can write environment-dependent image metadata; after rerunning `run_all.sh`, compare the numerical outputs and figure content rather than PNG hashes._

| File | Size (bytes) | SHA256 |
|---|---:|---|
| `.gitignore` | 33 | `6c115c4b0dac50cb6367904f1fe6f77123013bab56d92ddb518b88d09b23456b` |
| `CITATION.cff` | 2120 | `f5af5e4f0a5de08b11d78eeff1fa70a836f4ba54d533c9b1fbda80d14b61c86e` |
| `LICENSE` | 272 | `ed8e5cc4b7c082357eb6ceccb99aa8aad5a84092b6bd0fb93f62ca19aaf960dd` |
| `README.md` | 9323 | `1ee84e0f1874b35efbcf7b2b8285d1cb4f449a7265a87fc5637d06d6428497d2` |
| `data/assignment_table.csv` | 79317 | `15c5a91cde4312c0d82171af7726c57cb9c74e094168a9f1f6784149c7a24d8a` |
| `data/assignment_table_long.csv` | 424300 | `8dad5066823f96d5dc522e9ef1b8c8f8bfebd015cee01879296c35519d0dcdc8` |
| `data/itc_codebook.json` | 5682 | `2a7760a3c81ec02c4b29f0c830af27b8d69805ed93211b8dcf3723a9d84e8d53` |
| `data/lens_query.txt` | 4899 | `7b60132cec6883f420ef3afc98a77abfe8c6a1c75ec1b7c5dfc66e1cbef1e3cc` |
| `data/phase2_453_families.json` | 461554 | `d477ddd2e895661325399c033bf0f631e617dec19ebe833b78d0c0fb201251e4` |
| `environment.yml` | 169 | `f57195c7f6cf3377c552caa69673563c29e6f7d1e1ddade901aa7b77a40c7267` |
| `figures/fig2_itc_portfolio_bar.png` | 427777 | `ee6041fe3f7325c1593cc2794643829bd5489d8c0ccf5b3d5fbfbb7c5b25defe` |
| `figures/fig3_wbs_filing_year.png` | 247058 | `d2f4a079588ad0aeb38a520414c7139bf67185984b655670fdee61e51938f0e7` |
| `figures/fig4_itc_jaccard_matrix.png` | 646268 | `49a314624527cf85bd3f370d84d2ea979503e13514c86a1a61b84208c33c1930` |
| `figures/fig5_boundary_vs_convergence.png` | 454713 | `185b0b6d5d9399fccfdbb6cf7693b74f7c036c065d8023f91a8f063819b95acd` |
| `figures/figS1_cpc_coclass_heatmap.png` | 356161 | `7e34fbc9073f21c204451c26cdd2bb99f27042c661e5b6a583d1ea9f9bf0a13c` |
| `requirements-lock.txt` | 421 | `f5d337186eee1c59e9b1f6801404b0b6db1fa4d99607e2cb60714e023a511d27` |
| `requirements.txt` | 81 | `4f1d6f6a5fab0be9583e97a8328e2af6f140e754cd01ada87563a3a3945d4e1b` |
| `run_all.sh` | 1813 | `2c92c20be6991eea7f112464874d6956bace5438f92b753eebabbfb05b8c1210` |
| `scripts/anchor_sensitivity.py` | 8386 | `ece555db0091505b63ddc7030335839b4b719baff3a82905ac21f0b07bda54a7` |
| `scripts/baseline_comparison.py` | 4180 | `3446054a5afd0b8322e1f95c97240bad5b7c333d83cda3b5fda6eb1bf2af3bcc` |
| `scripts/build_long_form_table.py` | 5405 | `e99a5692d2061e0ff683bcc6d882272e89cb97e05447f6bdaf373df4baf82400` |
| `scripts/concordance_diagnostic.py` | 4754 | `c04922e3be94686e414274e1d52d4929374afc95112e5a4439eb501f3bd7fb92` |
| `scripts/domain_external_partition.py` | 5566 | `5f9ffb00255a65543820a41ee82ee8f71b8206f1befc9d101d07527340b2656e` |
| `scripts/expanded_baseline.py` | 9449 | `c1544bccd86abf917ca5e91e8cccd2457b9a4b0277f26d92353dd865a41d7b98` |
| `scripts/generate_all_figures.py` | 10963 | `83672508f11851ee127762df17d91d49b2b747130d5984446a967cd48289ad39` |
| `scripts/isru_data.py` | 17149 | `26f114eb91d8392600dd439ce02e8d07a4391ca438de9f47840997dae6dae659` |
| `scripts/nonseed_bridging.py` | 7656 | `27073ff795c02ce5c504656148991516767fa045b0b1b808ebc8c464f782d927` |
| `scripts/recompute_tables_from_json.py` | 15996 | `636e02e75d812d7b44dccf83fc8b151f7632b4d1ff688a5056f85241d20f2021` |
| `scripts/reproduce_boundary_adjacency.py` | 12858 | `36277f7f8f6c2641257acb6ef4c305afe151a8b3ceee0fa51baaa82887295795` |
| `scripts/seed_only_baseline.py` | 6399 | `e35131590d46ebe7b8c8b2219992607211cde24814e04f8035c9716c7001e105` |
| `scripts/sensitivity_checks.py` | 17796 | `a596d2272dce8b442870da0c0c1869a1e793c589c219cba0f75b41afff1ab92b` |
| `scripts/table_s12_profiles.py` | 6808 | `5b06960e1b504d3c6946c881edad8ac22a7e119596779ff0b901eb5c4560622f` |
| `sensitivity_notes.md` | 5872 | `498c4ce4e585774bace7797c37573b799163bced10cc6153fd285f378048c390` |
| `tables/table9_boundary_adjacency.csv` | 395 | `d8d25b0bc6ddf350f9d4c3aa7341828b6bbeba34185dd19ec4d7f8fcf8a57eb7` |
