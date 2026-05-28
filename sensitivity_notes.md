# Package sensitivity notes

This note documents the sensitivity checks supplied with the released analytical-layer dataset.

## Dataset scope

| Scope | N |
|---|---|
| Released families (JSON) | 453 |
| Excluded (9 false-match + 1 audit-only) | 10 |
| Analytically active | 443 |
| ITC-classified | 373 |
| Domain-external (active) | 70 |
| Jurisdiction (released) | CN=342, Non-CN=111 |
| Jurisdiction (active) | CN=336, Non-CN=107 |

Note: This script operates on the full 453-family released JSON. Because all 10 excluded families are unclassified (no ITC tags), ITC-based metrics are mathematically invariant to the exclusion.
CPC-level metrics are empirically stable (see classified-only comparison below).

## 1) Leave-out sensitivity excluding rescue families

- Released dataset size: **453 families**
- Rescue families retained after deduplication: **10**
- Leave-out dataset size: **443 families**

Online Resource 1 Table S13 accounting:

| Layer | With rescue-route classified families | Excluding 9 classified rescue-route families |
|---|---|---|
| Active analytical layer | 443 | 434 |
| ITC-classified subset    | 373 | 364 |

Note: the audit-only rescue record is excluded from both computations.

- Maximum absolute WBS-layer tag-share change: **0.42 percentage points**
- Top six Jaccard pairs unchanged: **Yes**

| WBS layer | All families tags | All families share | No-rescue tags | No-rescue share | Change |
|---|---|---|---|---|---|
| Materials | 223 | 28.05% | 215 | 27.63% | -0.42 pp |
| Manufacturing | 253 | 31.82% | 249 | 32.01% | +0.18 pp |
| Robotics | 37 | 4.65% | 36 | 4.63% | -0.03 pp |
| Structures & Systems | 282 | 35.47% | 278 | 35.73% | +0.26 pp |

Top 10 Jaccard pairs — all families

| Rank | Pair | Jaccard |
|---|---|---|
| 1 | 2-1↔2-4 | 0.341 |
| 2 | 4-1↔4-4 | 0.294 |
| 3 | 4-2↔4-4 | 0.260 |
| 4 | 2-1↔2-2 | 0.202 |
| 5 | 1-3↔2-1 | 0.172 |
| 6 | 1-3↔2-3 | 0.168 |
| 7 | 1-1↔2-4 | 0.167 |
| 8 | 1-3↔4-1 | 0.164 |
| 9 | 4-2↔4-5 | 0.163 |
| 10 | 4-1↔4-2 | 0.161 |

Top 10 Jaccard pairs — excluding rescue families

| Rank | Pair | Jaccard |
|---|---|---|
| 1 | 2-1↔2-4 | 0.351 |
| 2 | 4-1↔4-4 | 0.300 |
| 3 | 4-2↔4-4 | 0.260 |
| 4 | 2-1↔2-2 | 0.202 |
| 5 | 1-3↔2-1 | 0.172 |
| 6 | 1-3↔2-3 | 0.168 |
| 7 | 4-2↔4-5 | 0.167 |
| 8 | 1-3↔4-1 | 0.166 |
| 9 | 4-1↔4-2 | 0.164 |
| 10 | 1-2↔2-1 | 0.150 |

Interpretation: the leading Jaccard structure is stable, but lower-ranked pairs shift modestly once the 10 rescue families are removed.

## 2) CPC bridging sensitivity — all families vs classified-only

- Families excluded in the classified-only run: **80** (of which 70 are analytically active domain-external and 10 are non-active excluded records)
- Leading bridging trio unchanged (B64G, E04H, G01N): **Yes**

Top 10 CPC bridging codes — all 453 released families

| Rank | CPC | Freq | Degree | Betweenness |
|---|---|---|---|---|
| 1 | B64G | 94 | 0.422 | 0.243 |
| 2 | E04H | 40 | 0.301 | 0.109 |
| 3 | G01N | 53 | 0.289 | 0.152 |
| 4 | E04B | 52 | 0.265 | 0.087 |
| 5 | B33Y | 75 | 0.253 | 0.084 |
| 6 | C04B | 75 | 0.217 | 0.092 |
| 7 | E02D | 47 | 0.199 | 0.067 |
| 8 | H02J | 11 | 0.187 | 0.042 |
| 9 | Y02E | 12 | 0.181 | 0.030 |
| 10 | Y02A | 14 | 0.175 | 0.029 |

Top 10 CPC bridging codes — classified-only families

| Rank | CPC | Freq | Degree | Betweenness |
|---|---|---|---|---|
| 1 | B64G | 87 | 0.429 | 0.256 |
| 2 | E04H | 40 | 0.311 | 0.124 |
| 3 | G01N | 51 | 0.286 | 0.163 |
| 4 | E04B | 52 | 0.273 | 0.092 |
| 5 | B33Y | 75 | 0.261 | 0.087 |
| 6 | C04B | 62 | 0.217 | 0.096 |
| 7 | H02J | 11 | 0.193 | 0.045 |
| 8 | E21B | 11 | 0.174 | 0.031 |
| 9 | Y02E | 10 | 0.174 | 0.029 |
| 10 | B22F | 34 | 0.168 | 0.050 |

Interpretation: the leading bridging codes remain the same, while lower-ranked CPC positions shift once domain-external families are removed.

## 3) Jurisdiction-stratified sensitivity: CN vs non-CN subsets

- CN families (released 453): **342** (75.5%)
- Non-CN families (released 453): **111** (24.5%)
- Manuscript analytically active (443): CN = 336, Non-CN = 107 (10 excluded: 6 CN + 4 non-CN)

### Panel A: WBS-layer tag shares (classified families only)

| WBS layer | All | CN | Non-CN |
|---|---|---|---|
| Materials | 28.1% | 29.0% | 25.5% |
| Manufacturing | 31.8% | 35.8% | 21.3% |
| Robotics | 4.7% | 4.3% | 5.6% |
| Structures & Systems | 35.5% | 30.9% | 47.7% |

### Panel B: Top Jaccard pair stability

- All families: **2-1↔2-4** (J = 0.341)
- CN subset: **2-1↔2-4** (J = 0.333)
- Non-CN subset: **2-1↔2-4** (J = 0.381)

### Panel C: CPC bridging code stability

- All families top bridging code: **B64G**
- CN subset top bridging code: **G01N**
- Non-CN subset top bridging code: **B64G**

Interpretation: the manufacturing emphasis is partly driven by CN filing patterns, but the core convergence structure (top Jaccard pair and principal bridging code) persists across both subsets.

## 4) Shared-anchor inflation check: 4-1 ↔ 4-4 (E04H15)

E04H15 serves as a CPC anchor for both domain 4-1 (Habitat Structures) and domain 4-4 (Deployable Structures).
Families whose sole tagging route to either domain runs through E04H15 are removed to quantify the inflation effect.

- E04H15-only families removed: **9**
- Original 4-1 ↔ 4-4 Jaccard: **0.294** (|4-1| = 146, |4-4| = 65, intersection = 48)
- Adjusted 4-1 ↔ 4-4 Jaccard: **0.253** (|4-1| = 137, |4-4| = 56, intersection = 39)
- Change: **0.294 -> 0.253** (delta = -0.041)

Interpretation: removing E04H15-only families attenuates the 4-1 / 4-4 Jaccard somewhat, indicating that part of the habitat-deployable adjacency is driven by the shared E04H15 anchor rather than independent claim-level evidence.

---
Generated by `scripts/sensitivity_checks.py` from the released `data/phase2_453_families.json` and `data/decision_ledger.csv`.
