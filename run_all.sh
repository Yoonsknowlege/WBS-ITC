#!/usr/bin/env bash
# One-command reconstruction of the WBS-ITC reproduction package.
# Regenerates:
#   - Main analytical accounting and rule-to-adjudicated diagnostic outputs
#   - Jaccard top-pair table
#   - CPC bridging centrality and CPC co-classification matrix
#   - Online Resource 1 Tables S6a, S8, S9, S10, S13, S14, S16, S12 (numeric)
#   - Body Figs. 2, 3, 4 and Online Resource 1 Fig. S1
#   - data/decision_ledger_long.csv
#
# Usage:
#   bash run_all.sh
#
# Expected reference outputs are listed in MANIFEST.md.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

echo "[1/3] Recomputing body tables and concordance diagnostics ..."
python scripts/recompute_tables_from_json.py
python scripts/concordance_diagnostic.py
python scripts/baseline_comparison.py
python scripts/anchor_sensitivity.py
python scripts/nonseed_bridging.py
python scripts/sensitivity_checks.py
python scripts/seed_only_baseline.py
python scripts/expanded_baseline.py
python scripts/table_s12_profiles.py
python scripts/domain_external_partition.py
python scripts/build_long_form_ledger.py

echo "[2/3] Regenerating figures ..."
mkdir -p .mplconfig
MPLCONFIGDIR=.mplconfig python scripts/generate_all_figures.py

echo "[3/3] Done. Compare console outputs against MANIFEST.md headline values."
echo "Audit-protocol scripts (scripts/build_audit_sample.py, scripts/score_audit.py)"
echo "are intentionally not invoked here; see README.md Section 4 for the optional"
echo "Table S15 reviewer-executable blind audit protocol."
