#!/usr/bin/env bash
# One-command reconstruction of the WBS-ITC reproduction package.
# Regenerates the data-derived tables and figures reported in the manuscript
# and Online Resource 1:
#   - Main analytical accounting and rule-hit/assignment agreement outputs
#   - Top Jaccard co-tagging pairs
#   - CPC bridging centrality and CPC co-classification matrix
#   - Online Resource 1 Tables S6a, S8, S9, S10, S12 (numeric), S13, S14, S16
#   - Body Figs. 2, 3, 4 and Online Resource 1 Fig. S1
#   - Body Fig. 5 and Table 9 (inherited-vs-constructed boundary adjacency)
#   - data/assignment_table_long.csv
#
# Usage:
#   bash run_all.sh
#
# Expected reference outputs are listed in MANIFEST.md.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

# Force UTF-8 for Python stdout/file I/O so the scripts' non-ASCII console
# output (em dashes, arrows, Greek letters) does not crash on non-UTF-8
# default locales (e.g. Windows cp949/cp1252 consoles).
export PYTHONUTF8=1

echo "[1/2] Recomputing tables and agreement diagnostics ..."
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
python scripts/build_long_form_table.py

echo "[2/2] Regenerating figures ..."
mkdir -p .mplconfig
MPLCONFIGDIR=.mplconfig python scripts/generate_all_figures.py
MPLCONFIGDIR=.mplconfig python scripts/reproduce_boundary_adjacency.py --table data/assignment_table_long.csv --outdir .

echo "Done. Compare console outputs against MANIFEST.md headline values."
