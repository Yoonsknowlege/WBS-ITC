#!/usr/bin/env python3
"""
domain_external_partition.py — coarse CPC-prefix diagnostic for the 70
active domain-external families documented in Online Resource 1 Table S7a.

Paper: "Auditable denominator design for patent-based field
       delineation in classification-dispersed emerging
       technologies"

This script:

  1. Filters data/decision_ledger.csv to the 70 active domain-external
     families (screening_category = domain_external_active).
  2. Applies a coarse CPC-prefix categorisation to those 70 families
     using the released family-level CPC field in
     data/phase2_453_families.json.

The output is a coarse, fully deterministic CPC-prefix-based diagnostic.
It is provided to make the 70-family pool inspectable from the released
dataset without manual annotation. It does NOT exactly reproduce
Online Resource 1 Table S7a Panel A, because Table S7a's partition
incorporates an author-level audit of each family's first-claim
mechanism, which is curated rather than rule-derivable. The categories
that depend strongly on first-claim reading (notably "Boundary-adjacent")
will diverge between the coarse rule output and Table S7a Panel A.

Mapping rule (priority order; each family counted once):

    Foundation engineering    -> CPC prefix in {E02D, E21D}
    Crushing / size reduction -> CPC prefix in {B02C}
    Drilling / boring         -> CPC prefix in {E21B, E21C}
    Mixing / emplacement      -> CPC prefix in {B28C, B65G}
    Screening / beneficiation -> CPC prefix in {B07B, B03B, B03C}
    Boundary-adjacent (rule)  -> any of {H02J, H02S, Y02E, Y02P, Y02A,
                                         B32B, F04B, F03G}
    Other / curator-flagged   -> none of the above prefix sets matched

The total equals 70 (the active domain-external count) by construction.
The "Other / curator-flagged" bucket corresponds to families that
Table S7a Panel A redistributes between "Boundary-adjacent" (22) and
"Unclassified" (5) through manuscript-level reading; that redistribution
is not script-generated.

Usage:
    python scripts/domain_external_partition.py
"""

import csv
import json
import os
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
LEDGER_PATH = os.path.join(DATA_DIR, "decision_ledger.csv")
JSON_PATH = os.path.join(DATA_DIR, "phase2_453_families.json")

PARTITION_RULES = OrderedDict([
    ("Foundation engineering",    {"E02D", "E21D"}),
    ("Crushing / size reduction", {"B02C"}),
    ("Drilling / boring",         {"E21B", "E21C"}),
    ("Mixing / emplacement",      {"B28C", "B65G"}),
    ("Screening / beneficiation", {"B07B", "B03B", "B03C"}),
    ("Boundary-adjacent",         {"H02J", "H02S", "Y02E", "Y02P",
                                   "Y02A", "B32B", "F04B", "F03G"}),
])


def cpc_prefixes(cpc_str):
    if not cpc_str:
        return set()
    delim = ";;" if ";;" in cpc_str else ";"
    return {c.strip()[:4] for c in cpc_str.split(delim) if len(c.strip()) >= 4}


def categorise(prefixes):
    for label, prefix_set in PARTITION_RULES.items():
        if prefixes & prefix_set:
            return label
    return "Other / curator-flagged"


def main():
    # 1) Identify the 70 active domain-external lens_ids from the ledger
    domain_external_ids = set()
    with open(LEDGER_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("screening_category") == "domain_external_active":
                domain_external_ids.add(row["lens_id"])

    print("This script does NOT reproduce Online Resource 1 Table S7a:")
    print("it only reproduces the 70-family total and a coarse CPC-prefix")
    print("diagnostic for the 70 active domain-external families. Table S7a")
    print("uses a curator-audited first-claim partition that is not script-")
    print("generated.")
    print()
    print("Active domain-external families (decision_ledger.csv):", len(domain_external_ids))
    if len(domain_external_ids) != 70:
        print("  Warning: expected 70, found", len(domain_external_ids))
    print()

    # 2) Look up family CPC fields from the released JSON
    with open(JSON_PATH, encoding="utf-8") as f:
        families = json.load(f)

    family_cpc = {}
    for fam in families:
        lens_id = fam.get("lens_id", "")
        if lens_id in domain_external_ids:
            family_cpc[lens_id] = cpc_prefixes(fam.get("cpc", ""))

    # 3) Apply the mapping rule
    counts = OrderedDict((k, 0) for k in PARTITION_RULES.keys())
    counts["Other / curator-flagged"] = 0
    for lens_id in domain_external_ids:
        prefixes = family_cpc.get(lens_id, set())
        cat = categorise(prefixes)
        counts[cat] += 1

    # 4) Print partition
    print("%-32s %10s" % ("Category", "Families"))
    print("-" * 45)
    for cat, n in counts.items():
        print("%-32s %10d" % (cat, n))
    print("-" * 45)
    total = sum(counts.values())
    print("%-32s %10d" % ("Total", total))
    print()
    print("Note. This is a coarse CPC-prefix diagnostic. Online Resource 1")
    print("Table S7a Panel A reports the authors' first-claim audit, which")
    print("redistributes the 'Other / curator-flagged' bucket above between")
    print("Table S7a's 'Boundary-adjacent' (22) and 'Unclassified' (5)")
    print("categories. The 70-family total is reproducible; the curator-")
    print("audited subdivision into Boundary-adjacent vs Unclassified is")
    print("not script-generated.")


if __name__ == "__main__":
    main()
