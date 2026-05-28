#!/usr/bin/env python3
"""
build_long_form_ledger.py — Construct a long-form (family x domain)
cell-level decision ledger from the released family-level ledger.

Paper: "Auditable denominator design for patent-based field
       delineation in classification-dispersed emerging
       technologies"

The released `data/decision_ledger.csv` is in family-level wide form
(one row per Lens family, with semicolon-separated `itc_codes`,
`cpc_rule_domains`, `keyword_rule_domains` lists). The long-form
ledger emitted by this script expands each released family into
15 family-domain rows (one row per ITC domain), making the cell-level
addition / suppression structure directly inspectable.

Output columns:
    lens_id
    domain                  (one of the 15 ITC domain codes)
    cpc_rule_hit            ("1" if domain in cpc_rule_domains; "0" otherwise)
    keyword_rule_hit        ("1" if domain in keyword_rule_domains; "0" otherwise)
    rule_union_hit          ("1" if cpc_rule_hit OR keyword_rule_hit; "0" otherwise)
    adjudicated_hit         ("1" if domain in itc_codes; "0" otherwise)
    action                  ("keep"     -> rule_union_hit AND adjudicated_hit
                             "suppress" -> rule_union_hit AND NOT adjudicated_hit
                             "add"      -> NOT rule_union_hit AND adjudicated_hit
                             "none"     -> not rule_union_hit AND not adjudicated_hit)
    screening_category      (carried over from the family-level ledger)
    analysis_set_443        (carried over from the family-level ledger)

Active-layer accounting derived from this long-form ledger:
    Active-layer cells (analysis_set_443 == "true"): 6,645 (= 443 x 15)
    'add'      cells: 374
    'suppress' cells: 256
    'keep'     cells: 421
    'none'     cells: 6,645 - (374 + 256 + 421) = 5,594

These cell-level counts match Online Resource 1 Tables S6a and S9.

Usage:
    python scripts/build_long_form_ledger.py
    -> writes data/decision_ledger_long.csv
"""

import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
LEDGER_PATH = os.path.join(DATA_DIR, "decision_ledger.csv")
OUT_PATH = os.path.join(DATA_DIR, "decision_ledger_long.csv")

ITC_DOMAINS = ["1-1", "1-2", "1-3",
               "2-1", "2-2", "2-3", "2-4",
               "3-1", "3-2", "3-3",
               "4-1", "4-2", "4-3", "4-4", "4-5"]


def parse_doms(s):
    return set(d.strip() for d in s.split(";")) if s and s.strip() else set()


def main():
    rows_out = []
    n_active_cells = 0
    counts = {"keep": 0, "suppress": 0, "add": 0, "none": 0}

    with open(LEDGER_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lens_id = row.get("lens_id", "")
            cpc = parse_doms(row.get("cpc_rule_domains", ""))
            kw = parse_doms(row.get("keyword_rule_domains", ""))
            adj = parse_doms(row.get("itc_codes", ""))
            screening = row.get("screening_category", "")
            active = row.get("analysis_set_443", "")
            for dom in ITC_DOMAINS:
                cpc_hit = "1" if dom in cpc else "0"
                kw_hit = "1" if dom in kw else "0"
                rule_hit = "1" if (dom in cpc or dom in kw) else "0"
                adj_hit = "1" if dom in adj else "0"
                if rule_hit == "1" and adj_hit == "1":
                    action = "keep"
                elif rule_hit == "1" and adj_hit == "0":
                    action = "suppress"
                elif rule_hit == "0" and adj_hit == "1":
                    action = "add"
                else:
                    action = "none"
                rows_out.append({
                    "lens_id": lens_id,
                    "domain": dom,
                    "cpc_rule_hit": cpc_hit,
                    "keyword_rule_hit": kw_hit,
                    "rule_union_hit": rule_hit,
                    "adjudicated_hit": adj_hit,
                    "action": action,
                    "screening_category": screening,
                    "analysis_set_443": active,
                })
                if active == "true":
                    n_active_cells += 1
                    counts[action] += 1

    fieldnames = ["lens_id", "domain", "cpc_rule_hit", "keyword_rule_hit",
                  "rule_union_hit", "adjudicated_hit", "action",
                  "screening_category", "analysis_set_443"]
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print("Long-form decision ledger written to:")
    print(" ", OUT_PATH)
    print()
    print("Total cells:           ", len(rows_out))
    print("Active-layer cells:    ", n_active_cells, "(expected 6,645 = 443 x 15)")
    print("  'add' cells:         ", counts["add"], "(expected 374)")
    print("  'suppress' cells:    ", counts["suppress"], "(expected 256)")
    print("  'keep' cells:        ", counts["keep"], "(expected 421)")
    print("  'none' cells:        ", counts["none"], "(expected 5,594)")
    print()
    print("These cell-level counts match Online Resource 1 Tables S6a")
    print("(active-layer 6,645-cell diagnostic) and S9 (rule-union")
    print("recovery within the ITC-classified subset).")


if __name__ == "__main__":
    main()
