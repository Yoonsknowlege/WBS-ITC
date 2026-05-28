#!/usr/bin/env python3
"""
Add adjudication_action column to decision_ledger.csv based on ITC codes
and rule domains comparison logic.
"""

import csv
from pathlib import Path


def parse_domains(domain_str):
    """Parse semicolon-separated domains into a set. Empty strings return empty set."""
    if not domain_str or domain_str.strip() == '':
        return set()
    return set(d.strip() for d in domain_str.split(';') if d.strip())


def determine_adjudication_action(row):
    """
    Determine the adjudication action for a row based on the comparison logic.

    Logic:
    - For analytically active families (analysis_set_443 == 'true'):
      - Compare itc_codes with rule union (cpc_rule_domains or keyword_rule_domains)
      - If itc_codes == rule union exactly: "keep"
      - If itc_codes has domains NOT in rule union (additions): include "add"
      - If rule union has domains NOT in itc_codes (suppressions): include "suppress"
      - If both additions and suppressions: "add+suppress"
      - If family has no ITC codes but had rule hits: "suppress_all"
      - If family has ITC codes but no rule hits: "add_only"
    - For non-active families: leave empty
    """

    # Check if this is an analytically active family
    is_active = row.get('analysis_set_443', '').lower() == 'true'

    if not is_active:
        return ''

    # Parse the domain sets
    itc_codes = parse_domains(row.get('itc_codes', ''))
    cpc_rule_domains = parse_domains(row.get('cpc_rule_domains', ''))
    keyword_rule_domains = parse_domains(row.get('keyword_rule_domains', ''))

    # Create the rule union
    rule_union = cpc_rule_domains | keyword_rule_domains

    # Check for additions (itc_codes not in rule_union)
    additions = itc_codes - rule_union

    # Check for suppressions (rule_union not in itc_codes)
    suppressions = rule_union - itc_codes

    # Determine action based on the logic
    has_itc = len(itc_codes) > 0
    has_rule = len(rule_union) > 0
    has_additions = len(additions) > 0
    has_suppressions = len(suppressions) > 0

    # Case: no ITC codes but had rule hits
    if not has_itc and has_rule:
        return 'suppress_all'

    # Case: has ITC codes but no rule hits
    if has_itc and not has_rule:
        return 'add_only'

    # Case: exact match
    if not has_additions and not has_suppressions:
        return 'keep'

    # Case: both additions and suppressions
    if has_additions and has_suppressions:
        return 'add+suppress'

    # Case: only additions
    if has_additions:
        return 'add'

    # Case: only suppressions
    if has_suppressions:
        return 'suppress'

    # Default (should not reach here for active families)
    return ''


def main():
    csv_path = Path('/sessions/peaceful-youthful-darwin/mnt/20260329_isru/제출-acta/제출_Scientometrics/WBS-ITC/data/decision_ledger.csv')

    # Read the CSV file
    rows = []
    fieldnames = None

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Add the new column to fieldnames if not already present
    if 'adjudication_action' not in fieldnames:
        fieldnames.append('adjudication_action')

    # Process each row and add adjudication_action
    for row in rows:
        row['adjudication_action'] = determine_adjudication_action(row)

    # Write the CSV file back
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Added adjudication_action column to {csv_path}")
    print(f"✓ Total rows processed: {len(rows)}")

    # Print summary of action distribution
    action_counts = {}
    for row in rows:
        action = row.get('adjudication_action', '')
        action_counts[action] = action_counts.get(action, 0) + 1

    print("\nAdjudication Action Distribution:")
    print("-" * 50)
    for action in sorted(action_counts.keys()):
        count = action_counts[action]
        pct = (count / len(rows)) * 100
        print(f"  {action if action else '(empty)':<20} {count:>6} ({pct:>5.1f}%)")
    print("-" * 50)


if __name__ == '__main__':
    main()
