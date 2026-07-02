"""
Rule-hit / assignment agreement (Online Resource 1, Table S9).

Computes family-domain-tag-level agreement between the deterministic
rule-hit layers and the claim-reviewed WBS-ITC assignment for:
  - CPC-rule only
  - Keyword-rule only
  - CPC-or-keyword (rule union)

Reference set: the claim-reviewed WBS-ITC assignment (795 tags across
373 ITC-classified families). The quantities below are internal
layer-comparison counts, not classifier-performance metrics.

Column glossary (printed below):
  Shared tags              = cells present in both the rule-hit layer and the assignment
  Rule-only tags           = rule-positive cells not retained after claim review
  Rule-positive tags       = shared + rule-only
  Assignment-tag agreement = shared / total assignment tags
  Rule-positive agreement  = shared / rule-positive
  Assignment-only tags     = assignment tags with no corresponding rule hit
"""

import csv
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def parse_domains(s):
    """Parse semicolon-delimited domain string into a set."""
    if not s or s.strip() == '':
        return set()
    return set(s.strip().split(';'))


def main():
    table_path = os.path.join(DATA_DIR, 'assignment_table.csv')

    with open(table_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    active = [r for r in rows if r['analysis_set_443'] == 'true']
    classified = [r for r in active if r['is_classified'] == 'True']

    print("Analytically active families: {}".format(len(active)))
    print("ITC-classified families: {}".format(len(classified)))
    print()

    cpc_shared = cpc_ruleonly = 0
    kw_shared = kw_ruleonly = 0
    union_shared = union_ruleonly = union_asgonly = 0

    for r in classified:
        itc = parse_domains(r['itc_codes'])
        cpc = parse_domains(r['cpc_rule_domains'])
        kw = parse_domains(r['keyword_rule_domains'])
        union = cpc | kw
        cpc_shared += len(itc & cpc)
        cpc_ruleonly += len(cpc - itc)
        kw_shared += len(itc & kw)
        kw_ruleonly += len(kw - itc)
        union_shared += len(itc & union)
        union_ruleonly += len(union - itc)
        union_asgonly += len(itc - union)

    total_tags = sum(len(parse_domains(r['itc_codes'])) for r in classified)

    print("=" * 96)
    print("Online Resource 1, Table S9: Rule-hit / assignment agreement")
    print("(N = {} ITC-classified families, {} final WBS-ITC tags)".format(len(classified), total_tags))
    print("=" * 96)
    print()
    print("{:<28} {:>7} {:>10} {:>9} {:>14} {:>15} {:>12}".format(
        'Layer', 'Shared', 'Rule-only', 'Rule-pos', 'Asg-tag agree', 'Rule-pos agree', 'Asg-only'))
    print("-" * 96)

    def fmt_row(name, shared, rule_only, show_asg=False):
        rule_pos = shared + rule_only
        asg_agree = shared / total_tags * 100
        rpa = shared / rule_pos * 100 if rule_pos > 0 else 0
        asg_str = "{} ({:.1f}%)".format(union_asgonly, union_asgonly / total_tags * 100) if show_asg else "-"
        print("{:<28} {:>7} {:>10} {:>9} {:>12.1f}% {:>13.1f}% {:>12}".format(
            name, shared, rule_only, rule_pos, asg_agree, rpa, asg_str))

    fmt_row("CPC-rule only", cpc_shared, cpc_ruleonly)
    fmt_row("Keyword-rule only", kw_shared, kw_ruleonly)
    fmt_row("CPC-or-keyword rule union", union_shared, union_ruleonly, show_asg=True)

    pct_shared = union_shared / total_tags * 100
    pct_only = union_asgonly / total_tags * 100
    print()
    print("Of {} claim-reviewed WBS-ITC tags, {} also appear in the rule-hit".format(total_tags, union_shared))
    print("layer ({:.1f}%); {} tags ({:.1f}%) appear only after claim review.".format(pct_shared, union_asgonly, pct_only))
    print()
    print("Note: shared / rule-only / assignment-only are internal family-domain")
    print("layer-comparison counts between the rule-hit layer and the WBS-ITC")
    print("assignment, not classifier-performance metrics. They are restricted to")
    print("the 373-family ITC-classified subset and therefore differ from the")
    print("6,645-cell active-layer accounting in Table S6a.")


if __name__ == '__main__':
    main()
