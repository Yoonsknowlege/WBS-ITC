"""
ISRU Construction Patent Analysis — Core Data Module (v1.0)
============================================================
Paper:   "Denominator construction for patent mapping in classification-poor
          emerging domains: A framework-first WBS–ITC approach with a lunar
          ISRU construction illustration"

Dataset: 453 released patent families (Lens.org, search date 2026-03-28)
         → 443 analytically active (9 false-match records
           and 1 audit-only near-threshold record excluded)
         → 373 ITC-classified + 70 domain-external
Tagging: CPC-primary hybrid ITC method (see ITC_RULES)

This module provides all constants, matrices, and labels needed to
reproduce the figures and tables in the manuscript.
"""

import numpy as np

# ============================================================
# ITC Domain Definitions
# ============================================================
ITC_DOMAINS = ['1-1', '1-2', '1-3', '2-1', '2-2', '2-3', '2-4', '3-1', '3-2', '3-3', '4-1', '4-2', '4-3', '4-4', '4-5']

ITC_LABELS = {
    "1-1": "Regolith Processing",
    "1-2": "Binder/Geopolymer",
    "1-3": "Composite/Ceramic",
    "2-1": "Extrusion AM",
    "2-2": "Powder Bed",
    "2-3": "Solar / Laser Sintering",
    "2-4": "Process Monitoring",
    "3-1": "Autonomous Robots",
    "3-2": "Tele-operation",
    "3-3": "Autonomous Construction",
    "4-1": "Habitat Structures",
    "4-2": "Shielding",
    "4-3": "Landing Pads",
    "4-4": "Deployable Structures",
    "4-5": "Life Support / ECLSS"
}

WBS_COLORS = {
    "1": "#5B9BD5",   # Materials - blue
    "2": "#ED7D31",   # Manufacturing - orange
    "3": "#70AD47",   # Robotics - green
    "4": "#C0504D",   # Structures - red
}

def get_wbs(code):
    return code.split("-")[0]

# ============================================================
# ITC Rules — PRIMARY REPRODUCIBILITY INSTRUMENT
# ============================================================
# Tagging logic (OR-based, evaluated in order):
#   1. If ANY CPC code starts with any prefix in 'cpc' → tagged
#   2. Else if ANY keyword found in title+abstract → tagged
#   3. If no domain matches → 'domain-external'
# Multi-tagging is permitted.
#
# IMPORTANT — Scope of this codebook:
# ITC_RULES encodes the rule-based layer of the tagging methodology.
# The released itc_codes in phase2_453_families.json were produced by
# a two-stage process: (1) this CPC-anchor + keyword rule layer, followed
# by (2) consensus-based author adjudication (see manuscript §3.2 and
# Supplementary Table S6). AI-assisted tools were used only for
# screening-order prioritization; all final inclusion decisions and
# ITC label assignments were made by the authors. Applying ITC_RULES
# alone to the released CPC and title/abstract fields will therefore
# NOT exactly reproduce the released itc_codes. The released tags are
# authoritative; this codebook documents the deterministic first stage
# of the tagging pipeline.

ITC_RULES = {
    '1-1': {
        'name': 'Regolith Processing / Refining',
        'wbs':  'WBS-1 Materials',
        'cpc':  ['C22B', 'B07B', 'B02C', 'B03'],
        'keywords': ['regolith processing', 'beneficiation', 'particle size',
                     'mineral separation', 'excavat', 'regolith refin', 'oxygen extraction']
    },
    '1-2': {
        'name': 'Binder / Geopolymer',
        'wbs':  'WBS-1 Materials',
        'cpc':  ['C04B28', 'C04B12', 'C04B7'],
        'keywords': ['geopolymer', 'binder', 'cement', 'concrete',
                     'sulfur concrete', 'calcium carbonate']
    },
    '1-3': {
        'name': 'Composite / Ceramic / Sintered Bodies',
        'wbs':  'WBS-1 Materials',
        'cpc':  ['C04B35', 'B22F3', 'C04B33'],
        'keywords': ['ceramic', 'sintered body', 'vitrif', 'composite material', 'fiber reinforc']
    },
    '2-1': {
        'name': 'Extrusion-based Additive Manufacturing',
        'wbs':  'WBS-2 Manufacturing',
        'cpc':  ['B33Y10', 'B29C48', 'B28B1', 'B33Y30'],
        'keywords': ['3d print', 'additive manufactur', 'extrusion',
                     'contour crafting', 'layer-by-layer', 'fused deposition']
    },
    '2-2': {
        'name': 'Powder Bed Melting / Sintering',
        'wbs':  'WBS-2 Manufacturing',
        'cpc':  ['B22F10', 'B22F12', 'H05B6'],
        'keywords': ['selective laser sinter', 'SLS', 'SLM',
                     'electron beam melt', 'powder bed', 'microwave sinter']
    },
    '2-3': {
        'name': 'Solar / Laser Sintering',
        'wbs':  'WBS-2 Manufacturing',
        'cpc':  ['B23K26', 'F24S'],
        'keywords': ['solar sinter', 'laser sinter', 'concentrated solar', 'solar furnace']
    },
    '2-4': {
        'name': 'Process Monitoring / NDI',
        'wbs':  'WBS-2 Manufacturing',
        'cpc':  ['G01N', 'G01B', 'B33Y50'],
        'keywords': ['non-destructive', 'NDI', 'quality control',
                     'in-situ monitor', 'process monitor', 'inspect', 'ultrasonic test']
    },
    '3-1': {
        'name': 'Autonomous Mobile Robots',
        'wbs':  'WBS-3 Robotics',
        'cpc':  ['B25J9', 'B62D', 'G05D1'],
        'keywords': ['autonomous robot', 'mobile robot', 'lunar rover',
                     'mars rover', 'robotic vehicle', 'autonomous vehicle', 'navigation']
    },
    '3-2': {
        'name': 'Tele-operation',
        'wbs':  'WBS-3 Robotics',
        'cpc':  ['G06F3', 'H04L'],
        'keywords': ['tele-operat', 'teleoperat', 'remote operat',
                     'remote control', 'telerobot', 'telepresence']
    },
    '3-3': {
        'name': 'Autonomous Construction Systems',
        'wbs':  'WBS-3 Robotics',
        'cpc':  ['G05D1/02', 'E04G21'],
        'keywords': ['autonomous construct', 'swarm construct', 'multi-robot',
                     'automated construct', 'robotic construct', 'automated assembl']
    },
    '4-1': {
        'name': 'Habitat Structures',
        'wbs':  'WBS-4 Structures & Systems',
        'cpc':  ['E04H15', 'B64G1/48', 'E04B1', 'E04H1'],
        'keywords': ['habitat', 'shelter', 'living quarter', 'habitation',
                     'pressurized module', 'base structure', 'dwelling']
    },
    '4-2': {
        'name': 'Shielding Structures',
        'wbs':  'WBS-4 Structures & Systems',
        'cpc':  ['G21F1', 'E04B1/92'],
        'keywords': ['shielding', 'radiation protect', 'micrometeorite',
                     'regolith shield', 'protection structure', 'cosmic ray']
    },
    '4-3': {
        'name': 'Landing Pad / Infrastructure',
        'wbs':  'WBS-4 Structures & Systems',
        'cpc':  ['E01C', 'B64G1/62'],
        'keywords': ['landing pad', 'launch pad', 'runway', 'road',
                     'surface stabiliz', 'plume', 'infrastructure']
    },
    '4-4': {
        'name': 'Deployable Structures',
        'wbs':  'WBS-4 Structures & Systems',
        'cpc':  ['E04H15', 'B64G1/22'],
        'keywords': ['inflatable', 'deployable', 'expandable', 'foldable',
                     'membrane structure', 'tensile structure', 'pneumatic']
    },
    '4-5': {
        'name': 'Life Support / ECLSS',
        'wbs':  'WBS-4 Structures & Systems',
        'cpc':  ['B64G1/46', 'A01G'],
        'keywords': ['life support', 'ECLSS', 'oxygen generat',
                     'water recycl', 'air revitaliz', 'waste process', 'environmental control']
    },
}

# ============================================================
# Dataset Accounting
# ============================================================
# Released dataset: 453 families (phase2_453_families.json)
# Exclusions: 9 false-match + 1 audit-only = 10
# Analytically active: 443 families
#   → 373 ITC-classified + 70 domain-external
# All 10 excluded families were unclassified (no ITC tags);
# ITC tag shares and Jaccard values are mathematically invariant.
# CPC metrics are computed on the 443-active subset because excluded
# families carry CPC codes that shift lower-ranked positions.
# ============================================================
RELEASED_FAMILIES = 453
CATEGORY_C = 9
AUDIT_ONLY = 1
ANALYTICALLY_ACTIVE = 443   # = 453 - 9 - 1
TAGGED_FAMILIES = 373
DOMAIN_EXTERNAL_ACTIVE = 70  # = 443 - 373
DOMAIN_EXTERNAL_RELEASED = 80  # = 453 - 373
TOTAL_ITC_TAGS = 795

# Legacy alias (some scripts may reference this)
TOTAL_FAMILIES = RELEASED_FAMILIES

PORTFOLIO = {
    "1-1": 60,
    "1-2": 53,
    "1-3": 110,
    "2-1": 94,
    "2-2": 43,
    "2-3": 29,
    "2-4": 87,
    "3-1": 22,
    "3-2": 3,
    "3-3": 12,
    "4-1": 146,
    "4-2": 27,
    "4-3": 21,
    "4-4": 65,
    "4-5": 23
}

# ============================================================
# Phase-Two Jaccard Similarity Matrix (15 x 15)
# ============================================================
JACCARD_MATRIX = np.array([
    [1.000,0.076,0.056,0.069,0.030,0.060,0.167,0.065,0.000,0.059,0.108,0.012,0.066,0.033,0.037],
    [0.076,1.000,0.148,0.148,0.011,0.012,0.045,0.014,0.000,0.032,0.137,0.067,0.028,0.063,0.013],
    [0.056,0.148,1.000,0.172,0.077,0.168,0.101,0.015,0.000,0.034,0.164,0.046,0.023,0.101,0.015],
    [0.069,0.148,0.172,1.000,0.202,0.070,0.341,0.027,0.000,0.019,0.101,0.008,0.009,0.026,0.017],
    [0.030,0.011,0.077,0.202,1.000,0.043,0.102,0.016,0.000,0.000,0.044,0.000,0.000,0.009,0.015],
    [0.060,0.012,0.168,0.070,0.043,1.000,0.064,0.062,0.000,0.108,0.061,0.000,0.042,0.011,0.106],
    [0.167,0.045,0.101,0.341,0.102,0.064,1.000,0.028,0.000,0.021,0.084,0.000,0.000,0.013,0.000],
    [0.065,0.014,0.015,0.027,0.016,0.062,0.028,1.000,0.042,0.097,0.050,0.065,0.049,0.048,0.098],
    [0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.042,1.000,0.000,0.000,0.000,0.000,0.000,0.000],
    [0.059,0.032,0.034,0.019,0.000,0.108,0.021,0.097,0.000,1.000,0.039,0.026,0.000,0.000,0.129],
    [0.108,0.137,0.164,0.101,0.044,0.061,0.084,0.050,0.000,0.039,1.000,0.161,0.025,0.294,0.083],
    [0.012,0.067,0.046,0.008,0.000,0.000,0.000,0.065,0.000,0.026,0.161,1.000,0.021,0.260,0.163],
    [0.066,0.028,0.023,0.009,0.000,0.042,0.000,0.049,0.000,0.000,0.025,0.021,1.000,0.062,0.023],
    [0.033,0.063,0.101,0.026,0.009,0.011,0.013,0.048,0.000,0.000,0.294,0.260,0.062,1.000,0.114],
    [0.037,0.013,0.015,0.017,0.015,0.106,0.000,0.098,0.000,0.129,0.083,0.163,0.023,0.114,1.000]
])

# Top 10 Jaccard pairs
TOP_JACCARD_PAIRS = [
    [
        "2-1",
        "2-4",
        0.341
    ],
    [
        "4-1",
        "4-4",
        0.294
    ],
    [
        "4-2",
        "4-4",
        0.26
    ],
    [
        "2-1",
        "2-2",
        0.202
    ],
    [
        "1-3",
        "2-1",
        0.172
    ],
    [
        "1-3",
        "2-3",
        0.168
    ],
    [
        "1-1",
        "2-4",
        0.167
    ],
    [
        "1-3",
        "4-1",
        0.164
    ],
    [
        "4-2",
        "4-5",
        0.163
    ],
    [
        "4-1",
        "4-2",
        0.161
    ]
]

# ============================================================
# CPC Co-classification Network (Top 25 codes)
# ============================================================
CPC_TOP25 = [
    "B64G",
    "B33Y",
    "C04B",
    "G01N",
    "E04B",
    "E02D",
    "E04H",
    "B28B",
    "B22F",
    "B29C",
    "Y02P",
    "E04G",
    "C22C",
    "E21B",
    "E21C",
    "Y02W",
    "E04C",
    "E01C",
    "Y02E",
    "B25J",
    "H02J",
    "Y02A",
    "B02C",
    "F24S",
    "B28C"
]

CPC_COOCCURRENCE = np.array([
    [ 93,  5,  4,  4,  3,  2,  7,  1,  4,  2,  2,  2,  1,  5,  9,  0,  0,  0,  3,  1,  1,  5,  0,  3,  0],
    [  5, 75, 13,  1,  2,  3,  2, 23, 15, 27, 19,  6,  8,  0,  0,  2,  0,  0,  1,  0,  0,  0,  0,  3,  0],
    [  4, 13, 74,  2,  1,  2,  0, 15,  2,  2, 10,  1,  0,  0,  1,  9,  0,  2,  0,  0,  0,  0,  1,  1,  6],
    [  4,  1,  2, 53,  0,  1,  1,  1,  0,  0,  3,  0,  0,  2,  2,  0,  0,  0,  0,  1,  1,  2,  0,  0,  0],
    [  3,  2,  1,  0, 52,  2, 22,  0,  0,  0,  1, 13,  0,  0,  1,  0, 13,  0,  2,  0,  4,  4,  0,  0,  0],
    [  2,  3,  2,  1,  2, 45,  2,  0,  0,  0,  0,  3,  0,  7,  0,  2,  1,  2,  3,  0,  0,  1,  0,  0,  1],
    [  7,  2,  0,  1, 22,  2, 40,  0,  0,  0,  0,  9,  0,  1,  1,  0,  9,  0,  0,  0,  5,  4,  0,  0,  0],
    [  1, 23, 15,  1,  0,  0,  0, 40,  0,  0,  5,  0,  0,  0,  0,  2,  0,  0,  1,  0,  0,  0,  0,  1,  4],
    [  4, 15,  2,  0,  0,  0,  0,  0, 34,  2, 10,  0, 21,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0],
    [  2, 27,  2,  0,  0,  0,  0,  0,  2, 32,  3,  1,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  2,  0],
    [  2, 19, 10,  3,  1,  0,  0,  5, 10,  3, 31,  2,  6,  0,  0,  2,  1,  1,  0,  1,  0,  0,  0,  0,  3],
    [  2,  6,  1,  0, 13,  3,  9,  0,  0,  1,  2, 28,  0,  0,  0,  0,  5,  0,  1,  0,  0,  0,  0,  1,  1],
    [  1,  8,  0,  0,  0,  0,  0,  0, 21,  0,  6,  0, 21,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
    [  5,  0,  0,  2,  0,  7,  1,  0,  1,  0,  0,  0,  1, 15,  6,  0,  0,  0,  1,  1,  2,  1,  0,  0,  0],
    [  9,  0,  1,  2,  1,  0,  1,  0,  0,  0,  0,  0,  0,  6, 14,  0,  0,  1,  0,  0,  2,  0,  0,  1,  0],
    [  0,  2,  9,  0,  0,  2,  0,  2,  0,  2,  2,  0,  0,  0,  0, 14,  0,  1,  0,  0,  0,  0,  2,  0,  2],
    [  0,  0,  0,  0, 13,  1,  9,  0,  0,  0,  1,  5,  0,  0,  0,  0, 14,  0,  0,  0,  1,  0,  0,  0,  0],
    [  0,  0,  2,  0,  0,  2,  0,  0,  0,  0,  1,  0,  0,  0,  1,  1,  0, 13,  0,  0,  0,  1,  0,  0,  1],
    [  3,  1,  0,  0,  2,  3,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  0, 12,  0,  2,  1,  0,  3,  0],
    [  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0, 11,  0,  1,  0,  0,  0],
    [  1,  0,  0,  1,  4,  0,  5,  0,  0,  0,  0,  0,  0,  2,  2,  0,  1,  0,  2,  0, 11,  2,  0,  2,  0],
    [  5,  0,  0,  2,  4,  1,  4,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  1,  1,  2, 11,  0,  0,  0],
    [  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0, 11,  0,  1],
    [  3,  3,  1,  0,  0,  0,  0,  1,  1,  2,  0,  1,  0,  0,  1,  0,  0,  0,  3,  0,  2,  0,  0, 10,  0],
    [  0,  0,  6,  0,  0,  1,  0,  4,  0,  0,  3,  1,  0,  0,  0,  2,  0,  1,  0,  0,  0,  0,  1,  0, 10]
])

# ============================================================
# CPC Bridging Centrality (Table 4, full CPC network, top 10)
# ============================================================
# Degree and betweenness centrality computed from the full CPC
# co-classification network (166-node, all CPC group prefixes in
# the 443-family analytically active dataset), not from the top-25
# subset above.
# Each entry: [CPC code, degree centrality, betweenness centrality]
CPC_BRIDGING = [
    ["B64G", 0.424, 0.245],
    ["E04H", 0.303, 0.110],
    ["G01N", 0.291, 0.154],
    ["E04B", 0.267, 0.087],
    ["B33Y", 0.255, 0.085],
    ["C04B", 0.218, 0.096],
    ["E02D", 0.200, 0.072],
    ["H02J", 0.188, 0.042],
    ["Y02E", 0.182, 0.030],
    ["E21B", 0.170, 0.029],
]
# ============================================================
# Filing-Year × WBS-Layer Tag Counts (for Figure 3)
# ============================================================
# Each row: [year, WBS-1_tags, WBS-2_tags, WBS-3_tags, WBS-4_tags]
# A single family may contribute tags to multiple WBS layers;
# therefore column sums exceed TOTAL_FAMILIES.
FILING_YEAR_WBS = [
    [2000,  0,  0,  0,  2],
    [2002,  1,  0,  1,  1],
    [2006,  0,  0,  0,  4],
    [2007,  1,  0,  1,  1],
    [2008,  1,  0,  1,  7],
    [2009,  0,  0,  0,  2],
    [2010,  0,  0,  0,  3],
    [2011,  2,  0,  0,  1],
    [2012,  1,  0,  0,  2],
    [2013,  0,  1,  0,  0],
    [2014,  0,  0,  1,  7],
    [2015,  0,  0,  0,  1],
    [2016,  3,  1,  0,  4],
    [2017,  8,  4,  1,  1],
    [2018,  4,  8,  0,  7],
    [2019,  9, 10,  4,  9],
    [2020, 14, 13,  3, 22],
    [2021, 22, 30,  5, 32],
    [2022, 21, 37,  3, 37],
    [2023, 30, 38,  6, 29],
    [2024, 48, 44,  3, 50],
    [2025, 48, 62,  6, 51],
    [2026, 10,  5,  2,  9],
]

# Alias for compatibility: some README versions reference CPC_CENTRALITY
CPC_CENTRALITY = None  # Populated below after CPC_BRIDGING

# ============================================================
# Filing-Year × WBS-Layer Tag Counts (by Earliest Priority Year)
# ============================================================
# Same structure as FILING_YEAR_WBS but keyed on earliest_priority_year.
# The manuscript uses earliest priority year for Figure 3.
FILING_YEAR_BY_PRIORITY = [
    [1998,   0,   0,   0,   2],
    [1999,   1,   0,   1,   0],
    [2001,   0,   0,   0,   1],
    [2003,   0,   0,   0,   3],
    [2004,   0,   0,   0,   3],
    [2005,   0,   0,   1,   5],
    [2006,   1,   0,   1,   3],
    [2007,   1,   0,   0,   4],
    [2009,   2,   0,   0,   0],
    [2010,   2,   0,   0,   3],
    [2011,   0,   1,   0,   0],
    [2012,   0,   0,   0,   1],
    [2013,   0,   0,   1,   7],
    [2014,   1,   0,   0,   2],
    [2015,   3,   1,   0,   0],
    [2016,   5,   3,   2,   4],
    [2017,   7,   5,   1,   9],
    [2018,   6,  10,   1,  14],
    [2019,  15,  14,   4,  19],
    [2020,  13,  18,   4,  10],
    [2021,  22,  35,   2,  51],
    [2022,  16,  30,   3,  17],
    [2023,  43,  47,   6,  42],
    [2024,  56,  59,   5,  55],
    [2025,  29,  30,   5,  27],
]

# Alias
CPC_CENTRALITY = CPC_BRIDGING
