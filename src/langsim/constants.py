from typing import Dict, List, Tuple

METRIC_NAMES: List[Tuple[str, str]] = [
    ('Line', 'Line'),
    ('Distortion', 'Dist'),
    ('Length ratio std', 'LenRat'),
    ('Whitespace ratio diff', 'WSDiff'),
    ('Whitespace KS statistic', 'WSKS'),
    ('Punctuation JS divergence', 'PunctJS'),
    ('Entropy diff', 'EntDiff'),
    ('Lexical similarity', 'LexSim'),
    ('Cognate proportion', 'CogProp'),
    ('Cognate-based distortion', 'CogDist'),
    ('Morphological complexity', 'MorphComp'),
    ('Overall Similarity', 'OverallSim')
]

METRIC_DICT: Dict[str, str] = dict(METRIC_NAMES)
METRIC_ABBR_DICT: Dict[str, str] = {v: k for k, v in METRIC_DICT.items()}