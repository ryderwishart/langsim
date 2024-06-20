import math
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import kendalltau, ks_2samp, entropy
from scipy.spatial.distance import jensenshannon
from .preprocessing import preprocess_samples
from .constants import METRIC_DICT
from .config import DEBUG_MODE

def calculate_distortion(sample1: List[str], sample2: List[str]) -> float:
    if len(sample1) == 1 and len(sample2) == 1:
        return 1.0  # Perfect alignment for single-line comparison
    return kendalltau(range(len(sample1)), range(len(sample2)))[0]

def calculate_length_ratio_similarity(sample1: List[str], sample2: List[str]) -> float:
    length_ratios = [len(line1) / len(line2) if len(line2) > 0 else 1.0 for line1, line2 in zip(sample1, sample2)]
    return math.sqrt(sum((ratio - 1) ** 2 for ratio in length_ratios) / len(length_ratios))

def calculate_whitespace_metrics(sample1: List[str], sample2: List[str]) -> Tuple[float, float]:
    ws_ratio1 = sum(c.isspace() for line in sample1 for c in line) / max(sum(len(line) for line in sample1), 1)
    ws_ratio2 = sum(c.isspace() for line in sample2 for c in line) / max(sum(len(line) for line in sample2), 1)
    ws_ratio_diff = abs(ws_ratio1 - ws_ratio2)
    
    ws1 = [c.isspace() for line in sample1 for c in line]
    ws2 = [c.isspace() for line in sample2 for c in line]
    if len(ws1) == 0 or len(ws2) == 0:
        ws_ks_stat = 1.0  # Maximum difference if one or both samples are empty
    else:
        ws_ks_stat = ks_2samp(ws1, ws2).statistic
    return ws_ratio_diff, ws_ks_stat

def calculate_punctuation_distribution(sample1: List[str], sample2: List[str]) -> float:
    punct_classes = {'.!?': 'sentence-final', ',;:': 'clause-separating', '-–—': 'word-separating'}
    
    def get_punct_dist(sample):
        dist = {cls: sum(1 for line in sample for c in line if c in chars) 
                for chars, cls in punct_classes.items()}
        total = sum(dist.values())
        return {k: v / total if total > 0 else 0 for k, v in dist.items()}
    
    punct_class_dist1 = get_punct_dist(sample1)
    punct_class_dist2 = get_punct_dist(sample2)
    return jensenshannon(list(punct_class_dist1.values()), list(punct_class_dist2.values()))

def calculate_entropy_diff(sample1: List[str], sample2: List[str]) -> float:
    def text_to_prob(text):
        char_counts = np.array([text.count(c) for c in set(text)])
        return char_counts / len(text) if len(text) > 0 else np.array([])

    prob1 = text_to_prob(''.join(sample1))
    prob2 = text_to_prob(''.join(sample2))
    
    if len(prob1) == 0 and len(prob2) == 0:
        return 0.0
    elif len(prob1) == 0 or len(prob2) == 0:
        return 1.0
    
    return abs(entropy(prob1, base=2) - entropy(prob2, base=2))

def calculate_lexical_similarity(sample1: List[str], sample2: List[str]) -> float:
    words1 = set(word for line in sample1 for word in line.split())
    words2 = set(word for line in sample2 for word in line.split())
    if not words1 and not words2:
        return 1.0  # Both samples are empty, consider them similar
    return len(words1 & words2) / len(words1 | words2)

def calculate_cognate_metrics(rom_sample1: List[str], rom_sample2: List[str]) -> Tuple[float, float]:
    cognates = [w1 == w2 for line1, line2 in zip(rom_sample1, rom_sample2) 
                for w1, w2 in zip(line1.split(), line2.split())]
    
    if not cognates:
        return 0.0, 1.0  # No cognates found, return minimum similarity and maximum distortion
    
    cognate_prop = sum(cognates) / len(cognates)
    
    cognate_order1 = [i for i, cog in enumerate(cognates) if cog]
    cognate_order2 = list(range(len(cognate_order1)))
    
    if len(cognate_order1) <= 1:
        cognate_distortion = 1.0  # Perfect alignment for 0 or 1 cognate
    else:
        cognate_distortion = kendalltau(cognate_order1, cognate_order2)[0]
    
    return cognate_prop, cognate_distortion

def calculate_morphological_complexity(rom_sample1: List[str], rom_sample2: List[str], max_token_length: int = 6) -> float:
    def get_tokens(sample):
        return {word[i:j] for line in sample for word in line.split() 
                for i in range(len(word)) for j in range(i+1, min(i+max_token_length+1, len(word)+1))}
    
    tokens1 = get_tokens(rom_sample1)
    tokens2 = get_tokens(rom_sample2)
    
    if not tokens1 and not tokens2:
        return 1.0  # Both samples are empty, consider them similar
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

def calculate_overall_similarity(scores: Dict[str, float]) -> float:
    weights = {
        'Dist': 0.15, 'LenRat': 0.05, 'WSDiff': 0.05, 'WSKS': 0.05,
        'PunctJS': 0.1, 'EntDiff': 0.1, 'LexSim': 0.2,
        'CogProp': 0.15, 'MorphComp': 0.1, 'CogDist': 0.05
    }
    
    normalized_scores = {
        'Dist': 1 - abs(scores['Dist']),
        'LenRat': 1 - scores['LenRat'],
        'WSDiff': 1 - scores['WSDiff'],
        'WSKS': 1 - scores['WSKS'],
        'PunctJS': 1 - scores['PunctJS'],
        'EntDiff': 1 - min(scores['EntDiff'], 1),
        'LexSim': scores['LexSim'],
        'CogProp': scores['CogProp'],
        'MorphComp': scores['MorphComp'],
        'CogDist': 1 - abs(scores['CogDist'])
    }
    
    return sum(weights[metric] * normalized_scores[metric] for metric in weights)

def compare_languages(sample1: List[str], sample2: List[str], max_token_length: int = 6, debug: bool = None) -> Dict[str, float]:
    # Use the global DEBUG_MODE if debug is not explicitly set
    debug = DEBUG_MODE if debug is None else debug
    
    # print(f"DEBUG passed into compare_languages: debug: {debug}") - see FIXME in config.py
    if debug:
        print(f"DEBUG: Comparing samples:\n{sample1}\n{sample2}")
        print(f"DEBUG: max_token_length: {max_token_length}")
    
    # Ensure samples are lists
    sample1 = [sample1] if isinstance(sample1, str) else sample1
    sample2 = [sample2] if isinstance(sample2, str) else sample2

    sample1, sample2, rom_sample1, rom_sample2 = preprocess_samples(sample1, sample2)
    
    distortion = calculate_distortion(sample1, sample2)
    length_ratio_std = calculate_length_ratio_similarity(sample1, sample2)
    ws_ratio_diff, ws_ks_stat = calculate_whitespace_metrics(sample1, sample2)
    punct_js_div = calculate_punctuation_distribution(sample1, sample2)
    entropy_diff = calculate_entropy_diff(sample1, sample2)
    lexical_sim = calculate_lexical_similarity(sample1, sample2)
    cognate_prop, cognate_distortion = calculate_cognate_metrics(rom_sample1, rom_sample2)
    morph_complexity = calculate_morphological_complexity(rom_sample1, rom_sample2, max_token_length)
    
    results = {
        METRIC_DICT['Distortion']: distortion,
        METRIC_DICT['Length ratio std']: length_ratio_std,
        METRIC_DICT['Whitespace ratio diff']: ws_ratio_diff,
        METRIC_DICT['Whitespace KS statistic']: ws_ks_stat,
        METRIC_DICT['Punctuation JS divergence']: punct_js_div,
        METRIC_DICT['Entropy diff']: entropy_diff,
        METRIC_DICT['Lexical similarity']: lexical_sim,
        METRIC_DICT['Cognate proportion']: cognate_prop,
        METRIC_DICT['Morphological complexity']: morph_complexity,
        METRIC_DICT['Cognate-based distortion']: cognate_distortion,
        METRIC_DICT['Overall Similarity']: calculate_overall_similarity(results)
    }
    
    if debug:
        print("DEBUG: Calculated metrics:")
        for metric, value in results.items():
            print(f"  {metric}: {value}")
    
    return results
