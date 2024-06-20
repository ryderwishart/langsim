import math
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import kendalltau, ks_2samp, entropy
from scipy.spatial.distance import jensenshannon
from .preprocessing import preprocess_samples
from .constants import METRIC_DICT
from .config import DEBUG_MODE
import functools

def cache_result(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in wrapper.cache:
            wrapper.cache[key] = func(*args, **kwargs)
        return wrapper.cache[key]
    wrapper.cache = {}
    return wrapper

@cache_result
def calculate_whitespace_metrics(sample1: List[str], sample2: List[str]) -> Tuple[float, float]:
    ws_ratio1 = sum(line.count(' ') for line in sample1) / max(sum(len(line) for line in sample1), 1)
    ws_ratio2 = sum(line.count(' ') for line in sample2) / max(sum(len(line) for line in sample2), 1)
    ws_ratio_diff = abs(ws_ratio1 - ws_ratio2)
    
    ws1 = np.array([c == ' ' for line in sample1 for c in line])
    ws2 = np.array([c == ' ' for line in sample2 for c in line])
    if len(ws1) == 0 or len(ws2) == 0:
        ws_ks_stat = 1.0
    else:
        ws_ks_stat = ks_2samp(ws1, ws2).statistic
    return ws_ratio_diff, ws_ks_stat

@cache_result
def calculate_punctuation_distribution(sample1: List[str], sample2: List[str]) -> float:
    punct_classes = {'.!?': 'sentence-final', ',;:': 'clause-separating', '-–—': 'word-separating'}
    
    def get_punct_dist(sample):
        dist = {cls: sum(sum(c in chars for c in line) for line in sample) for chars, cls in punct_classes.items()}
        total = sum(dist.values())
        return {k: v / total if total > 0 else 0 for k, v in dist.items()}
    
    punct_class_dist1 = get_punct_dist(sample1)
    punct_class_dist2 = get_punct_dist(sample2)
    
    if all(v == 0 for v in punct_class_dist1.values()) and all(v == 0 for v in punct_class_dist2.values()):
        return 0.0
    
    dist1 = np.array(list(punct_class_dist1.values()))
    dist2 = np.array(list(punct_class_dist2.values()))
    
    if np.sum(dist1) == 0 or np.sum(dist2) == 0:
        return 1.0
    
    return jensenshannon(dist1, dist2)

@cache_result
def calculate_entropy_diff(sample1: List[str], sample2: List[str]) -> float:
    def text_to_prob(text):
        char_counts = np.bincount(np.frombuffer(text.encode(), dtype=np.uint8))
        return char_counts[char_counts > 0] / len(text)

    text1 = ''.join(sample1)
    text2 = ''.join(sample2)
    
    if not text1 and not text2:
        return 0.0
    elif not text1 or not text2:
        return 1.0
    
    prob1 = text_to_prob(text1)
    prob2 = text_to_prob(text2)
    
    return abs(entropy(prob1, base=2) - entropy(prob2, base=2))

@cache_result
def calculate_lexical_similarity(sample1: List[str], sample2: List[str]) -> float:
    words1 = set(' '.join(sample1).split())
    words2 = set(' '.join(sample2).split())
    if not words1 and not words2:
        return 1.0
    return len(words1 & words2) / len(words1 | words2)

@cache_result
def calculate_distortion(sample1: List[str], sample2: List[str]) -> float:
    if len(sample1) == 1 and len(sample2) == 1:
        return 1.0  # Perfect alignment for single-line comparison
    return kendalltau(range(len(sample1)), range(len(sample2)))[0]

@cache_result
def calculate_length_ratio_similarity(sample1: List[str], sample2: List[str]) -> float:
    length_ratios = [len(line1) / len(line2) if len(line2) > 0 else 1.0 for line1, line2 in zip(sample1, sample2)]
    return math.sqrt(sum((ratio - 1) ** 2 for ratio in length_ratios) / len(length_ratios))

@cache_result
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

@cache_result
def calculate_morphological_complexity(rom_sample1: List[str], rom_sample2: List[str], max_token_length: int = 6) -> float:
    def get_tokens(sample):
        return {word[i:j] for line in sample for word in line.split() 
                for i in range(len(word)) for j in range(i+1, min(i+max_token_length+1, len(word)+1))}
    
    tokens1 = get_tokens(rom_sample1)
    tokens2 = get_tokens(rom_sample2)
    
    if not tokens1 and not tokens2:
        return 1.0  # Both samples are empty, consider them similar
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

@cache_result
def calculate_overall_similarity(scores: Dict[str, float], custom_weights: Dict[str, float] = None) -> float:
    default_weights = {
        'Dist': 0.15, 'LenRat': 0.05, 'WSDiff': 0.05, 'WSKS': 0.05,
        'PunctJS': 0.1, 'EntDiff': 0.1, 'LexSim': 0.2,
        'CogProp': 0.15, 'MorphComp': 0.1, 'CogDist': 0.05
    }
    
    if custom_weights is None:
        weights = default_weights
    else:
        weights = {**default_weights, **custom_weights}
    
    if DEBUG_MODE:
        print(f"DEBUG: Calculating overall similarity with weights: {weights}")
    
    normalized_scores = {
        # TODO: I think we could be a bit more sophisticated in what we allow a user/bot to pass in.
        # For example, it probably makes sense to allow a user/bot to pass in a function along with a
        # constant, instead of what we currently have, which is just a constant that gets applied to
        # a normalization function defined here.
        'Dist': 1 - abs(scores['Dist']),
        'LenRat': 1 / (1 + scores['LenRat']),  # Changed to avoid negative values
        'WSDiff': 1 - scores['WSDiff'],
        'WSKS': 1 - scores['WSKS'],
        'PunctJS': 1 - scores['PunctJS'] if not np.isnan(scores['PunctJS']) else 1.0,  # Handle nan
        'EntDiff': 1 / (1 + scores['EntDiff']),  # Changed to avoid negative values
        'LexSim': scores['LexSim'],
        'CogProp': scores['CogProp'],
        'MorphComp': scores['MorphComp'],
        'CogDist': 1 - abs(scores['CogDist'])
    }
    
    # Filter out nan values
    valid_scores = {k: v for k, v in normalized_scores.items() if not np.isnan(v)}
    valid_weights = {k: weights[k] for k in valid_scores}
    
    # Normalize weights
    weight_sum = sum(valid_weights.values())
    normalized_weights = {k: v / weight_sum for k, v in valid_weights.items()}
    
    return sum(normalized_weights[metric] * valid_scores[metric] for metric in valid_scores)

@cache_result # Might not need to cache here, but in some cases it's useful
def compare_languages(sample1: List[str], sample2: List[str], max_token_length: int = 6, debug: bool = None, custom_weights: Dict[str, float] = None) -> Dict[str, float]:
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
    }
    
    # Calculate overall similarity after creating the results dictionary
    results[METRIC_DICT['Overall Similarity']] = calculate_overall_similarity(results, custom_weights)
    
    if debug:
        print("DEBUG: Calculated metrics:")
        for metric, value in results.items():
            print(f"  {metric}: {value}")
    
    return results

