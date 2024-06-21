import sys
import os
import datetime
from typing import Dict, List, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langsim import compare_languages
from rich.console import Console
from rich.table import Table
from itertools import combinations
import json
from openai import OpenAI
from multiprocessing import Pool
from functools import lru_cache
import numpy as np
from collections import Counter

client = OpenAI()

try:
    import openai
except ImportError:
    print("OpenAI is not installed. Please install it using `pip install openai`.")
    exit()
    
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"comparison_results_{timestamp}.json"
print(f"Streaming results to output file: {output_file}")

# Define sentences in various languages (same as in compare_major_languages.py)
languages: Dict[str, List[str]] = {
    "English": [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question."
    ],
    "Spanish": [
        "El rápido zorro marrón salta sobre el perro perezoso.",
        "Un viaje de mil millas comienza con un solo paso.",
        "Ser o no ser, esa es la cuestión."
    ],
    "Mandarin": [
        "快速的棕色狐狸跳过懒惰的狗。",
        "千里之行，始于足下。",
        "生存还是毁灭，这是一个问题。"
    ],
    "Arabic": [
        "الثعلب البني السريع يقفز فوق الكلب الكسول.",
        "رحلة الألف ميل تبدأ بخطوة واحدة.",
        "أن تكون أو لا تكون، هذا هو السؤال."
    ],
    "Swahili": [
        "Mbweha mwepesi wa kahawia anaruka juu ya mbwa mvivu.",
        "Safari ya maili elfu huanza na hatua moja.",
        "Kuwa au kutokuwa, hiyo ndiyo swali."
    ],
    "Hindi": [
        "तेज़ भूरी लोमड़ी आलसी कुत्ते पर कूदती है।",
        "हजार मील की यात्रा एक कदम से शुरू होती है।",
        "होना या न होना, यही सवाल है।"
    ],
    "Russian": [
        "Быстрая коричневая лиса прыгает через ленивую собаку.",
        "Путешествие в тысячу миль начинается с одного шага.",
        "Быть или не быть, вот в чем вопрос."
    ],
    "Japanese": [
        "素早い茶色のキツネは怠けた犬を飛び越えます。",
        "千里の道も一歩から。",
        "生きるべきか死ぬべきか、それが問題だ。"
    ],
    "French": [
        "Le rapide renard brun saute par-dessus le chien paresseux.",
        "Un voyage de mille lieues commence par un seul pas.",
        "Être ou ne pas être, telle est la question."
    ],
    "German": [
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Eine Reise von tausend Meilen beginnt mit einem einzigen Schritt.",
        "Sein oder nicht sein, das ist hier die Frage."
    ],
    "RandomNoise1": [
        "asdkjfhaskjdfh eueiheeuihe eeueiheuie",
        "qwertyuiop asdfghjkl zxcvbnm",
        "1234567890 !@#$%^&*()"
    ],
    "RandomNoise2": [
        "28929828 298329 8739.389283 982 / 238 98 * 3289",
        "0987654321 1234567890 0987654321",
        "!!! ??? *** $$$"
    ],
    "EnglishALT1": [
        "The fast brown fox jumped o'er the tired dog.",
        "A long journey starts with a single step.",
        "To exist or not to exist, that is the query."
    ],
    "EnglishALT2": [
        "The slow red cat rolled over the small pig.",
        "A short trip begins with a single stride.",
        "To live or not to live, that is the dilemma."
    ]
}

# Define expected outcomes (you can adjust these based on linguistic knowledge)
expected_outcomes: Dict[str, float] = {
    "English-German": 0.7,
    "English-French": 0.6,
    "English-Spanish": 0.5,
    "English-Russian": 0.4,
    "English-Hindi": 0.2,
    "English-Arabic": 0.1,
    "English-Mandarin": 0.05,
    "Spanish-French": 0.6,
    "German-Russian": 0.3,
    "Mandarin-Japanese": 0.4,
    "RandomNoise1-RandomNoise2": 0.0,
    "English-RandomNoise1": 0.0,
    "EnglishALT1-EnglishALT2": 0.95
}

@lru_cache(maxsize=None)
def precompute_language_data(sample):
    words = set(' '.join(sample).split())
    char_dist = np.bincount(np.frombuffer(''.join(sample).encode(), dtype=np.uint8))
    return words, char_dist

def compute_base_metrics_parallel(languages):
    with Pool() as pool:
        results = pool.starmap(compute_pair_metrics, combinations(languages.items(), 2))
    return dict(results)

def compute_pair_metrics(lang1_data, lang2_data):
    lang1, sents1 = lang1_data
    lang2, sents2 = lang2_data
    scores = compare_languages(sents1, sents2, custom_weights=None)
    return (f"{lang1}-{lang2}", {k: v for k, v in scores.items() if k != 'OverallSim'})

def apply_weights(base_metrics: Dict[str, Dict[str, float]], custom_weights: Dict[str, float]) -> Dict[str, float]:
    results = {}
    for lang_pair, metrics in base_metrics.items():
        overall_sim = sum(custom_weights[metric] * value for metric, value in metrics.items())
        results[lang_pair] = overall_sim
    return results

def run_comparison(base_metrics: Dict[str, Dict[str, float]], custom_weights: Dict[str, float] = None) -> Dict[str, float]:
    if custom_weights is None:
        custom_weights = {
            'Dist': 0.15, 'LenRat': 0.05, 'WSDiff': 0.05, 'WSKS': 0.05,
            'PunctJS': 0.1, 'EntDiff': 0.1, 'LexSim': 0.2,
            'CogProp': 0.15, 'MorphComp': 0.1, 'CogDist': 0.05
        }
    return apply_weights(base_metrics, custom_weights)

def display_results(results: Dict[str, float]) -> None:
    table = Table(title="Language Similarity Scores")
    table.add_column("Language Pair", style="cyan", no_wrap=True)
    table.add_column("Overall Similarity", style="magenta")
    for lang_pair, similarity in sorted(results.items(), key=lambda x: x[1], reverse=True):
        table.add_row(lang_pair, f"{similarity:.4f}")
    console = Console()
    console.print(table)

def get_llm_suggestion(results: Dict[str, float], expected_outcomes: Dict[str, float], current_weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    prompt = f"""
    You are an expert linguist and programmer. Analyze the following language comparison results and suggest updates to the weights used in the comparison algorithm.

    Metrics under consideration:
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

    Current weights: {json.dumps(current_weights, indent=2)}

    Comparison results: {json.dumps(results, indent=2)}

    Expected outcomes for key language pairs: {json.dumps(expected_outcomes, indent=2)}

    Based on the differences between the results and expected outcomes, suggest updates to the weights. Provide your response as a JSON object with the weights to be updated. Only include weights that you think should be changed.

    Your response should be in the following format:
    {{
        "weight_updates": {{
            "weight_name": new_value,
            ...
        }},
        "explanation": "Your explanation here"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
    )

    return json.loads(response.choices[0].message.content)

def main() -> None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # Ryder ran 10 iterations on 2024-06-20 to arrive at these weights
    current_weights = {
        "Dist": 0.1,
        "LenRat": 0.3,
        "WSDiff": 0.25,
        "WSKS": 0.15,
        "PunctJS": 0.05,
        "EntDiff": 0.2,
        "LexSim": 0.35,
        "CogProp": 0.3,
        "MorphComp": 0.15,
        "CogDist": 0.25
    }

    # Compute base metrics once
    print("Computing base metrics...")
    base_metrics = compute_base_metrics_parallel(languages)

    for iteration in range(5):  # Run 5 iterations
        print(f"\nIteration {iteration + 1}")
        results = run_comparison(base_metrics, current_weights)
        display_results(results)

        llm_suggestion = get_llm_suggestion(results, expected_outcomes, current_weights)
        print("\nLLM Suggestion:")
        print(json.dumps(llm_suggestion, indent=2))

        # Update weights
        current_weights.update(llm_suggestion['weight_updates'])
        print("\nUpdated weights:")
        print(json.dumps(current_weights, indent=2))

        # Stream results to output file
        with open(output_file, 'a') as f:
            f.write(json.dumps({
                "iteration": iteration + 1,
                "results": results,
                "llm_suggestion": llm_suggestion,
                "updated_weights": current_weights
            }, indent=2) + "\n")

if __name__ == "__main__":
    main()