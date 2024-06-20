import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langsim import compare_languages
from rich.console import Console
from rich.table import Table
from itertools import combinations
import json
from openai import OpenAI

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
languages = {
    "English": "The quick brown fox jumps over the lazy dog.",
    "Spanish": "El rápido zorro marrón salta sobre el perro perezoso.",
    "Mandarin": "快速的棕色狐狸跳过懒惰的狗。",
    "Arabic": "الثعلب البني السريع يقفز فوق الكلب الكسول.",
    "Swahili": "Mbweha mwepesi wa kahawia anaruka juu ya mbwa mvivu.",
    "Hindi": "तेज़ भूरी ल��मड़ी आलसी कुत्ते पर कूदती है।",
    "Russian": "Быстрая коричневая лиса прыгает через ленивую собаку.",
    "Japanese": "素早い茶色のキツネは怠けた犬を飛び越えます。",
    "French": "Le rapide renard brun saute par-dessus le chien paresseux.",
    "German": "Der schnelle braune Fuchs springt über den faulen Hund.",
    "RandomNoise1": "asdkjfhaskjdfh",
    "RandomNoise2": "qwerqwerqwer",
    "EnglishUK": "The quick brown fox jumps over the lazy dog.",
    "EnglishUS": "The quick brown fox jumps over the lazy dog."
}

# Define expected outcomes (you can adjust these based on linguistic knowledge)
expected_outcomes = {
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
    "EnglishUK-EnglishUS": 0.95
}

def compute_base_metrics(languages):
    base_metrics = {}
    for (lang1, sent1), (lang2, sent2) in combinations(languages.items(), 2):
        scores = compare_languages([sent1], [sent2], custom_weights=None)
        # Store all metrics except OverallSim
        base_metrics[f"{lang1}-{lang2}"] = {k: v for k, v in scores.items() if k != 'OverallSim'}
    return base_metrics

def apply_weights(base_metrics, custom_weights):
    results = {}
    for lang_pair, metrics in base_metrics.items():
        overall_sim = sum(custom_weights[metric] * value for metric, value in metrics.items())
        results[lang_pair] = overall_sim
    return results

def run_comparison(base_metrics, custom_weights=None):
    if custom_weights is None:
        custom_weights = {
            'Dist': 0.15, 'LenRat': 0.05, 'WSDiff': 0.05, 'WSKS': 0.05,
            'PunctJS': 0.1, 'EntDiff': 0.1, 'LexSim': 0.2,
            'CogProp': 0.15, 'MorphComp': 0.1, 'CogDist': 0.05
        }
    return apply_weights(base_metrics, custom_weights)

def display_results(results):
    table = Table(title="Language Similarity Scores")
    table.add_column("Language Pair", style="cyan", no_wrap=True)
    table.add_column("Overall Similarity", style="magenta")
    for lang_pair, similarity in sorted(results.items(), key=lambda x: x[1], reverse=True):
        table.add_row(lang_pair, f"{similarity:.4f}")
    console = Console()
    console.print(table)

def get_llm_suggestion(results, expected_outcomes, current_weights):
    prompt = f"""
    You are an expert linguist and programmer. Analyze the following language comparison results and suggest updates to the weights used in the comparison algorithm.

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

def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # Ryder ran 10 iterations on 2024-06-20 to arrive at these weights
    current_weights = {
        "Dist": 0.1,
        "LenRat": 0.12,
        "WSDiff": 0.3,
        "WSKS": 0.2,
        "PunctJS": 0.1,
        "EntDiff": 0.22,
        "LexSim": 0.18,
        "CogProp": 0.25,
        "MorphComp": 0.2,
        "CogDist": 0.2
    }

    # Compute base metrics once
    print("Computing base metrics...")
    base_metrics = compute_base_metrics(languages)

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