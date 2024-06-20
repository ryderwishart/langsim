print("Comparing major languages. This may take a few minutes...")

from langsim import compare_languages
from rich.console import Console
from rich.table import Table
from itertools import combinations

# Define sentences in various languages
languages = {
    "English": "The quick brown fox jumps over the lazy dog.",
    "Spanish": "El rápido zorro marrón salta sobre el perro perezoso.",
    "Mandarin": "快速的棕色狐狸跳过懒惰的狗。",
    "Arabic": "الثعلب البني السريع يقفز فوق الكلب الكسول.",
    "Swahili": "Mbweha mwepesi wa kahawia anaruka juu ya mbwa mvivu.",
    "Hindi": "तेज़ भूरी लोमड़ी आलसी कुत्ते पर कूदती है।",
    "Russian": "Быстрая коричневая лиса прыгает через ленивую собаку.",
    "Japanese": "素早い茶色のキツネは怠けた犬を飛び越えます。",
    "French": "Le rapide renard brun saute par-dessus le chien paresseux.",
    "German": "Der schnelle braune Fuchs springt über den faulen Hund."
}

# Create a table
table = Table(title="Language Similarity Scores")
table.add_column("Language Pair", style="cyan", no_wrap=True)
table.add_column("Overall Similarity", style="magenta")

# Compare all language pairs and store results
comparison_results = []
for (lang1, sent1), (lang2, sent2) in combinations(languages.items(), 2):
    scores = compare_languages([sent1], [sent2])
    overall_sim = scores['OverallSim']
    comparison_results.append((f"{lang1} - {lang2}", f"{overall_sim:.4f}"))

# Sort the comparison results by overall similarity score (descending order)
comparison_results.sort(key=lambda x: float(x[1]), reverse=True)

# Add sorted results to the table
for lang_pair, similarity in comparison_results:
    table.add_row(lang_pair, similarity)

# Display the table
console = Console()
console.print(table)