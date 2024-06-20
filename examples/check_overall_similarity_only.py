from langsim import compare_languages, view_pairwise_scores

sample1 = [
    "This is a sample sentence.",
    "Another example sentence.",
    "A third sentence for comparison."
]

sample2 = [
    "This is a similar sentence.",
    "Another comparable sentence.",
    "A third sentence for contrast."
]


print("English to English - slightly different wordings")
scores = compare_languages(sample1, sample2, debug=True)

print(f"Overall similarity (should be high): {scores['OverallSim']}")

# Now two much more distant languages
sample3 = [
    "This is a sample sentence.",
    "Another example sentence.",
    "A third sentence for comparison."
]

sample4 = [
    "こんにちは、世界",
    "こんにちは、世界",
    "こんにちは、世界"
]

scores = compare_languages(sample3, sample4, debug=True)
print(f"Overall similarity (should be low): {scores['OverallSim']}")

