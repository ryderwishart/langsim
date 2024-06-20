from langsim import compare_languages, view_pairwise_scores, set_debug_mode

# Sample texts
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

print("Running comparison with debug mode enabled:")

scores_debug = compare_languages(sample1, sample2, debug=True)
view_pairwise_scores(sample1, sample2, scores_debug, debug=True)

