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

scores = compare_languages(sample1, sample2)
view_pairwise_scores(sample1, sample2, scores)