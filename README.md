# langsim

Compare languages even with just a few translation pairs.

> NOTE: This project is very much a work in progress. It is not recommended to use it in a production environment. If you do use it, please open an issue and leave your impressions. I've only tested this on an AARCH Macbook Pro.

## How to install from source

Clone the repository:

```bash
git clone https://github.com/ryderwishart/langsim.git
```

Run `pip install -r requirements-dev.txt` to install the dependencies.

Run `pip install -e .` to install the package in editable mode.

## How to run

Run `python examples/basic_usage.py` to run the basic usage example.

Run `python examples/using_debug_mode.py` to run the basic usage example with debug mode.

## Tests

Run `pytest` to test the code.

> Note: tests are currently failing apparently due to a mismatch with the `hydra-core` version.

## Example output

```zsh        Overall Scores              
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric                         ┃ Score ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Dist                           │ 1.000 │
│ LenRat                         │ 0.076 │
│ WSDiff                         │ 0.003 │
│ WSKS                           │ 0.003 │
│ PunctJS                        │ 0.000 │
│ EntDiff                        │ 0.013 │
│ LexSim                         │ 0.600 │
│ CogProp                        │ 0.769 │
│ MorphComp                      │ 0.474 │
│ CogDist                        │ 1.000 │
│ OverallSim                     │ 0.628 │
└────────────────────────────────┴───────┘
╭─────────────────────────────────────────────── Metric Legend ────────────────────────────────────────────────╮
│       Line: Line                                                                                             │
│                                                                                                              │
│       Dist: Distortion                                                                                       │
│                                                                                                              │
│     LenRat: Length ratio std                                                                                 │
│                                                                                                              │
│     WSDiff: Whitespace ratio diff                                                                            │
│                                                                                                              │
│       WSKS: Whitespace KS statistic                                                                          │
│                                                                                                              │
│    PunctJS: Punctuation JS divergence                                                                        │
│                                                                                                              │
│    EntDiff: Entropy diff                                                                                     │
│                                                                                                              │
│     LexSim: Lexical similarity                                                                               │
│                                                                                                              │
│    CogProp: Cognate proportion                                                                               │
│                                                                                                              │
│    CogDist: Cognate-based distortion                                                                         │
│                                                                                                              │
│  MorphComp: Morphological complexity                                                                         │
│                                                                                                              │
│ OverallSim: Overall Similarity                                                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                              Pairwise Line Scores                                              
┏━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃   Line ┃   Dist ┃ LenRat ┃ WSDiff ┃   WSKS ┃ Punct… ┃ EntDi… ┃  LexSim ┃ CogPr… ┃ CogDist ┃ Morph… ┃ Overal… ┃
┡━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│      1 │  1.000 │  0.037 │  0.006 │  0.006 │  0.000 │  0.005 │   0.667 │  0.800 │   1.000 │  0.560 │   0.656 │
│      2 │  1.000 │  0.107 │  0.009 │  0.009 │  0.000 │  0.121 │   0.500 │  0.667 │   1.000 │  0.551 │   0.589 │
│      3 │  1.000 │  0.067 │  0.008 │  0.008 │  0.000 │  0.208 │   0.667 │  0.800 │   1.000 │  0.442 │   0.626 │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴─────────┴────────┴─────────┴────────┴─────────┘
```

## Metrics Explanation

Before listing the comparison metrics, a few caveats are necessary. 
The comparisons employed in this library are intended to serve as a starting point. Please open issues or PRs with any suggested revisions. I have attempted to be language-agnostic, so you could really compare *any* two sets of strings. Accordingly, I have avoided comparisons that rely on structured knowledge. Tokenization is completely naive, for example, and syntax is not considered.

I would like to add phonological similarity metrics in the future, and am waiting for code to be released from the [Greek Room library](https://greekroom.org).

The following metrics are used in `metrics.py` to compare language samples. Each metric provides a different perspective on the similarity or difference between the samples:

1. **Distortion (Dist)**: Measures the alignment of lines between two samples. A value of 1.0 indicates perfect alignment, while lower values indicate greater distortion.

2. **Length Ratio Standard Deviation (LenRat)**: Calculates the standard deviation of the length ratios of corresponding lines in the samples. A lower value indicates more consistent line lengths between the samples.

3. **Whitespace Ratio Difference (WSDiff)**: Compares the proportion of whitespace characters in the samples. A lower value indicates more similar whitespace usage.

4. **Whitespace KS Statistic (WSKS)**: Uses the Kolmogorov-Smirnov statistic to compare the distribution of whitespace characters in the samples. A lower value indicates more similar distributions.

5. **Punctuation Jensen-Shannon Divergence (PunctJS)**: Measures the divergence in punctuation usage between the samples using the Jensen-Shannon divergence. A lower value indicates more similar punctuation distributions.

6. **Entropy Difference (EntDiff)**: Compares the entropy (a measure of randomness) of the character distributions in the samples. A lower value indicates more similar entropy values.

7. **Lexical Similarity (LexSim)**: Measures the proportion of shared words between the samples. A higher value indicates greater lexical similarity.

8. **Cognate Proportion (CogProp)**: Calculates the proportion of cognates (words with a common etymological origin) between the samples. A higher value indicates a higher proportion of cognates.

9. **Cognate-based Distortion (CogDist)**: Measures the alignment of cognates between the samples. A value of 1.0 indicates perfect alignment, while lower values indicate greater distortion.

10. **Morphological Complexity (MorphComp)**: Compares the complexity of word forms in the samples by analyzing the distribution of word tokens. A higher value indicates more similar morphological complexity.

11. **Overall Similarity (OverallSim)**: A weighted combination of the above metrics to provide an overall similarity score between the samples. Higher values indicate greater overall similarity.
