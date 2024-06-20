import unittest
import numpy as np
from langsim.metrics import (
    calculate_distortion,
    calculate_length_ratio_similarity,
    calculate_whitespace_metrics,
    calculate_punctuation_distribution,
    calculate_entropy_diff,
    calculate_lexical_similarity,
    calculate_cognate_metrics,
    calculate_morphological_complexity,
    compare_languages
)
from langsim.constants import METRIC_DICT

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.sample1 = ["Hello world", "This is a test"]
        self.sample2 = ["Hola mundo", "Esta es una prueba"]
        self.rom_sample1 = ["Hello world", "This is a test"]
        self.rom_sample2 = ["Hola mundo", "Esta es una prueba"]

    def test_calculate_distortion(self):
        distortion = calculate_distortion(self.sample1, self.sample2)
        self.assertIsInstance(distortion, float)
        self.assertGreaterEqual(distortion, -1)
        self.assertLessEqual(distortion, 1)

    def test_calculate_length_ratio_similarity(self):
        similarity = calculate_length_ratio_similarity(self.sample1, self.sample2)
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0)

    def test_calculate_whitespace_metrics(self):
        ws_ratio_diff, ws_ks_stat = calculate_whitespace_metrics(self.sample1, self.sample2)
        self.assertIsInstance(ws_ratio_diff, float)
        self.assertIsInstance(ws_ks_stat, float)
        self.assertGreaterEqual(ws_ratio_diff, 0)
        self.assertGreaterEqual(ws_ks_stat, 0)
        self.assertLessEqual(ws_ks_stat, 1)

    def test_calculate_punctuation_distribution(self):
        punct_js_div = calculate_punctuation_distribution(self.sample1, self.sample2)
        self.assertIsInstance(punct_js_div, float)
        self.assertGreaterEqual(punct_js_div, 0)
        self.assertLessEqual(punct_js_div, 1)

    def test_calculate_entropy_diff(self):
        entropy_diff = calculate_entropy_diff(self.sample1, self.sample2)
        self.assertIsInstance(entropy_diff, float)
        self.assertGreaterEqual(entropy_diff, 0)

    def test_calculate_lexical_similarity(self):
        lexical_sim = calculate_lexical_similarity(self.sample1, self.sample2)
        self.assertIsInstance(lexical_sim, float)
        self.assertGreaterEqual(lexical_sim, 0)
        self.assertLessEqual(lexical_sim, 1)

    def test_calculate_cognate_metrics(self):
        cognate_prop, cognate_distortion = calculate_cognate_metrics(self.rom_sample1, self.rom_sample2)
        self.assertIsInstance(cognate_prop, float)
        self.assertIsInstance(cognate_distortion, float)
        self.assertGreaterEqual(cognate_prop, 0)
        self.assertLessEqual(cognate_prop, 1)
        self.assertGreaterEqual(cognate_distortion, -1)
        self.assertLessEqual(cognate_distortion, 1)

    def test_calculate_morphological_complexity(self):
        morph_complexity = calculate_morphological_complexity(self.rom_sample1, self.rom_sample2)
        self.assertIsInstance(morph_complexity, float)
        self.assertGreaterEqual(morph_complexity, 0)
        self.assertLessEqual(morph_complexity, 1)

    def test_compare_languages(self):
        results = compare_languages(self.sample1, self.sample2)
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 10)  # Check if all metrics are present
        
        for metric, value in results.items():
            self.assertIn(metric, METRIC_DICT.values())
            self.assertIsInstance(value, float)
            
        # Test with debug=True
        results_debug = compare_languages(self.sample1, self.sample2, debug=True)
        self.assertEqual(results, results_debug)
        
        # Test with string input
        results_string = compare_languages("Hello world", "Hola mundo")
        self.assertIsInstance(results_string, dict)
        self.assertEqual(len(results_string), 10)

if __name__ == '__main__':
    unittest.main()