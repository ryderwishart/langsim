import unittest
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

    # Add similar tests for other metric functions...

    def test_compare_languages(self):
        results = compare_languages(self.sample1, self.sample2)
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 10)  # Check if all metrics are present

if __name__ == '__main__':
    unittest.main()