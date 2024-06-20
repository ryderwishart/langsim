import unittest
from langsim.preprocessing import preprocess_samples

class TestPreprocessing(unittest.TestCase):
    def test_preprocess_samples_basic(self):
        sample1 = ["  Hello world  ", "Test sentence"]
        sample2 = ["Hola mundo", "  Oración de prueba  "]
        processed1, processed2, rom1, rom2 = preprocess_samples(sample1, sample2)
        
        self.assertEqual(processed1, ["Hello world", "Test sentence"])
        self.assertEqual(processed2, ["Hola mundo", "Oración de prueba"])
        self.assertEqual(rom1, ["Hello world", "Test sentence"])
        self.assertEqual(rom2, ["Hola mundo", "Oracion de prueba"])
    
    def test_preprocess_samples_non_latin(self):
        sample1 = ["こんにちは世界", "テスト文"]
        sample2 = ["Здравствуй, мир", "Тестовое предложение"]
        processed1, processed2, rom1, rom2 = preprocess_samples(sample1, sample2)
        
        self.assertEqual(processed1, ["こんにちは世界", "テスト文"])
        self.assertEqual(processed2, ["Здравствуй, мир", "Тестовое предложение"])
        self.assertEqual(rom1, ["kon'nichiha sekai", "tesuto bun"])
        self.assertEqual(rom2, ["Zdravstvuj, mir", "Testovoe predlozhenie"])
    
    def test_empty_samples(self):
        with self.assertRaises(ValueError):
            preprocess_samples([], [])
    
    def test_unequal_samples(self):
        with self.assertRaises(ValueError):
            preprocess_samples(["One"], ["One", "Two"])
    
    def test_none_samples(self):
        with self.assertRaises(ValueError):
            preprocess_samples(None, None)
    
    def test_mixed_type_samples(self):
        with self.assertRaises(ValueError):
            preprocess_samples(["One", 2], ["One", "Two"])

if __name__ == '__main__':
    unittest.main()