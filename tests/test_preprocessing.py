import unittest
from langsim.preprocessing import preprocess_samples

class TestPreprocessing(unittest.TestCase):
    def test_preprocess_samples(self):
        sample1 = ["  Hello world  ", "Test sentence"]
        sample2 = ["Hola mundo", "  Oración de prueba  "]
        processed1, processed2, rom1, rom2 = preprocess_samples(sample1, sample2)
        
        self.assertEqual(processed1, ["Hello world", "Test sentence"])
        self.assertEqual(processed2, ["Hola mundo", "Oración de prueba"])
        self.assertEqual(rom1, ["Hello world", "Test sentence"])
        self.assertEqual(rom2, ["Hola mundo", "Oracion de prueba"])
    
    def test_empty_samples(self):
        with self.assertRaises(ValueError):
            preprocess_samples([], [])
    
    def test_unequal_samples(self):
        with self.assertRaises(ValueError):
            preprocess_samples(["One"], ["One", "Two"])

if __name__ == '__main__':
    unittest.main()