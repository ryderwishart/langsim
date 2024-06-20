import unittest
from io import StringIO
import sys
from langsim.visualization import view_pairwise_scores

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.sample1 = ["Hello world", "This is a test"]
        self.sample2 = ["Hola mundo", "Esta es una prueba"]
        self.scores = {
            'Distortion': 1.0,
            'Length ratio std': 0.1,
            'Whitespace ratio diff': 0.05,
            'Whitespace KS statistic': 0.2,
            'Punctuation JS divergence': 0.1,
            'Entropy diff': 0.3,
            'Lexical similarity': 0.5,
            'Cognate proportion': 0.4,
            'Morphological complexity': 0.6,
            'Cognate-based distortion': 0.8
        }

    def test_view_pairwise_scores(self):
        captured_output = StringIO()
        sys.stdout = captured_output
        view_pairwise_scores(self.sample1, self.sample2, self.scores)
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        self.assertIn("Metric", output)
        self.assertIn("Score", output)
        self.assertIn("Pairwise line scores", output)

if __name__ == '__main__':
    unittest.main()