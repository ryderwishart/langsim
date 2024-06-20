import unittest
from io import StringIO
import sys
from langsim.visualization import view_pairwise_scores
from langsim.constants import METRIC_DICT

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.sample1 = ["Hello world", "This is a test"]
        self.sample2 = ["Hola mundo", "Esta es una prueba"]
        self.scores = {
            METRIC_DICT['Distortion']: 1.0,
            METRIC_DICT['Length ratio std']: 0.1,
            METRIC_DICT['Whitespace ratio diff']: 0.05,
            METRIC_DICT['Whitespace KS statistic']: 0.2,
            METRIC_DICT['Punctuation JS divergence']: 0.1,
            METRIC_DICT['Entropy diff']: 0.3,
            METRIC_DICT['Lexical similarity']: 0.5,
            METRIC_DICT['Cognate proportion']: 0.4,
            METRIC_DICT['Morphological complexity']: 0.6,
            METRIC_DICT['Cognate-based distortion']: 0.8
        }

    def test_view_pairwise_scores(self):
        captured_output = StringIO()
        sys.stdout = captured_output
        view_pairwise_scores(self.sample1, self.sample2, self.scores)
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Test overall scores table
        self.assertIn("Overall Scores", output)
        self.assertIn("Metric", output)
        self.assertIn("Score", output)
        for metric in self.scores.keys():
            self.assertIn(metric, output)

        # Test legend
        self.assertIn("Metric Legend", output)
        
        # Test pairwise line scores table
        self.assertIn("Pairwise Line Scores", output)
        self.assertIn("Line", output)
        
        # Test debug mode
        captured_output = StringIO()
        sys.stdout = captured_output
        view_pairwise_scores(self.sample1, self.sample2, self.scores, debug=True)
        sys.stdout = sys.__stdout__
        debug_output = captured_output.getvalue()
        self.assertIn("DEBUG:", debug_output)

if __name__ == '__main__':
    unittest.main()