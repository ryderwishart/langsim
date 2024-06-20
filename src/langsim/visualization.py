from typing import List, Dict
from .metrics import compare_languages
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import numpy as np
from .constants import METRIC_NAMES, METRIC_DICT, METRIC_ABBR_DICT
from .config import DEBUG_MODE

console = Console()

def view_pairwise_scores(sample1: List[str], sample2: List[str], scores: Dict[str, float], debug: bool = None) -> None:
    # Use the global DEBUG_MODE if debug is not explicitly set
    debug = DEBUG_MODE if debug is None else debug

    if debug:
        print("DEBUG: Entered view_pairwise_scores function")
        print(f"DEBUG: sample1 = {sample1}")
        print(f"DEBUG: sample2 = {sample2}")
        print(f"DEBUG: scores = {scores}")

    # Overall scores table
    overall_table = Table(title="Overall Scores", show_header=True, header_style="bold cyan")
    overall_table.add_column("Metric", style="dim", width=30)
    overall_table.add_column("Score", justify="right")
    
    for metric, score in scores.items():
        overall_table.add_row(metric, f"{score:.3f}")
    
    console.print(overall_table)
    
    # Legend
    legend = Table.grid(padding=1)
    legend.add_column(style="cyan", justify="right")
    legend.add_column(style="magenta")
    for full_name, abbr in METRIC_NAMES:
        legend.add_row(abbr + ":", full_name)
    
    console.print(Panel(legend, title="Metric Legend", border_style="bright_blue"))
    
    # Pairwise line scores table
    pairwise_table = Table(title="Pairwise Line Scores", show_header=True, header_style="bold cyan")
    headers = ['Line'] + [abbr for _, abbr in METRIC_NAMES if abbr != 'Line']
    
    for header in headers:
        pairwise_table.add_column(header, justify="right", width=10)
    
    for i, (line1, line2) in enumerate(zip(sample1, sample2), start=1):
        if debug:
            print(f"DEBUG: Comparing line {i}")
            print(f"DEBUG: line1 = {line1}")
            print(f"DEBUG: line2 = {line2}")
        line_scores = compare_languages([line1], [line2], debug=debug)
        if debug:
            print(f"DEBUG: line_scores = {line_scores}")
        score_values = [i] + [line_scores.get(metric, np.nan) for metric in headers[1:]]
        score_values = [f"{v:.3f}" if isinstance(v, (float, np.float64)) else str(v) for v in score_values]
        pairwise_table.add_row(*score_values)
    
    console.print(pairwise_table)
