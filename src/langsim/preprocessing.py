from typing import List, Tuple
from uroman import uroman

def preprocess_samples(sample1: List[str], sample2: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    if not sample1 or not sample2 or len(sample1) != len(sample2):
        raise ValueError("Sample lists must be non-empty and of equal length")
    
    sample1 = [line.strip() for line in sample1]
    sample2 = [line.strip() for line in sample2]
    rom_sample1 = [uroman(line) for line in sample1]
    rom_sample2 = [uroman(line) for line in sample2]
    
    return sample1, sample2, rom_sample1, rom_sample2