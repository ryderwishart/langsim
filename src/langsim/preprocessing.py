from typing import List, Tuple, Union
from uroman import uroman

def preprocess_samples(sample1: Union[List[str], str], sample2: Union[List[str], str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    if not isinstance(sample1, (list, str)) or not isinstance(sample2, (list, str)):
        raise TypeError("Samples must be either strings or lists of strings")
    
    # Convert single strings to lists
    sample1 = [sample1] if isinstance(sample1, str) else sample1
    sample2 = [sample2] if isinstance(sample2, str) else sample2
    
    if not sample1 or not sample2 or len(sample1) != len(sample2):
        raise ValueError("Sample lists must be non-empty and of equal length")
    
    if not all(isinstance(s, str) for s in sample1 + sample2):
        raise TypeError("All elements in samples must be strings")
    
    sample1 = [line.strip() for line in sample1]
    sample2 = [line.strip() for line in sample2]
    rom_sample1 = [uroman(line) for line in sample1]
    rom_sample2 = [uroman(line) for line in sample2]
    
    return sample1, sample2, rom_sample1, rom_sample2