# src/ai_detect.py
"""
Kitty's upgraded heuristic AI detector: now looks for em-dashes.
Returns a score between 0.0 and 1.0 where higher ≈ more likely AI-generated.

Notes:
- Detects the Unicode em-dash '—' and the ASCII double-hyphen '--' (both treated as dash tokens).
- This is a heuristic for demo/MVP use only.
"""

import math
import re
from typing import Dict

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def heuristic_score(text: str) -> float:
    if not text or not text.strip():
        return 0.5

    # normalize
    t = str(text).strip()

    # tokens and characters
    words = [w for w in re.findall(r"[A-Za-z']+", t)]
    n_words = max(1, len(words))

    # basic features
    avg_word_len = sum(len(w) for w in words) / n_words
    unique_ratio = len(set(w.lower() for w in words)) / n_words
    punct_count = sum(1 for ch in t if ch in ".,!?;:")
    punct_per_word = punct_count / n_words

    # Kitty's em-dash condition
    # Count Unicode em-dash '—' and ASCII double-hyphen '--'
    emdash_count = t.count("—") + t.count("--")
    emdash_per_word = emdash_count / n_words

    # heuristics (tunable)
    f_long_word = _sigmoid((avg_word_len - 5.5) * 0.9)        # longer words -> slightly more AI-ish
    f_low_div = _sigmoid((0.6 - unique_ratio) * 3.5)         # low lexical diversity -> AI-ish
    f_low_punct = _sigmoid((0.08 - punct_per_word) * 10.0)   # unusually low punctuation -> AI-ish
    f_many_emdash = _sigmoid((emdash_per_word - 0.02) * 40.0) # many em-dashes -> AI-ish (Kitty's rule)

    # combine weights (adjustable)
    score = (
        0.34 * f_low_div +
        0.26 * f_long_word +
        0.20 * f_low_punct +
        0.20 * f_many_emdash
    )

    return float(max(0.0, min(1.0, score)))

def detect(text: str) -> Dict:
    """Return a dict with a score and short note."""
    s = heuristic_score(text)
    return {
        "score": s,
        "method": "heuristic",
        "note": "Kitty's em-dash enhanced heuristic detector 🐾"
    }

