"""
ScholarBridge Engine Configuration
==================================
Defines data paths, configurable recommendation scoring weights,
benefit normalization thresholds, and application parameters.
"""

import os
from pathlib import Path

# Base package directory
PACKAGE_DIR = Path(__file__).resolve().parent

# Bundled data directory inside the package
BUNDLED_DATA_DIR = PACKAGE_DIR / "data"

def get_kb_dir() -> Path:
    """
    Resolves knowledge base directory in priority order:
    1. SCHOLARBRIDGE_KB_DIR environment variable
    2. Bundled package data directory (scholarbridge_engine/data)
    3. Workspace / data directory relative to repository root
    """
    # 1. Environment Variable
    env_path = os.environ.get("SCHOLARBRIDGE_KB_DIR")
    if env_path:
        p = Path(env_path).resolve()
        if p.exists() and (p / "Gujarat_Scholarships_Master.csv").exists():
            return p

    # 2. Bundled Package Data
    if BUNDLED_DATA_DIR.exists() and (BUNDLED_DATA_DIR / "Gujarat_Scholarships_Master.csv").exists():
        return BUNDLED_DATA_DIR

    # 3. Relative workspace / parent data fallbacks
    for candidate in [PACKAGE_DIR.parent / "data", Path.cwd() / "data", Path.cwd() / "scholarbridge_engine" / "data"]:
        if candidate.exists() and (candidate / "Gujarat_Scholarships_Master.csv").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "ScholarBridge Knowledge Base could not be located. "
        "Ensure CSV files exist in 'scholarbridge_engine/data/' or set the "
        "'SCHOLARBRIDGE_KB_DIR' environment variable to the directory containing "
        "Gujarat_Scholarships_Master.csv and related files."
    )

# Multi-Factor Recommendation Weights
# The weights are configurable and represent the initial project-defined prioritization strategy.
RECOMMENDATION_WEIGHTS = {
    "benefit": 0.40,       # Financial impact & tuition coverage
    "confidence": 0.25,    # Verification tier and source reliability
    "urgency": 0.20,       # Application window status & timeline urgency
    "specificity": 0.15    # Targeted scheme bonus (e.g. SC Girl in STEM vs generic grant)
}

# Hybrid Scoring Weights (Section 11)
# Combines the existing rule/relevance score with the Section 9 Random Forest
# confidence signal and the Section 10 NLP semantic similarity signal into a
# single 0-100 hybrid_score. Configurable so weight tuning never requires
# touching the scoring logic itself. Eligibility status (ELIGIBLE / CONDITIONAL /
# INELIGIBLE / SUSPENDED) is decided exclusively by RuleMatcher and is NEVER
# influenced by these weights or by rf_confidence_score / nlp_similarity_score.
HYBRID_SCORING_WEIGHTS = {
    "rule_relevance": 0.60,   # Existing rule/relevance score (RecommendationScorer.final_score)
    "rf_confidence": 0.20,    # Section 9 rf_confidence_score
    "nlp_similarity": 0.20,   # Section 10 nlp_similarity_score
}

# Benefit Normalization Parameters
# Normalizes diverse benefit types (grants, tuition caps, stipends) to a 0–100 scale
BENEFIT_NORMALIZATION = {
    "max_benchmark_amount": 250000.0,    # Any benefit >= Rs. 2.5L gets maximum benefit score of 100
    "monthly_to_annual_months": 10,       # Standard academic year month multiplier for stipends
    "full_tuition_fallback_cap": 80000.0, # Estimated average annual fee if tuition fee is unspecified
    "default_score_if_unspecified": 30.0  # Safe fallback for intangible benefits
}

# Note: Source-tier to base confidence weighting is deferred; raw verified confidence_score from knowledge base is used directly.

# Deadline Status Score Mapping
DEADLINE_URGENCY_MAP = {
    "ANNOUNCED_CYCLE": 90.0,
    "OPEN": 100.0,
    "CLOSING_SOON": 100.0,
    "YEAR_ROUND": 75.0,
    "NOT_ANNOUNCED": 50.0,
    "SUSPENDED": 0.0,
    "CLOSED": 10.0
}
