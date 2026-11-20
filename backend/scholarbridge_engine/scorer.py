"""
ScholarBridge Recommendation Scorer (Heuristic Prioritization Policy)
=====================================================================
Computes multi-factor recommendation scores across Benefit Value,
Verification Confidence, Deadline Urgency, and Student-Specific Specificity.

Note: The recommendation weights and formula represent the project's
configurable recommendation policy / heuristic prioritization strategy,
ensuring explainable and deterministic ranking above the eligibility engine.
"""

from datetime import date
from typing import List, Optional, Dict, Any
from .config import (
    RECOMMENDATION_WEIGHTS, BENEFIT_NORMALIZATION,
    DEADLINE_URGENCY_MAP
)
from .models import (
    Scholarship, MatchResult, RecommendationItem, StudentProfile
)
from .deadline_tracker import DeadlineTracker


class RecommendationScorer:
    """Calculates granular component scores and ranks eligible scholarships."""

    @staticmethod
    def calculate_benefit_score(scholarship: Scholarship) -> float:
        """Normalizes diverse benefit types to a standardized 0–100 scale."""
        max_bench = BENEFIT_NORMALIZATION["max_benchmark_amount"]
        raw_max = scholarship.max_benefit_amount_inr
        
        if raw_max > 0:
            # Scaled linearly up to the benchmark amount of Rs. 2.50L
            score = min(100.0, (raw_max / max_bench) * 100.0)
            return max(15.0, score)
        
        # If max benefit is unspecified / full fee waiver fallback
        tuition_cov = scholarship.tuition_fee_coverage or ""
        benefit_t = scholarship.benefit_type or ""
        
        if "Full" in tuition_cov or "100%" in tuition_cov or "Waiver" in benefit_t or "Full" in benefit_t:
            fallback = BENEFIT_NORMALIZATION["full_tuition_fallback_cap"]
            return (fallback / max_bench) * 100.0
            
        return BENEFIT_NORMALIZATION["default_score_if_unspecified"]

    @staticmethod
    def calculate_confidence_score(scholarship: Scholarship) -> float:
        """Extracts source verification confidence score from verified knowledge base metadata."""
        raw_conf = scholarship.confidence_score
        if 0.0 < raw_conf <= 1.0:
            return raw_conf * 100.0
        return 75.0

    @staticmethod
    def calculate_urgency_score(
        scholarship: Scholarship,
        timeline_info: Optional[Dict[str, Any]] = None,
        ref_date: Optional[date] = None
    ) -> float:
        """Evaluates deadline timeline urgency, accounting for live open/closed staleness."""
        t_info = timeline_info if timeline_info is not None else DeadlineTracker.get_timeline_info(scholarship, ref_date=ref_date)
        timeline_disp = t_info.get("timeline_display", "") if t_info else ""
        d_status = scholarship.deadline_status.upper() if scholarship.deadline_status else "NOT_ANNOUNCED"

        # If live timeline shows application window is closed (and not year-round / suspended)
        if timeline_disp.startswith("Closed on") and d_status not in ["YEAR_ROUND", "SUSPENDED"]:
            return 5.0

        return DEADLINE_URGENCY_MAP.get(d_status, 50.0)

    @staticmethod
    def calculate_specificity_score(scholarship: Scholarship, profile: StudentProfile) -> float:
        """
        Calculates student-specific specificity bonus.
        Rewards scholarships specifically targeted to THIS student's unique
        demographic, financial, institutional, and academic profile.
        """
        score = 30.0 # Base general score
        
        # 1. Gender-specific alignment
        if scholarship.gender in ["Female", "Male"]:
            if profile.gender.lower() == scholarship.gender.lower():
                score += 15.0
                
        # 2. Social Category-specific alignment
        s_cats = [c.strip().upper() for c in scholarship.eligible_categories.replace(", ", "|").split("|")]
        if "ALL" not in s_cats and profile.social_category.upper() in s_cats:
            score += 15.0
            
        # 3. Special Context alignment (Hostel, PwD, Construction, Defence, Sports)
        spec_group = scholarship.special_beneficiary_group.upper()
        
        if ("HOSTEL" in spec_group or "FOOD" in scholarship.scholarship_name.upper()) and profile.hostel_status in ["Hosteller", "College_Affiliated_Hostel"]:
            score += 15.0
            
        if ("DISABILITY" in spec_group or "PWD" in spec_group or scholarship.disability_eligible == "Yes") and profile.is_pwd:
            score += 20.0
            
        if ("CONSTRUCTION" in spec_group or "SHRAMIK" in scholarship.scholarship_name.upper()) and (profile.is_construction_worker_ward or (profile.parent_occupation and "CONSTRUCTION" in profile.parent_occupation.upper())):
            score += 20.0
            
        if ("DEFENCE" in spec_group or "PMSS" in scholarship.scheme_code) and (profile.is_defence_ward or (profile.parent_occupation and any(o in profile.parent_occupation.upper() for o in ["DEFENCE", "EX-SERVICEMEN", "CAPF"]))):
            score += 20.0
            
        if ("SPORTS" in spec_group or "KHEL" in scholarship.scholarship_name.upper()) and profile.is_sports_medalist:
            score += 20.0
            
        # 4. Stream / Professional Course alignment
        s_streams = [s.strip().upper() for s in scholarship.stream.replace(", ", "|").replace("/", "|").split("|")]
        if "ALL" not in s_streams and profile.stream.upper() in s_streams:
            score += 10.0
            
        return min(100.0, score)

    def score_and_rank(
        self, matches: List[MatchResult], profile: StudentProfile, weights: dict = None
    ) -> List[RecommendationItem]:
        """Scores all matched scholarships and returns sorted recommendation items."""
        w = weights or RECOMMENDATION_WEIGHTS
        w_ben = w.get("benefit", 0.40)
        w_conf = w.get("confidence", 0.25)
        w_urg = w.get("urgency", 0.20)
        w_spec = w.get("specificity", 0.15)
        
        items: List[RecommendationItem] = []
        
        for m in matches:
            s = m.scholarship
            b_score = self.calculate_benefit_score(s)
            c_score = self.calculate_confidence_score(s)
            u_score = self.calculate_urgency_score(s, timeline_info=m.timeline_info)
            s_score = self.calculate_specificity_score(s, profile)
            
            final_composite = (
                w_ben * b_score +
                w_conf * c_score +
                w_urg * u_score +
                w_spec * s_score
            )
            
            items.append(RecommendationItem(
                scholarship=s,
                match_result=m,
                benefit_score=round(b_score, 1),
                confidence_score=round(c_score, 1),
                urgency_score=round(u_score, 1),
                specificity_score=round(s_score, 1),
                final_score=round(final_composite, 1),
                timeline_info=m.timeline_info,
                explanation_details=m.explanation_details
            ))
            
        # Sort by final composite score descending
        items.sort(key=lambda x: x.final_score, reverse=True)
        
        # Assign 1-indexed ranks
        for idx, item in enumerate(items, start=1):
            item.rank = idx
            
        return items
