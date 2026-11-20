"""
ScholarBridge Interactive Command-Line Interface (CLI)
======================================================
Allows developers, evaluators, and students to test profiles interactively
and inspect granular match results, scores, explanations, checklists,
and authoritative Phase 2 Evidence Packs with source citations.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scholarbridge_engine.models import StudentProfile
from scholarbridge_engine.evidence.engine_bridge import EngineEvidenceBridge


def run_demo():
    print("=" * 75)
    print("SCHOLARBRIDGE DETERMINISTIC DECISION & RAG EVIDENCE DEMO")
    print("Core Model: The Engine Decides. RAG Proves. The LLM Explains.")
    print("=" * 75)
    
    bridge = EngineEvidenceBridge()
    
    # Demo Profile: SC Girl in 1st Year Engineering (Gujarat Domicile)
    profile = StudentProfile(
        social_category="SC",
        gender="Female",
        annual_family_income=240000.0,
        domicile_state="Gujarat",
        study_state="Gujarat",
        education_level="Undergraduate",
        stream="Engineering",
        course_name="B.Tech Computer Engineering",
        marks_percentile=84.5,
        hostel_status="College_Affiliated_Hostel"
    )
    
    marks_display = f"{profile.marks_percentile}% Percentile" if profile.marks_percentile is not None else (f"{profile.marks_percentage}%" if profile.marks_percentage is not None else "Unstated")
    print(f"\n[STUDENT PROFILE]")
    print(f"  * Category:   {profile.social_category} | Gender: {profile.gender}")
    print(f"  * Income:     Rs. {profile.annual_family_income:,.0f} / year")
    print(f"  * Education:  {profile.education_level} ({profile.stream})")
    print(f"  * Marks:      {marks_display}")
    print(f"  * Hostel:     {profile.hostel_status}")
    print(f"  * Domicile:   {profile.domicile_state} (Studying in {profile.study_state})")
    
    enriched_out = bridge.recommend_with_evidence(profile, evidence_per_scheme=2)
    out = enriched_out.engine_output
    
    print("\n" + "=" * 75)
    print(f"EVALUATION SUMMARY ({out.total_scholarships_evaluated} Scholarships Evaluated)")
    print("=" * 75)
    print(f"  [PASS] Fully Eligible Matches:      {out.eligible_count}")
    print(f"  [COND] Conditionally Eligible:     {out.conditional_count}")
    print(f"  [EXCL] Ineligible / Excluded:       {out.ineligible_count}")
    
    print("\n" + "=" * 75)
    print("TOP RANKED RECOMMENDATIONS WITH AUTHORITATIVE EVIDENCE CITATIONS")
    print("=" * 75)
    
    for idx, enriched_item in enumerate(enriched_out.enriched_recommendations[:4], start=1):
        rec = enriched_item.item
        s = rec.scholarship
        m = rec.match_result
        print(f"\n#{rec.rank} {s.scholarship_name} [{s.short_name}]")
        print(f"   Provider: {s.provider_name} | Type: {s.scholarship_type}")
        print(f"   Scores:   Composite = {rec.final_score}/100  (Benefit: {rec.benefit_score}, Conf: {rec.confidence_score}, Urgency: {rec.urgency_score}, Specificity: {rec.specificity_score})")
        print(f"   Decision Rationale: {m.explanation}")
        
        print("   Official Evidence & Provenance:")
        for cit_summary in enriched_item.citations_summary:
            print(f"     -> {cit_summary}")
        
    print("\n" + "=" * 75)
    print(f"UNIFIED ACTIONABLE DOCUMENT CHECKLIST ({len(out.unified_checklist)} Deduplicated Documents)")
    print("=" * 75)
    
    mandatory_docs = [d for d in out.unified_checklist if d.is_mandatory]
    conditional_docs = [d for d in out.unified_checklist if not d.is_mandatory]
    
    print("\n[MANDATORY CORE DOCUMENTS]")
    for d in mandatory_docs:
        print(f"  [OK] {d.document_name} ({d.document_category}) -- Needed by: {', '.join(d.requesting_scholarships[:4])}")
        
    if conditional_docs:
        print("\n[CONDITIONAL / SPECIAL DOCUMENTS]")
        for d in conditional_docs:
            cond_str = f" | {d.conditional_notes[0]}" if d.conditional_notes else ""
            print(f"  [?]  {d.document_name} ({d.document_category}){cond_str} -- Needed by: {', '.join(d.requesting_scholarships[:3])}")
            
    print("\n" + "=" * 75)
    print("SAMPLE EXCLUSION REASONS (Deterministic Explainability)")
    print("=" * 75)
    sample_excluded = out.excluded_scholarships[:3]
    for ex in sample_excluded:
        print(f"  [X] {ex.scholarship.short_name}: {ex.explanation}")


if __name__ == "__main__":
    run_demo()
