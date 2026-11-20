"""
ScholarBridge Centralized Grounded Prompt Templates
===================================================
Defines immutable system prompts and dynamic grounded user prompt builders
grounded in deterministic engine outcomes and Phase 2 validated evidence packs.
"""

from typing import Optional, List, Dict, Any
from ..models import StudentProfile, MatchResult, Scholarship, EligibilityStatus
from ..evidence.models import EvidencePack

SYSTEM_PROMPT = """You are ScholarBridge's grounded explanation layer.
ScholarBridge is a scholarship intelligence platform for Gujarat students.

YOUR CORE MANDATE:
"The deterministic engine decides. RAG provides evidence. The LLM explains."

CRITICAL ARCHITECTURAL RULES & GUARDRAILS:
1. NEVER determine or override eligibility. The deterministic engine output provided in the prompt is 100% AUTHORITATIVE.
   - If the engine status is INELIGIBLE, you must explain why it is ineligible based on the failed rule(s); you cannot claim the student is eligible or relax any rule.
   - If the engine status is ELIGIBLE, explain how the criteria are satisfied.
   - If the engine status is CONDITIONAL, explain what specific information/condition is required.
2. NEVER override recommendation rankings or scores.
3. NEVER invent scholarship rules, income limits, academic cutoffs, deadlines, fees, benefits, or document requirements.
4. NEVER invent citation IDs. You may ONLY cite Evidence Citation IDs (e.g. 'EV-...') explicitly provided in the AUTHORITATIVE EVIDENCE section.
5. If evidence for a specific detail is missing or insufficient, state clearly that the detail could not be verified from official records rather than guessing.
6. Do NOT use external, unverified, or hallucinated knowledge. Ground all factual assertions in the supplied engine decisions and official evidence.

OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this schema:
{
  "answer": "Direct, empathetic, clear answer explaining the student's eligibility outcome for this scholarship based strictly on the authoritative engine decision.",
  "explanation_points": [
    "Point-by-point breakdown explaining how each profile attribute meets or fails official criteria, with reference to citation IDs where applicable."
  ],
  "reasoning_summary": "High-level synthesis explaining the deterministic decision grounded in official guidelines.",
  "warnings": [
    "Important warnings, exclusions, or mandatory verification requirements."
  ],
  "citations": [
    "EV-..."
  ]
}
"""


def build_eligibility_explanation_prompt(
    profile: StudentProfile,
    scholarship: Scholarship,
    match_result: MatchResult,
    evidence_pack: Optional[EvidencePack] = None,
    rank: Optional[int] = None,
    score: Optional[float] = None,
    user_query: Optional[str] = None
) -> str:
    """Constructs a fully grounded prompt containing student context, deterministic results, and evidence."""
    query_str = user_query or f"Why am I {match_result.status.value.lower()} for {scholarship.scholarship_name} ({scholarship.scheme_code})?"

    # 1. Student Profile Summary
    marks_str = (
        f"{profile.marks_percentile}% Percentile"
        if profile.marks_percentile is not None
        else (f"{profile.marks_percentage}%" if profile.marks_percentage is not None else "Not Specified")
    )
    profile_block = f"""[STUDENT PROFILE]
- Social Category: {profile.social_category}
- Gender: {profile.gender}
- Annual Family Income: ₹{profile.annual_family_income:,.0f}
- Domicile State: {profile.domicile_state} (Studying in {profile.study_state})
- Education Level: {profile.education_level} (Stream: {profile.stream})
- Enrolled Course: {profile.course_name or 'General'}
- Qualifying Marks: {marks_str}
- Is Diploma-to-Degree (D2D): {'Yes' if profile.is_d2d_admission else 'No'}
- Hostel Status: {profile.hostel_status or 'Not Specified'}
- Person with Disability (PwD): {'Yes (' + str(profile.disability_percentage) + '%)' if profile.is_pwd else 'No'}
- Special Demographics: Construction Worker Ward: {'Yes' if profile.is_construction_worker_ward else 'No'}, Defence Ward: {'Yes' if profile.is_defence_ward else 'No'}
"""

    # 2. Deterministic Engine Decision
    passed_rules_str = "\n".join([f"  * [PASSED] {r.rule.field}: {r.reason}" for r in match_result.passed_rules]) or "  * None"
    failed_rules_str = "\n".join([f"  * [FAILED] {r.rule.field}: {r.reason}" for r in match_result.failed_rules]) or "  * None"
    conditional_rules_str = "\n".join([f"  * [CONDITIONAL] {r.rule.field}: {r.reason}" for r in match_result.conditional_rules]) or "  * None"

    rank_str = f"Rank: #{rank}" if rank is not None else "Rank: Unranked (Ineligible/Conditional)"
    score_str = f"Recommendation Score: {score:.1f}/100" if score is not None else "Score: N/A"

    engine_block = f"""[AUTHORITATIVE DETERMINISTIC ENGINE EVALUATION]
- Scholarship ID: {scholarship.scholarship_id}
- Scheme Code: {scholarship.scheme_code}
- Scholarship Name: {scholarship.scholarship_name}
- Provider: {scholarship.provider_name} ({scholarship.government_level} Government)
- Official Portal: {scholarship.application_portal} ({scholarship.official_website})
- Authoritative Status: {match_result.status.value.upper()}
- {rank_str} | {score_str}
- Rule Evaluation Details:
{passed_rules_str}
{failed_rules_str}
{conditional_rules_str}
- Overall Explanation Summary: {match_result.explanation}
- Exclusion Reason (if any): {match_result.exclusion_reason or 'None'}
"""

    # 3. Authoritative Evidence Pack from Phase 2 RAG
    evidence_items = evidence_pack.evidence if evidence_pack else []
    if evidence_items:
        ev_lines = []
        for ev in evidence_items:
            cit = ev.citation
            ev_lines.append(
                f"--- EVIDENCE ITEM [{cit.citation_id}] ---\n"
                f"Document: {cit.document_title} (Academic Year: {cit.academic_year}, Version: {cit.version})\n"
                f"Section: {cit.section} > {cit.subsection} (Page {cit.page})\n"
                f"Official Source URL: {cit.source_url}\n"
                f"Content Hash: {cit.chunk_hash[:16]}...\n"
                f"Excerpt:\n{ev.text}\n"
            )
        evidence_block = "[AUTHORITATIVE OFFICIAL EVIDENCE]\n" + "\n".join(ev_lines)
    else:
        evidence_block = """[AUTHORITATIVE OFFICIAL EVIDENCE]
No official evidence retrieved for this scheme in current search context.
(You MUST note that official text evidence could not be retrieved, but preserve the engine's deterministic decision.)
"""

    # 4. User Question
    query_block = f"""[STUDENT QUESTION]
"{query_str}"

Please produce a grounded explanation in valid JSON adhering strictly to all architectural rules.
"""

    return f"{profile_block}\n{engine_block}\n{evidence_block}\n{query_block}"
