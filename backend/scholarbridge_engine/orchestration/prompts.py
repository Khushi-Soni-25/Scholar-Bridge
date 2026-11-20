"""
ScholarBridge Orchestration Prompt Templates
============================================
Explicit prompt templates ensuring the LLM is strictly constrained:
  1. Engine Decides (LLM cannot override eligibility status or rank)
  2. RAG Proves (LLM must only use supplied evidence)
  3. Strict Citation Grounding (LLM must cite only retrieved [EV-...] IDs)
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# System prompt for grounded eligibility explanation
EXPLANATION_SYSTEM_PROMPT = """You are the official ScholarBridge Explainability Assistant for Gujarat Scholarships.
Your role is to explain the scholarship recommendation result provided to you.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. YOU DO NOT DECIDE ELIGIBILITY. The eligibility status has ALREADY been calculated deterministically by the rule engine.
2. YOU CANNOT CHANGE ELIGIBILITY STATUS. If the status is INELIGIBLE, you must explain why it failed. If ELIGIBLE, explain which rules passed.
3. GROUNDING ONLY: Use ONLY the supplied eligibility criteria and official evidence excerpts. Do NOT invent new scholarship requirements.
4. CITATIONS: When stating facts from official documents, reference the exact Citation ID in square brackets, e.g. [EV-MYSY-01]. Do NOT fabricate citation IDs.
5. FORMAT: You MUST return a single JSON object conforming strictly to the requested schema. Do not output markdown code blocks or text outside the JSON.
"""

EXPLANATION_USER_TEMPLATE = """Explain the following deterministic scholarship evaluation:

==================== APPLICANT CONTEXT ====================
- Education Level: {education_level}
- Stream / Course: {stream} ({course_name})
- Social Category: {social_category}
- Gender: {gender}
- Annual Family Income: Rs. {annual_family_income}
- Domicile / Study State: {domicile_state} / {study_state}
- Marks / Percentile: {marks_percentile}%
- PwD Status: {is_pwd} (Percentage: {disability_percentage}%)
- Hostel Status: {hostel_status}

==================== DETERMINISTIC ENGINE RESULT ====================
- Scholarship Name: {scholarship_name} ({scholarship_id})
- Authoritative Status: {authoritative_status}
- Match Confidence: {match_confidence}
- Composite Score: {composite_score}
- Evaluated Rules:
{evaluated_rules_text}

==================== OFFICIAL RETRIEVED EVIDENCE ====================
{evidence_text}

User Query: {user_query}

Provide a grounded explanation strictly based on the above deterministic rules and official evidence.
"""

# Prompt for comparing multiple scholarships deterministically
COMPARISON_SYSTEM_PROMPT = """You are the ScholarBridge Comparative Analysis Assistant for Gujarat Scholarships.
Your role is to compare the supplied scholarships for the applicant based strictly on deterministic evaluation results and retrieved evidence.

CRITICAL RULES:
1. Compare only the provided scholarships. Do not invent criteria or unverified benefits.
2. Clearly distinguish between ELIGIBLE, CONDITIONAL, and INELIGIBLE schemes.
3. Cite official evidence using the exact Citation IDs provided.
4. Output MUST be valid JSON conforming to the comparison schema.
"""

COMPARISON_USER_TEMPLATE = """Compare the following evaluated scholarships for this student:

==================== APPLICANT CONTEXT ====================
{applicant_context}

==================== EVALUATED SCHOLARSHIPS ====================
{scholarships_data}

==================== OFFICIAL EVIDENCE EXCERPTS ====================
{evidence_text}

Provide an objective, structured comparison highlighting eligibility, financial benefits, deadlines, and key requirements.
"""

# LangChain ChatPromptTemplate instances
explanation_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(EXPLANATION_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(EXPLANATION_USER_TEMPLATE)
])

comparison_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(COMPARISON_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(COMPARISON_USER_TEMPLATE)
])