"""
ScholarBridge Exhaustive Validators & Startup Integrity Gate
============================================================
Performs exhaustive schema, referential integrity, operator validity,
and data sanity validations on the knowledge base at startup.
"""

from typing import List, Dict, Tuple, Set
from .models import StudentProfile, Scholarship, EligibilityRule, BenefitDetail, DocumentRequirement


class ValidationError(Exception):
    pass


from .matcher import SUPPORTED_RULE_FIELDS

SUPPORTED_OPERATORS: Set[str] = {
    "=", "==", "!=", "<", "<=", ">", ">=", "IN", "NOT_IN", "BETWEEN"
}

VALID_STATUSES: Set[str] = {
    "ACTIVE", "SUSPENDED", "DISCONTINUED", "VERIFIED", "PARTIALLY_VERIFIED", "NEEDS_REVIEW"
}


def normalize_category(cat: str) -> str:
    """Normalizes informal category terms to standard tokens."""
    c = str(cat).strip().upper()
    if c in ["SC", "SCHEDULED CASTE"]: return "SC"
    if c in ["ST", "SCHEDULED TRIBE"]: return "ST"
    if c in ["SEBC", "OBC", "OTHER BACKWARD CLASS", "BAXI PANCH"]: return "SEBC"
    if c in ["EWS", "ECONOMICALLY WEAKER SECTION"]: return "EWS"
    if c in ["NTDNT", "NT", "DNT", "NOMADIC"]: return "NTDNT"
    if c in ["MINORITY", "MUSLIM", "CHRISTIAN", "SIKH", "BUDDHIST", "JAIN", "PARSI"]: return "Minority"
    return "General"


def normalize_gender(gen: str) -> str:
    """Normalizes gender to Male / Female / Other / All."""
    g = str(gen).strip().capitalize()
    if g.startswith("F") or "GIRL" in g.upper() or "WOMEN" in g.upper():
        return "Female"
    if g.startswith("M") or "BOY" in g.upper():
        return "Male"
    return "Other"


def validate_student_profile(profile: StudentProfile) -> Tuple[bool, List[str]]:
    """Validates student profile data and returns validation errors if any."""
    errors = []
    
    # Income Check
    if profile.annual_family_income < 0:
        errors.append("Annual family income cannot be negative.")
        
    # Marks Check (skip validation if unstated / None)
    if profile.marks_percentage is not None and (profile.marks_percentage < 0 or profile.marks_percentage > 100):
        errors.append("Marks percentage must be between 0 and 100.")
    if profile.marks_percentile is not None and (profile.marks_percentile < 0 or profile.marks_percentile > 100):
        errors.append("Marks percentile must be between 0 and 100.")
        
    # Disability Check
    if profile.is_pwd and profile.disability_percentage < 0:
        errors.append("Disability percentage cannot be negative.")
        
    # Normalize fields in-place
    profile.social_category = normalize_category(profile.social_category)
    profile.gender = normalize_gender(profile.gender)
    
    return len(errors) == 0, errors


def validate_knowledge_base_integrity(
    scholarships: List[Scholarship],
    rules: List[EligibilityRule],
    benefits: List[BenefitDetail],
    documents: List[DocumentRequirement]
) -> Tuple[bool, List[str]]:
    """
    Exhaustive Startup Integrity Gate:
    Validates primary keys, foreign keys, supported fields, operators,
    numeric intervals, URLs, and status consistency.
    """
    errors = []
    
    # 1. Primary Key Uniqueness
    master_ids = set()
    for s in scholarships:
        if s.scholarship_id in master_ids:
            errors.append(f"Duplicate Scholarship_ID in Master table: {s.scholarship_id}")
        master_ids.add(s.scholarship_id)
        
        # 1.1 Income Interval Consistency
        if s.income_limit_min_inr > 0 and s.income_limit_max_inr > 0:
            if s.income_limit_min_inr >= s.income_limit_max_inr:
                errors.append(f"Invalid income interval for {s.scholarship_id}: Min (₹{s.income_limit_min_inr}) >= Max (₹{s.income_limit_max_inr})")
                
        # 1.2 Verification Status Validity
        if s.verification_status.upper() not in VALID_STATUSES:
            errors.append(f"Invalid Verification_Status '{s.verification_status}' in {s.scholarship_id}")
            
        # 1.3 URL Validity Check
        if not s.official_source_url.startswith("http"):
            errors.append(f"Malformed or missing Official_Source_URL for {s.scholarship_id}: '{s.official_source_url}'")
            
        # 1.4 Suspended Consistency
        if s.status == "Suspended" and s.deadline_status not in ["SUSPENDED", "CLOSED"]:
            errors.append(f"Inconsistent deadline status for suspended scheme {s.scholarship_id}: {s.deadline_status}")

    # 2. Rule Validation & Referential Integrity
    rule_sids = set()
    for r in rules:
        rule_sids.add(r.scholarship_id)
        if r.scholarship_id not in master_ids:
            errors.append(f"Orphan rule {r.rule_id} pointing to non-existent scholarship: {r.scholarship_id}")
        if r.field not in SUPPORTED_RULE_FIELDS:
            errors.append(f"Unsupported rule field '{r.field}' in rule {r.rule_id} ({r.scholarship_id})")
        if r.operator not in SUPPORTED_OPERATORS:
            errors.append(f"Unsupported operator '{r.operator}' in rule {r.rule_id} ({r.scholarship_id})")

    # Check for scholarships without rules
    schols_without_rules = master_ids - rule_sids
    if schols_without_rules:
        errors.append(f"Scholarships missing eligibility rules in KB: {schols_without_rules}")

    # 3. Benefit Referential Integrity
    for b in benefits:
        if b.scholarship_id not in master_ids:
            errors.append(f"Orphan benefit {b.benefit_entry_id} pointing to non-existent scholarship: {b.scholarship_id}")

    # 4. Document Requirements Referential Integrity
    for d in documents:
        if d.scholarship_id not in master_ids:
            errors.append(f"Orphan document {d.doc_requirement_id} pointing to non-existent scholarship: {d.scholarship_id}")

    return len(errors) == 0, errors
