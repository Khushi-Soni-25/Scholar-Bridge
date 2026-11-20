"""
ScholarBridge Data Models
=========================
Defines typed models for student profiles, scholarships, rules,
benefits, documents, match evaluations, and recommendation outputs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any


class EligibilityStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    CONDITIONAL = "CONDITIONAL"
    INELIGIBLE = "INELIGIBLE"
    SUSPENDED = "SUSPENDED"


@dataclass
class StudentProfile:
    """Represents a student profile provided to the engine for matching."""
    # Basic Demographics
    social_category: str                  # 'General', 'EWS', 'SC', 'ST', 'SEBC', 'OBC', 'NTDNT', 'Minority'
    gender: str                           # 'Male', 'Female', 'Other'
    annual_family_income: float           # Family income in INR (e.g. 250000.0)
    
    # Geography
    domicile_state: str = "Gujarat"       # Domicile state
    study_state: str = "Gujarat"          # State where the college/school is located
    district: Optional[str] = None        # Specific district in Gujarat
    
    # Academics
    education_level: str = "Undergraduate"# 'Primary', 'Secondary', 'Class_9', 'Class_10', 'Class_11', 'Class_12', 'Diploma', 'Undergraduate', 'Postgraduate', 'PhD', 'ITI'
    stream: str = "All"                   # 'Science', 'Commerce', 'Arts', 'Engineering', 'Medical', 'Technical', 'Law', 'Agriculture', 'Management'
    course_name: Optional[str] = None     # 'B.Tech Computer Science', 'MBBS', 'Diploma Mechanical'
    institution_name: Optional[str] = None# 'GTU', 'IIT Gandhinagar', 'MSU Baroda', etc.
    marks_percentage: Optional[float] = None # Percentage in qualifying examination (0-100)
    marks_percentile: Optional[float] = None # Percentile in qualifying board exam (0-100)
    is_d2d_admission: bool = False        # True if Diploma-to-Degree lateral entry admission
    entrance_exam_qualified: Optional[str] = None # 'JEE', 'NEET', 'GATE', 'GPAT', 'CAT', 'CLAT', 'UGC-NET', 'CSIR-NET', 'NMMS', 'YET'
    
    # Special Demographics & Context
    is_pwd: bool = False                  # True if student has benchmark disability
    disability_percentage: float = 0.0    # Disability percentage (e.g. 40.0, 50.0)
    has_udid_card: bool = False           # True if student has UDID disability card
    hostel_status: Optional[str] = None   # 'Day_Scholar', 'Hosteller', 'College_Affiliated_Hostel', 'Private_Hostel'
    parent_occupation: Optional[str] = None # 'Registered Construction Worker', 'Beedi / Mine Worker', 'Ex-Serviceman', 'CAPF Personnel'
    is_construction_worker_ward: bool = False # True if parent is registered construction worker (BOCWWB / e-Nirman card)
    is_defence_ward: bool = False         # True if ward of Ex-Serviceman or CAPF personnel
    is_orphan_or_martyr_ward: bool = False# True if orphan, COVID-19 single-parent loss, or martyr child
    is_sports_medalist: bool = False      # True if state/national sports medal winner
    age: Optional[int] = None             # Age in years


@dataclass
class Scholarship:
    """Represents a master scholarship entity from the knowledge base."""
    scholarship_id: str
    scheme_code: str
    scholarship_name: str
    short_name: str
    alternative_name: str
    scholarship_type: str
    duplicate_group_id: str
    government_level: str
    ministry_or_body: str
    department: str
    provider_type: str
    provider_name: str
    gujarat_scheme_type: str
    gujarat_applicability: str
    domicile_state_required: str
    study_state_required: str
    eligible_categories: str
    gender: str
    minority_eligible: str
    disability_eligible: str
    special_beneficiary_group: str
    parent_occupation: str
    education_level_normalized: str
    stream: str
    course_specific: str
    academic_criteria: str
    min_marks_numeric: float
    min_marks_type: str
    entrance_exam_required: str
    annual_income_limit_inr: str
    income_limit_min_inr: float
    income_limit_max_inr: float
    benefit_type: str
    tuition_fee_coverage: str
    benefit_description: str
    max_benefit_amount_inr: float
    application_portal: str
    official_website: str
    application_start_date: str
    application_end_date: str
    application_cycle: str
    deadline_status: str
    eligibility_explanation_template: str
    ineligibility_exclusion_template: str
    confidence_score: float               # Sourced from verified KB metadata (Source Verification Tier)
    official_source_url: str
    verification_status: str
    status: str


@dataclass
class EligibilityRule:
    """Represents an atomic machine-readable eligibility rule."""
    rule_id: str
    scholarship_id: str
    rule_category: str
    field: str
    operator: str
    value: str
    unit: str
    mandatory: str
    logical_operator: str
    condition_group: str
    rule_description: str
    source_url: str


@dataclass
class BenefitDetail:
    """Represents a specific benefit component of a scholarship."""
    benefit_entry_id: str
    scholarship_id: str
    benefit_id: str
    benefit_category: str
    benefit_subcategory: str
    amount_inr_fixed: float
    amount_type: str
    percentage_coverage: float
    max_cap_inr: float
    amount_period: str
    beneficiary_group: str
    condition_summary: str
    notes: str


@dataclass
class DocumentRequirement:
    """Represents a required document for a scholarship application."""
    doc_requirement_id: str
    scholarship_id: str
    document_name: str
    document_category: str
    mandatory: str
    conditional_on: str
    condition_description: str
    notes: str


@dataclass
class RuleEvaluation:
    """Result of evaluating a single rule against a student profile."""
    rule: EligibilityRule
    passed: bool
    is_conditional: bool = False
    reason: str = ""
    actual_value: Any = None
    expected_value: Any = None


@dataclass
class MatchResult:
    """Evaluation outcome for one scholarship against a student profile."""
    scholarship: Scholarship
    status: EligibilityStatus
    passed_rules: List[RuleEvaluation] = field(default_factory=list)
    failed_rules: List[RuleEvaluation] = field(default_factory=list)
    conditional_rules: List[RuleEvaluation] = field(default_factory=list)
    explanation: str = ""
    exclusion_reason: str = ""
    documents: List[DocumentRequirement] = field(default_factory=list)
    benefits: List[BenefitDetail] = field(default_factory=list)
    explanation_details: Dict[str, Any] = field(default_factory=dict)
    timeline_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationItem:
    """A ranked scholarship with multi-factor component and composite scores."""
    scholarship: Scholarship
    match_result: MatchResult
    benefit_score: float         # 0 - 100 (Financial grant / fee waiver value)
    confidence_score: float      # 0 - 100 (Source verification reliability tier)
    urgency_score: float         # 0 - 100 (Application timeline urgency)
    specificity_score: float     # 0 - 100 (Alignment with student's exact demographic/academic profile)
    final_score: float           # 0 - 100 composite heuristic recommendation score
    rank: int = 0
    timeline_info: Dict[str, Any] = field(default_factory=dict)
    explanation_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedDocument:
    """Deduplicated document across multiple matched scholarships."""
    document_name: str
    document_category: str
    is_mandatory: bool
    requesting_scholarships: List[str]
    conditional_notes: List[str] = field(default_factory=list)


@dataclass
class EngineOutput:
    """Complete, structured output generated by ScholarBridge pipeline."""
    student_profile: StudentProfile
    total_scholarships_evaluated: int
    eligible_count: int
    conditional_count: int
    ineligible_count: int
    ranked_recommendations: List[RecommendationItem]
    conditional_recommendations: List[RecommendationItem]
    excluded_scholarships: List[MatchResult]
    unified_checklist: List[UnifiedDocument]
    # --- Step 10: Sections 11-14 pipeline output (additive; existing fields
    # above are UNCHANGED so existing consumers, e.g. EngineEvidenceBridge,
    # keep working exactly as before). Populated by ScholarBridgeEngine.recommend();
    # default to empty so any other EngineOutput construction site remains valid.
    hybrid_results: List[Dict[str, Any]] = field(default_factory=list)      # Section 11, per-scholarship
    ranking: Dict[str, Any] = field(default_factory=dict)                  # Section 12 (ranker.rank_scholarships output)
    final_recommendations: Dict[str, Any] = field(default_factory=dict)    # Section 13 (recommender.build_final_recommendations output)
    explanations: Dict[str, Any] = field(default_factory=dict)             # Section 14A (explainer.build_final_recommendation_explanations output)
