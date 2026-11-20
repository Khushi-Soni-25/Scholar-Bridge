"""
ScholarBridge MongoDB-Backed Knowledge Base (Production Module)
==================================================================
Step 10 integration note: this module PORTS the MongoDB loading logic that
already existed in `recommendation_engine.ipynb` (Sections 3-5) into a real,
importable production module, so `ScholarBridgeEngine` can use a
MongoDB-backed knowledge base without that logic living only inside the
notebook. The field-coercion helpers, per-collection document builders, and
`MongoKnowledgeBase` class below are a faithful, unmodified port of the
notebook's own implementation -- no new loading/mapping logic was written,
and `loader.py`'s exact per-field default/exclusion rules are preserved.

`MongoKnowledgeBase` exposes the SAME public attributes as the existing
CSV-backed `scholarbridge_engine.loader.KnowledgeBase`, so `RuleMatcher`,
`RecommendationScorer`, `hybrid_scorer`, `ranker`, `recommender`, and
`explainer` can all consume it without any changes:
    .scholarships              List[Scholarship]
    .scholarships_by_id        Dict[str, Scholarship]
    .rules_by_scholarship      Dict[str, List[EligibilityRule]]
    .benefits_by_scholarship   Dict[str, List[BenefitDetail]]
    .docs_by_scholarship       Dict[str, List[DocumentRequirement]]
    .portals                   List[Dict[str, str]]

`pymongo` is imported lazily (only inside `get_mongo_knowledge_base`), so
importing this module -- and importing `scholarbridge_engine.engine`, which
now references it for an OPTIONAL code path -- never fails in an
environment where `pymongo` isn't installed. The CSV-backed `KnowledgeBase`
remains the default knowledge-base source everywhere; this module is an
opt-in alternative, not a replacement.
"""

from typing import Any, Dict, List, Optional

from .models import BenefitDetail, DocumentRequirement, EligibilityRule, Scholarship
from .validators import ValidationError, validate_knowledge_base_integrity

# Logical name -> default MongoDB collection name (mirrors the notebook's
# COLLECTION_MAP, Section 3). Callers can override per-deployment.
DEFAULT_COLLECTION_MAP: Dict[str, str] = {
    "scholarships": "scholarships_master",
    "rules": "eligibility_rules",
    "benefits": "benefit_details",
    "documents": "document_requirements",
    "portals": "portals_reference",
}


class MongoUnavailableError(RuntimeError):
    """Raised when a MongoDB-backed knowledge base is requested but pymongo
    is not installed, or MONGODB_URI/MONGODB_DB_NAME are not configured, or
    the connection/ping fails. Callers should catch this and fall back to
    the CSV-backed KnowledgeBase (the default) rather than crash."""


# --- Field-coercion helpers (ported verbatim from the notebook, Section 5) --
# Mongo documents may store numeric fields as native int/float (if imported
# with type inference) or as strings (if imported as plain text). These
# helpers accept either, and faithfully reproduce loader.py's exact
# per-field default/exclusion rules so behavior matches the CSV-backed
# loader precisely.

def _s(value: Any, default: str = "") -> str:
    """Coerce a Mongo field to a stripped string (mirrors csv.DictReader's str values)."""
    if value is None:
        return default
    return str(value).strip()


def _num(value: Any, default: float, exclude_neg1: bool = False) -> float:
    """
    Coerce a Mongo field to float, reproducing loader.py's exact per-field rules:
    - empty/missing -> default
    - for fields where loader.py also excludes the literal '-1' (income limits),
      pass exclude_neg1=True
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if exclude_neg1 and float(value) == -1.0:
            return default
        return float(value)
    s = str(value).strip()
    if s == "":
        return default
    if exclude_neg1 and s == "-1":
        return default
    return float(s)


# --- Row -> dataclass builders (ported verbatim; mirror loader.py's field mapping exactly) --

def build_scholarship(doc: Dict[str, Any]) -> Scholarship:
    return Scholarship(
        scholarship_id=_s(doc.get("Scholarship_ID")),
        scheme_code=_s(doc.get("Scheme_Code")),
        scholarship_name=_s(doc.get("Scholarship_Name")),
        short_name=_s(doc.get("Short_Name")),
        alternative_name=_s(doc.get("Alternative_Name")),
        scholarship_type=_s(doc.get("Scholarship_Type")),
        duplicate_group_id=_s(doc.get("Duplicate_Group_ID")),
        government_level=_s(doc.get("Government_Level")),
        ministry_or_body=_s(doc.get("Ministry_or_Body")),
        department=_s(doc.get("Department")),
        provider_type=_s(doc.get("Provider_Type")),
        provider_name=_s(doc.get("Provider_Name")),
        gujarat_scheme_type=_s(doc.get("Gujarat_Scheme_Type")),
        gujarat_applicability=_s(doc.get("Gujarat_Applicability")),
        domicile_state_required=_s(doc.get("Domicile_State_Required")),
        study_state_required=_s(doc.get("Study_State_Required")),
        eligible_categories=_s(doc.get("Eligible_Categories")),
        gender=_s(doc.get("Gender")),
        minority_eligible=_s(doc.get("Minority_Eligible")),
        disability_eligible=_s(doc.get("Disability_Eligible")),
        special_beneficiary_group=_s(doc.get("Special_Beneficiary_Group")),
        parent_occupation=_s(doc.get("Parent_Occupation")),
        education_level_normalized=_s(doc.get("Education_Level_Normalized")),
        stream=_s(doc.get("Stream")),
        course_specific=_s(doc.get("Course_Specific")),
        academic_criteria=_s(doc.get("Academic_Criteria")),
        min_marks_numeric=_num(doc.get("Min_Marks_Numeric"), -1.0),
        min_marks_type=_s(doc.get("Min_Marks_Type")),
        entrance_exam_required=_s(doc.get("Entrance_Exam_Required")),
        annual_income_limit_inr=_s(doc.get("Annual_Income_Limit_INR")),
        income_limit_min_inr=_num(doc.get("Income_Limit_Min_INR"), 0.0, exclude_neg1=True),
        income_limit_max_inr=_num(doc.get("Income_Limit_Max_INR"), -1.0, exclude_neg1=True),
        benefit_type=_s(doc.get("Benefit_Type")),
        tuition_fee_coverage=_s(doc.get("Tuition_Fee_Coverage")),
        benefit_description=_s(doc.get("Benefit_Description")),
        max_benefit_amount_inr=_num(doc.get("Max_Benefit_Amount_INR"), 0.0),
        application_portal=_s(doc.get("Application_Portal")),
        official_website=_s(doc.get("Official_Website")),
        application_start_date=_s(doc.get("Application_Start_Date")),
        application_end_date=_s(doc.get("Application_End_Date")),
        application_cycle=_s(doc.get("Application_Cycle")),
        deadline_status=_s(doc.get("Deadline_Status")),
        eligibility_explanation_template=_s(doc.get("Eligibility_Explanation_Template")),
        ineligibility_exclusion_template=_s(doc.get("Ineligibility_Exclusion_Template")),
        confidence_score=_num(doc.get("Confidence_Score"), 0.85),
        official_source_url=_s(doc.get("Official_Source_URL")),
        verification_status=_s(doc.get("Verification_Status")),
        status=_s(doc.get("Status")),
    )


def build_rule(doc: Dict[str, Any]) -> EligibilityRule:
    return EligibilityRule(
        rule_id=_s(doc.get("Rule_ID")),
        scholarship_id=_s(doc.get("Scholarship_ID")),
        rule_category=_s(doc.get("Rule_Category")),
        field=_s(doc.get("Field")),
        operator=_s(doc.get("Operator")),
        value=_s(doc.get("Value")),
        unit=_s(doc.get("Unit")) or "-",
        mandatory=_s(doc.get("Mandatory")) or "Yes",
        logical_operator=_s(doc.get("Logical_Operator")) or "AND",
        condition_group=_s(doc.get("Condition_Group")) or "G1",
        rule_description=_s(doc.get("Rule_Description")),
        source_url=_s(doc.get("Source_URL")),
    )


def build_benefit(doc: Dict[str, Any]) -> BenefitDetail:
    return BenefitDetail(
        benefit_entry_id=_s(doc.get("Benefit_Entry_ID")),
        scholarship_id=_s(doc.get("Scholarship_ID")),
        benefit_id=_s(doc.get("Benefit_ID")),
        benefit_category=_s(doc.get("Benefit_Category")),
        benefit_subcategory=_s(doc.get("Benefit_Subcategory")),
        amount_inr_fixed=_num(doc.get("Amount_INR_Fixed"), 0.0),
        amount_type=_s(doc.get("Amount_Type")),
        percentage_coverage=_num(doc.get("Percentage_Coverage"), 0.0),
        max_cap_inr=_num(doc.get("Max_Cap_INR"), 0.0),
        amount_period=_s(doc.get("Amount_Period")),
        beneficiary_group=_s(doc.get("Beneficiary_Group")) or "All",
        condition_summary=_s(doc.get("Condition_Summary")),
        notes=_s(doc.get("Notes")),
    )


def build_document(doc: Dict[str, Any]) -> DocumentRequirement:
    return DocumentRequirement(
        doc_requirement_id=_s(doc.get("Doc_Requirement_ID")),
        scholarship_id=_s(doc.get("Scholarship_ID")),
        document_name=_s(doc.get("Document_Name")),
        document_category=_s(doc.get("Document_Category")),
        mandatory=_s(doc.get("Mandatory")) or "Yes",
        conditional_on=_s(doc.get("Conditional_On")),
        condition_description=_s(doc.get("Condition_Description")),
        notes=_s(doc.get("Notes")),
    )


class MongoKnowledgeBase:
    """
    MongoDB-backed equivalent of scholarbridge_engine.loader.KnowledgeBase.
    Ported verbatim from the notebook (Section 5) -- same public attributes,
    same loading/validation behavior, so matcher.py / scorer.py / engine.py
    can consume it exactly like the CSV-backed KnowledgeBase.
    """

    def __init__(self, database, collection_map: Optional[Dict[str, str]] = None):
        self.db = database
        self.collection_map = collection_map or DEFAULT_COLLECTION_MAP

        self.scholarships: List[Scholarship] = []
        self.scholarships_by_id: Dict[str, Scholarship] = {}
        self.rules_by_scholarship: Dict[str, List[EligibilityRule]] = {}
        self.benefits_by_scholarship: Dict[str, List[BenefitDetail]] = {}
        self.docs_by_scholarship: Dict[str, List[DocumentRequirement]] = {}
        self.portals: List[Dict[str, str]] = []

        self.validation_errors: List[str] = []
        self.is_valid: bool = False

        self._load_and_index()

    def _fetch(self, logical_name: str) -> List[Dict[str, Any]]:
        coll_name = self.collection_map[logical_name]
        docs = list(self.db[coll_name].find({}))
        if not docs:
            raise FileNotFoundError(
                f"MongoDB collection '{coll_name}' (logical name '{logical_name}') "
                f"returned zero documents. Check the collection map and that the "
                f"database has been populated."
            )
        return docs

    def _load_and_index(self):
        # 1. Master scholarships
        for doc in self._fetch("scholarships"):
            s = build_scholarship(doc)
            self.scholarships.append(s)
            self.scholarships_by_id[s.scholarship_id] = s
            self.rules_by_scholarship[s.scholarship_id] = []
            self.benefits_by_scholarship[s.scholarship_id] = []
            self.docs_by_scholarship[s.scholarship_id] = []

        # 2. Eligibility rules
        all_rules: List[EligibilityRule] = []
        for doc in self._fetch("rules"):
            r = build_rule(doc)
            all_rules.append(r)
            if r.scholarship_id in self.rules_by_scholarship:
                self.rules_by_scholarship[r.scholarship_id].append(r)

        # 3. Benefit details
        all_benefits: List[BenefitDetail] = []
        for doc in self._fetch("benefits"):
            b = build_benefit(doc)
            all_benefits.append(b)
            if b.scholarship_id in self.benefits_by_scholarship:
                self.benefits_by_scholarship[b.scholarship_id].append(b)

        # 4. Document requirements
        all_docs: List[DocumentRequirement] = []
        for doc in self._fetch("documents"):
            d = build_document(doc)
            all_docs.append(d)
            if d.scholarship_id in self.docs_by_scholarship:
                self.docs_by_scholarship[d.scholarship_id].append(d)

        # 5. Portals (raw dict passthrough, mirroring loader.py)
        self.portals = [
            {k: (str(v) if v is not None else "") for k, v in doc.items() if k != "_id"}
            for doc in self._fetch("portals")
        ]

        # 6. Referential/schema integrity validation -- REUSING validators.py UNCHANGED
        is_valid, errors = validate_knowledge_base_integrity(
            self.scholarships, all_rules, all_benefits, all_docs
        )
        self.is_valid = is_valid
        self.validation_errors = errors
        if not is_valid:
            raise ValidationError(
                "Knowledge Base Integrity Validation Failed (MongoDB source):\n" + "\n".join(errors)
            )


def get_mongo_knowledge_base(
    mongo_uri: Optional[str] = None,
    db_name: Optional[str] = None,
    collection_map: Optional[Dict[str, str]] = None,
) -> MongoKnowledgeBase:
    """Factory: connects to MongoDB and returns a populated MongoKnowledgeBase.

    `pymongo` is imported lazily here (not at module scope), so environments
    without pymongo installed can still import `scholarbridge_engine.engine`
    and use the default CSV-backed KnowledgeBase without error -- this
    function is only called when a caller explicitly opts into Mongo.

    Raises MongoUnavailableError (never a raw pymongo/network exception) if
    pymongo isn't installed, the URI/DB name aren't provided, or the
    connection fails -- so callers (e.g. ScholarBridgeEngine) can catch this
    one exception type and fall back to the CSV KnowledgeBase.
    """
    if not mongo_uri or not db_name:
        raise MongoUnavailableError(
            "MongoDB knowledge base requested but mongo_uri/db_name were not provided "
            "(e.g. via the MONGODB_URI / MONGODB_DB_NAME environment variables)."
        )

    try:
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError
    except ImportError as exc:
        raise MongoUnavailableError(
            "MongoDB knowledge base requested but the 'pymongo' package is not installed "
            "in this environment. Install it (see requirements.txt: pymongo==4.10.1) or "
            "use the default CSV-backed KnowledgeBase."
        ) from exc

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=8000)
        client.admin.command("ping")
        db = client[db_name]
    except PyMongoError as exc:
        raise MongoUnavailableError(f"Could not connect to MongoDB at the configured URI: {exc}") from exc

    return MongoKnowledgeBase(db, collection_map or DEFAULT_COLLECTION_MAP)
