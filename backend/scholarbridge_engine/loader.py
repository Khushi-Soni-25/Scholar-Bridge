"""
ScholarBridge Knowledge Base Loader
===================================
Loads the 5 normalized CSV tables, parses typed records, builds lookup
indexes, and validates relational integrity.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional
from .config import get_kb_dir
from .models import Scholarship, EligibilityRule, BenefitDetail, DocumentRequirement
from .validators import validate_knowledge_base_integrity, ValidationError


class KnowledgeBase:
    """In-memory indexed representation of the 5 scholarship knowledge base tables."""
    
    def __init__(self, kb_dir: Optional[Path] = None):
        self.kb_dir = kb_dir or get_kb_dir()
        self.scholarships: List[Scholarship] = []
        self.scholarships_by_id: Dict[str, Scholarship] = {}
        self.rules_by_scholarship: Dict[str, List[EligibilityRule]] = {}
        self.benefits_by_scholarship: Dict[str, List[BenefitDetail]] = {}
        self.docs_by_scholarship: Dict[str, List[DocumentRequirement]] = {}
        self.portals: List[Dict[str, str]] = []
        self._load_and_index()

    def _read_csv(self, filename: str) -> List[Dict[str, str]]:
        file_path = self.kb_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Knowledge base file missing: {file_path}")
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def _load_and_index(self):
        # 1. Load Master Table
        raw_master = self._read_csv("Gujarat_Scholarships_Master.csv")
        for row in raw_master:
            s = Scholarship(
                scholarship_id=row["Scholarship_ID"],
                scheme_code=row["Scheme_Code"],
                scholarship_name=row["Scholarship_Name"],
                short_name=row["Short_Name"],
                alternative_name=row.get("Alternative_Name", ""),
                scholarship_type=row["Scholarship_Type"],
                duplicate_group_id=row.get("Duplicate_Group_ID", ""),
                government_level=row["Government_Level"],
                ministry_or_body=row["Ministry_or_Body"],
                department=row["Department"],
                provider_type=row["Provider_Type"],
                provider_name=row["Provider_Name"],
                gujarat_scheme_type=row["Gujarat_Scheme_Type"],
                gujarat_applicability=row["Gujarat_Applicability"],
                domicile_state_required=row["Domicile_State_Required"],
                study_state_required=row["Study_State_Required"],
                eligible_categories=row["Eligible_Categories"],
                gender=row["Gender"],
                minority_eligible=row["Minority_Eligible"],
                disability_eligible=row["Disability_Eligible"],
                special_beneficiary_group=row["Special_Beneficiary_Group"],
                parent_occupation=row["Parent_Occupation"],
                education_level_normalized=row["Education_Level_Normalized"],
                stream=row["Stream"],
                course_specific=row.get("Course_Specific", ""),
                academic_criteria=row["Academic_Criteria"],
                min_marks_numeric=float(row["Min_Marks_Numeric"]) if row["Min_Marks_Numeric"] else -1.0,
                min_marks_type=row["Min_Marks_Type"],
                entrance_exam_required=row["Entrance_Exam_Required"],
                annual_income_limit_inr=row["Annual_Income_Limit_INR"],
                income_limit_min_inr=float(row["Income_Limit_Min_INR"]) if row["Income_Limit_Min_INR"] not in ["", "-1"] else 0.0,
                income_limit_max_inr=float(row["Income_Limit_Max_INR"]) if row["Income_Limit_Max_INR"] not in ["", "-1"] else -1.0,
                benefit_type=row["Benefit_Type"],
                tuition_fee_coverage=row.get("Tuition_Fee_Coverage", ""),
                benefit_description=row["Benefit_Description"],
                max_benefit_amount_inr=float(row["Max_Benefit_Amount_INR"]) if row["Max_Benefit_Amount_INR"] else 0.0,
                application_portal=row["Application_Portal"],
                official_website=row["Official_Website"],
                application_start_date=row["Application_Start_Date"],
                application_end_date=row["Application_End_Date"],
                application_cycle=row["Application_Cycle"],
                deadline_status=row["Deadline_Status"],
                eligibility_explanation_template=row["Eligibility_Explanation_Template"],
                ineligibility_exclusion_template=row["Ineligibility_Exclusion_Template"],
                confidence_score=float(row["Confidence_Score"]) if row["Confidence_Score"] else 0.85,
                official_source_url=row["Official_Source_URL"],
                verification_status=row["Verification_Status"],
                status=row["Status"]
            )
            self.scholarships.append(s)
            self.scholarships_by_id[s.scholarship_id] = s
            self.rules_by_scholarship[s.scholarship_id] = []
            self.benefits_by_scholarship[s.scholarship_id] = []
            self.docs_by_scholarship[s.scholarship_id] = []

        # 2. Load Eligibility Rules
        raw_rules = self._read_csv("Eligibility_Rules.csv")
        all_rules = []
        for row in raw_rules:
            r = EligibilityRule(
                rule_id=row["Rule_ID"],
                scholarship_id=row["Scholarship_ID"],
                rule_category=row["Rule_Category"],
                field=row["Field"],
                operator=row["Operator"],
                value=row["Value"],
                unit=row.get("Unit", "-"),
                mandatory=row.get("Mandatory", "Yes"),
                logical_operator=row.get("Logical_Operator", "AND"),
                condition_group=row.get("Condition_Group", "G1"),
                rule_description=row.get("Rule_Description", ""),
                source_url=row.get("Source_URL", "")
            )
            all_rules.append(r)
            if r.scholarship_id in self.rules_by_scholarship:
                self.rules_by_scholarship[r.scholarship_id].append(r)

        # 3. Load Benefit Details
        raw_bens = self._read_csv("Benefit_Details.csv")
        all_bens = []
        for row in raw_bens:
            b = BenefitDetail(
                benefit_entry_id=row["Benefit_Entry_ID"],
                scholarship_id=row["Scholarship_ID"],
                benefit_id=row["Benefit_ID"],
                benefit_category=row["Benefit_Category"],
                benefit_subcategory=row["Benefit_Subcategory"],
                amount_inr_fixed=float(row["Amount_INR_Fixed"]) if row["Amount_INR_Fixed"] else 0.0,
                amount_type=row["Amount_Type"],
                percentage_coverage=float(row["Percentage_Coverage"]) if row["Percentage_Coverage"] else 0.0,
                max_cap_inr=float(row["Max_Cap_INR"]) if row["Max_Cap_INR"] else 0.0,
                amount_period=row["Amount_Period"],
                beneficiary_group=row.get("Beneficiary_Group", "All"),
                condition_summary=row.get("Condition_Summary", ""),
                notes=row.get("Notes", "")
            )
            all_bens.append(b)
            if b.scholarship_id in self.benefits_by_scholarship:
                self.benefits_by_scholarship[b.scholarship_id].append(b)

        # 4. Load Document Requirements
        raw_docs = self._read_csv("Document_Requirements.csv")
        all_docs = []
        for row in raw_docs:
            d = DocumentRequirement(
                doc_requirement_id=row["Doc_Requirement_ID"],
                scholarship_id=row["Scholarship_ID"],
                document_name=row["Document_Name"],
                document_category=row["Document_Category"],
                mandatory=row.get("Mandatory", "Yes"),
                conditional_on=row.get("Conditional_On", ""),
                condition_description=row.get("Condition_Description", ""),
                notes=row.get("Notes", "")
            )
            all_docs.append(d)
            if d.scholarship_id in self.docs_by_scholarship:
                self.docs_by_scholarship[d.scholarship_id].append(d)

        # 5. Load Portals
        self.portals = self._read_csv("Gujarat_Portals_Reference.csv")

        # 6. Validate Referential Integrity
        is_valid, errors = validate_knowledge_base_integrity(
            self.scholarships, all_rules, all_bens, all_docs
        )
        if not is_valid:
            raise ValidationError(f"Knowledge Base Integrity Validation Failed:\n" + "\n".join(errors))


# Singleton cache
_KB_INSTANCE: Optional[KnowledgeBase] = None

def get_knowledge_base(kb_dir: Optional[Path] = None) -> KnowledgeBase:
    """Returns the cached knowledge base instance or creates one."""
    global _KB_INSTANCE
    if _KB_INSTANCE is None or (kb_dir and _KB_INSTANCE.kb_dir != kb_dir):
        _KB_INSTANCE = KnowledgeBase(kb_dir)
    return _KB_INSTANCE
