"""
ScholarBridge Document Ingestion & Normalization Pipeline
=========================================================
Normalizes official scholarship guidelines, government resolutions (GRs),
and portal documentation into structured hierarchical document models.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from .models import SourceMetadata
from ..models import Scholarship, EligibilityRule, BenefitDetail, DocumentRequirement


@dataclass
class DocumentSection:
    """Represents a structured section within an official scholarship document."""
    section_name: str              # e.g. 'Eligibility', 'Benefits', 'Documents Required'
    subsection_name: str           # e.g. 'Income Requirements', 'Tuition Fee Assistance'
    page_number: int
    content: str
    is_table: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedDocument:
    """A standardized, normalized document representation ready for structure-aware chunking."""
    source_metadata: SourceMetadata
    sections: List[DocumentSection] = field(default_factory=list)
    raw_text: str = ""


class DocumentIngestionPipeline:
    """Normalizes raw official guidelines and structured KB records into canonical documents."""

    @staticmethod
    def build_normalized_document_from_kb(
        scholarship: Scholarship,
        rules: List[EligibilityRule],
        benefits: List[BenefitDetail],
        documents: List[DocumentRequirement],
        source: SourceMetadata
    ) -> NormalizedDocument:
        """
        Synthesizes an authoritative, structured canonical document from
        verified knowledge base entities with exact official citations.
        """
        doc = NormalizedDocument(source_metadata=source)
        sections: List[DocumentSection] = []
        raw_text_parts: List[str] = []

        # Header Section (Page 1)
        header_text = (
            f"# {scholarship.scholarship_name} ({scholarship.short_name})\n"
            f"**Scheme Code**: {scholarship.scheme_code} | **Provider**: {scholarship.provider_name}\n"
            f"**Department / Body**: {scholarship.department} ({scholarship.ministry_or_body})\n"
            f"**Government Level**: {scholarship.government_level} | **Academic Year**: {scholarship.application_cycle}\n"
            f"**Official Portal**: {scholarship.application_portal} ({scholarship.official_website})\n"
            f"**Current Status**: {scholarship.status} | **Deadline Status**: {scholarship.deadline_status}\n"
        )
        sections.append(DocumentSection("Overview", "Scheme Details", 1, header_text, False, {
            "document_title": f"{scholarship.scholarship_name} Official Guidelines",
            "scheme_name": scholarship.scholarship_name,
            "short_name": scholarship.short_name,
            "provider_name": scholarship.provider_name
        }))
        raw_text_parts.append(header_text)

        # 1. Eligibility Criteria (Page 2)
        # 1.1 Income & Domicile
        inc_str = scholarship.annual_income_limit_inr
        inc_formatted = f"₹{scholarship.income_limit_max_inr:,.0f}" if scholarship.income_limit_max_inr > 0 else inc_str
        inc_text = (
            f"### 1.1 Income & Domicile Requirements — {scholarship.scholarship_name} ({scholarship.short_name})\n"
            f"**Scheme Code**: {scholarship.scheme_code} | **Scheme**: {scholarship.scholarship_name}\n"
            f"- **Annual Family Income Ceiling**: {inc_str} ({inc_formatted})\n"
            f"- **Income Floor Requirement**: ₹{scholarship.income_limit_min_inr:,.0f} (if applicable)\n"
            f"- **Domicile Requirement**: {scholarship.domicile_state_required} State Domicile Required\n"
            f"- **Study Location Requirement**: Study in {scholarship.study_state_required} Required\n"
        )
        sections.append(DocumentSection("Eligibility", "Income & Domicile", 2, inc_text, False, {
            "income_limit": scholarship.income_limit_max_inr,
            "domicile": scholarship.domicile_state_required,
            "document_title": f"{scholarship.scholarship_name} Official Guidelines",
            "scheme_name": scholarship.scholarship_name,
            "short_name": scholarship.short_name
        }))
        raw_text_parts.append(inc_text)

        # 1.2 Category & Gender
        cat_text = (
            f"### 1.2 Social Category & Gender Eligibility — {scholarship.scholarship_name} ({scholarship.short_name})\n"
            f"**Scheme Code**: {scholarship.scheme_code} | **Scheme**: {scholarship.scholarship_name}\n"
            f"- **Eligible Social Categories**: {scholarship.eligible_categories}\n"
            f"- **Gender Eligibility**: {scholarship.gender}\n"
            f"- **Minority Eligibility**: {scholarship.minority_eligible}\n"
            f"- **Disability Eligibility**: {scholarship.disability_eligible}\n"
            f"- **Special Group / Parent Occupation**: {scholarship.special_beneficiary_group or 'None'}\n"
        )
        sections.append(DocumentSection("Eligibility", "Category & Gender", 2, cat_text, False, {
            "category": scholarship.eligible_categories,
            "gender": scholarship.gender,
            "document_title": f"{scholarship.scholarship_name} Official Guidelines",
            "scheme_name": scholarship.scholarship_name,
            "short_name": scholarship.short_name
        }))
        raw_text_parts.append(cat_text)

        # 1.3 Academic & Course
        acad_text = (
            f"### 1.3 Academic & Course Criteria — {scholarship.scholarship_name} ({scholarship.short_name})\n"
            f"**Scheme Code**: {scholarship.scheme_code} | **Scheme**: {scholarship.scholarship_name}\n"
            f"- **Education Level**: {scholarship.education_level_normalized}\n"
            f"- **Applicable Stream(s)**: {scholarship.stream}\n"
            f"- **Course Scope**: {scholarship.course_specific or 'All recognized courses in stream'}\n"
            f"- **Minimum Qualifying Marks / Percentile**: {scholarship.academic_criteria}\n"
            f"- **Entrance Exam Requirement**: {scholarship.entrance_exam_required}\n"
        )
        sections.append(DocumentSection("Eligibility", "Academic & Course Requirements", 2, acad_text, False, {
            "education_level": scholarship.education_level_normalized,
            "stream": scholarship.stream,
            "document_title": f"{scholarship.scholarship_name} Official Guidelines",
            "scheme_name": scholarship.scholarship_name,
            "short_name": scholarship.short_name
        }))
        raw_text_parts.append(acad_text)

        # 2. Financial Benefits & Allowances (Page 3)
        ben_rows = []
        for b in benefits:
            fixed = f"₹{b.amount_inr_fixed:,.0f}" if b.amount_inr_fixed > 0 else "As approved"
            cap = f"₹{b.max_cap_inr:,.0f}" if b.max_cap_inr > 0 else "No cap"
            ben_rows.append(f"| {b.benefit_category} | {b.benefit_subcategory} | {fixed} | {b.amount_type} | {b.percentage_coverage}% | {cap} | {b.amount_period} |")

        table_text = (
            f"### 2.1 Benefit Schedule & Financial Assistance — {scholarship.scholarship_name} ({scholarship.short_name})\n"
            f"**Scheme Code**: {scholarship.scheme_code} | **Scheme**: {scholarship.scholarship_name}\n"
            f"**Benefit Summary**: {scholarship.benefit_description}\n\n"
            f"| Category | Component | Fixed Amount | Type | Coverage | Max Cap | Frequency |\n"
            f"|---|---|---|---|---|---|---|\n" +
            ("\n".join(ben_rows) if ben_rows else f"| Financial Grant | Educational Assistance | ₹{scholarship.max_benefit_amount_inr:,.0f} | Fixed | 100% | ₹{scholarship.max_benefit_amount_inr:,.0f} | Annual |") + "\n"
        )
        sections.append(DocumentSection("Benefits", "Financial Assistance Schedule", 3, table_text, True, {
            "max_benefit": scholarship.max_benefit_amount_inr,
            "document_title": f"{scholarship.scholarship_name} Official Guidelines",
            "scheme_name": scholarship.scholarship_name,
            "short_name": scholarship.short_name
        }))
        raw_text_parts.append(table_text)

        # 3. Mandatory Document Checklist (Page 4)
        doc_rows = []
        for d in documents:
            cond = f" (Conditional: {d.condition_description})" if d.condition_description else ""
            doc_rows.append(f"- **{d.document_name}** [{d.document_category}] — Mandatory: {d.mandatory}{cond}")

        doc_text = (
            f"### 3.1 Document Requirements Checklist — {scholarship.scholarship_name} ({scholarship.short_name})\n"
            f"**Scheme Code**: {scholarship.scheme_code} | **Scheme**: {scholarship.scholarship_name}\n"
            f"The following documents are required for application and verification on {scholarship.application_portal}:\n" +
            ("\n".join(doc_rows) if doc_rows else "- **Aadhaar Card** [Identity_Proof] — Mandatory: Yes\n- **Family Income Certificate** [Income_Proof] — Mandatory: Yes\n- **Bonafide Certificate** [Academic_Proof] — Mandatory: Yes") + "\n"
        )
        sections.append(DocumentSection("Documents Required", "Checklist & Proofs", 4, doc_text, False, {
            "document_title": f"{scholarship.scholarship_name} Official Guidelines",
            "scheme_name": scholarship.scholarship_name,
            "short_name": scholarship.short_name
        }))
        raw_text_parts.append(doc_text)

        # 4. Important Dates & Application Timeline (Page 5)
        timeline_text = (
            f"### 4.1 Important Dates & Application Window — {scholarship.scholarship_name} ({scholarship.short_name})\n"
            f"**Scheme Code**: {scholarship.scheme_code} | **Scheme**: {scholarship.scholarship_name}\n"
            f"- **Application Start Date**: {scholarship.application_start_date}\n"
            f"- **Application End Date / Deadline**: {scholarship.application_end_date}\n"
            f"- **Application Cycle**: {scholarship.application_cycle}\n"
            f"- **Deadline Status**: {scholarship.deadline_status}\n"
            f"- **Official Portal**: {scholarship.application_portal} ({scholarship.official_website})\n"
        )
        sections.append(DocumentSection("Important Dates", "Application Timeline", 5, timeline_text, False, {
            "deadline_status": scholarship.deadline_status,
            "document_title": f"{scholarship.scholarship_name} Official Guidelines",
            "scheme_name": scholarship.scholarship_name,
            "short_name": scholarship.short_name
        }))
        raw_text_parts.append(timeline_text)

        # 5. Terms, Conditions & Exclusions (Page 6)
        terms_text = (
            f"### 5.1 Exclusions, Renewal & Scheme Policies — {scholarship.scholarship_name} ({scholarship.short_name})\n"
            f"**Scheme Code**: {scholarship.scheme_code} | **Scheme**: {scholarship.scholarship_name}\n"
            f"- **Renewal Requirements**: Annual academic progression and minimum pass criteria required\n"
            f"- **Exclusions & Disqualifications**: {scholarship.ineligibility_exclusion_template or 'Students receiving duplicate government scholarships for same expense category are disqualified.'}\n"
            f"- **Official Guidelines URL**: {scholarship.official_source_url}\n"
        )
        sections.append(DocumentSection("Terms & Conditions", "Exclusions & Explanations", 6, terms_text, False, {
            "document_title": f"{scholarship.scholarship_name} Official Guidelines",
            "scheme_name": scholarship.scholarship_name,
            "short_name": scholarship.short_name
        }))
        raw_text_parts.append(terms_text)

        doc.sections = sections
        doc.raw_text = "\n\n".join(raw_text_parts)
        return doc
