"""
ScholarBridge Document Checklist Aggregator
============================================
Aggregates, deduplicates, and organizes document requirements across all
matched scholarships into a unified, actionable checklist for the student.
"""

from typing import List, Dict
from .models import DocumentRequirement, UnifiedDocument, RecommendationItem


class DocumentAggregator:
    """Aggregates and deduplicates document requirements for student applications."""

    @staticmethod
    def build_unified_checklist(
        recommendations: List[RecommendationItem],
        docs_by_scholarship: Dict[str, List[DocumentRequirement]]
    ) -> List[UnifiedDocument]:
        """Merges document requirements across all recommended scholarships."""
        
        doc_map: Dict[str, UnifiedDocument] = {}
        
        for rec in recommendations:
            sid = rec.scholarship.scholarship_id
            s_name = rec.scholarship.short_name
            req_docs = docs_by_scholarship.get(sid, [])
            
            for d in req_docs:
                clean_name = d.document_name.strip()
                is_mand = d.mandatory.lower() == "yes"
                
                if clean_name not in doc_map:
                    doc_map[clean_name] = UnifiedDocument(
                        document_name=clean_name,
                        document_category=d.document_category,
                        is_mandatory=is_mand,
                        requesting_scholarships=[s_name],
                        conditional_notes=[d.condition_description] if d.condition_description else []
                    )
                else:
                    if s_name not in doc_map[clean_name].requesting_scholarships:
                        doc_map[clean_name].requesting_scholarships.append(s_name)
                    if is_mand:
                        doc_map[clean_name].is_mandatory = True
                    if d.condition_description and d.condition_description not in doc_map[clean_name].conditional_notes:
                        doc_map[clean_name].conditional_notes.append(d.condition_description)

        # Sort: Mandatory first, then by Category, then alphabetically
        checklist = list(doc_map.values())
        checklist.sort(key=lambda x: (not x.is_mandatory, x.document_category, x.document_name))
        return checklist
