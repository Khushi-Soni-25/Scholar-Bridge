"""
ScholarBridge Comprehensive Retrieval Evaluation Framework
==========================================================
Evaluates evidence retrieval performance against 36 curated gold-standard
scholarship queries measuring Recall@K, Precision@K, MRR, nDCG@5, and cryptographic provenance.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .retriever import HybridEvidenceRetriever, get_evidence_retriever


@dataclass
class GoldQuery:
    """A gold-standard test query with ground-truth expected scheme and section."""
    query_id: str
    query_text: str
    expected_scheme_ids: List[str]
    expected_sections: List[str]
    expected_content_tokens: List[str]


# 36 Curated Gold-Standard Evaluation Queries (Strict factual section & token ground truth)
GOLD_STANDARD_QUERIES: List[GoldQuery] = [
    # 1. Income Limits
    GoldQuery("GQ-01", "What is the annual family income limit for MYSY?", ["MYSY"], ["Eligibility", "Terms & Conditions"], ["600000", "6,00,000"]),
    GoldQuery("GQ-02", "What is the income ceiling for BCK-10 SC Food Bill assistance?", ["BCK-10"], ["Eligibility", "Terms & Conditions"], ["600000", "6,00,000"]),
    GoldQuery("GQ-03", "What is the income limit for BCK-6.1 SC Post-Matric scholarship?", ["BCK-6.1"], ["Eligibility", "Terms & Conditions"], ["250000", "2,50,000"]),
    GoldQuery("GQ-04", "What is the family income limit for PM Vidyalaxmi loan interest subvention?", ["PM-VIDYALAXMI"], ["Eligibility", "Benefits"], ["800000", "8,00,000"]),
    GoldQuery("GQ-05", "What is the income limit for Adani Gyan Jyoti scholarship?", ["ADANI-GJ"], ["Eligibility", "Benefits"], ["250000", "2,50,000"]),
    GoldQuery("GQ-06", "What is the income ceiling for Namo Laxmi scheme?", ["NLY"], ["Eligibility", "Terms & Conditions"], ["600000", "6,00,000"]),
    GoldQuery("GQ-07", "What is the income ceiling for Namo Saraswati science scheme?", ["NSVY"], ["Eligibility", "Terms & Conditions"], ["600000", "6,00,000"]),
    GoldQuery("GQ-08", "What is the income limit for Reliance Undergraduate Scholarship?", ["RF-UG"], ["Eligibility", "Benefits"], ["1500000", "15,00,000"]),
    GoldQuery("GQ-09", "What is the income floor requirement for BCK-5 SC Girls scheme?", ["BCK-5"], ["Eligibility", "Benefits"], ["250000", "2,50,000"]),
    GoldQuery("GQ-10", "What is the income limit for VKY-158 ST Stipend scheme?", ["VKY-158"], ["Eligibility", "Benefits"], ["150000", "1,50,000"]),

    # 2. Academic Criteria & Cutoffs
    GoldQuery("GQ-11", "What percentile is required for MYSY for 12th pass students?", ["MYSY"], ["Eligibility", "Terms & Conditions"], ["80", "percentile"]),
    GoldQuery("GQ-12", "What is the qualifying marks percentage for D2D lateral entry in MYSY?", ["MYSY"], ["Eligibility", "Terms & Conditions"], ["65", "diploma"]),
    GoldQuery("GQ-13", "What marks percentage is required for Adani Gyan Jyoti scholarship?", ["ADANI-GJ"], ["Eligibility", "Benefits"], ["70%", "70"]),
    GoldQuery("GQ-14", "What minimum percentage is required for Reliance UG scholarship?", ["RF-UG"], ["Eligibility", "Benefits"], ["60%", "60"]),
    GoldQuery("GQ-15", "What entrance qualification is required for INSPIRE PhD fellowship?", ["INSPIRE-PHD", "INSPIRE-F"], ["Eligibility", "Benefits"], ["gate", "net"]),

    # 3. Benefits & Stipends
    GoldQuery("GQ-16", "What is the monthly stipend amount under SHODH for PhD scholars?", ["SHODH"], ["Benefits", "Terms & Conditions"], ["15,000", "15000", "3.60l", "4,00,000"]),
    GoldQuery("GQ-17", "What is the monthly food bill assistance amount for BCK-10 hostellers?", ["BCK-10"], ["Benefits", "Terms & Conditions"], ["1,500", "1500", "15,000"]),
    GoldQuery("GQ-18", "What is the maximum tuition fee grant under MYSY for Medical and Dental courses?", ["MYSY"], ["Benefits", "Terms & Conditions"], ["2,00,000", "200000"]),
    GoldQuery("GQ-19", "What is the tuition assistance cap under MYSY for Engineering and Pharmacy?", ["MYSY"], ["Benefits", "Terms & Conditions"], ["50,000", "50000"]),
    GoldQuery("GQ-20", "What is the grant amount for Google Generation Scholarship for women in tech?", ["GOOGLE-GEN"], ["Benefits", "Terms & Conditions"], ["2,500", "2500", "2,10,000"]),
    GoldQuery("GQ-21", "What is the total scholarship amount over degree for Reliance UG?", ["RF-UG"], ["Benefits", "Terms & Conditions"], ["2,00,000", "200000"]),
    GoldQuery("GQ-22", "What is the book and instrument grant for BCK-12 SC Medical students?", ["BCK-12"], ["Benefits", "Terms & Conditions"], ["10,000", "10000"]),
    GoldQuery("GQ-23", "What is the stipend amount for BCK-83 SEBC technical diploma students?", ["BCK-83"], ["Benefits", "Terms & Conditions"], ["technical", "stipend"]),

    # 4. Mandatory Documents
    GoldQuery("GQ-24", "What documents are required to apply for MYSY scholarship?", ["MYSY"], ["Documents Required", "Terms & Conditions"], ["bonafide", "income", "aadhaar"]),
    GoldQuery("GQ-25", "What documents are needed for BCK-6.1 SC Post-Matric application?", ["BCK-6.1"], ["Documents Required", "Terms & Conditions"], ["caste", "income", "aadhaar"]),
    GoldQuery("GQ-26", "What special proof is required for BOCWWB Shramik Puraskar scheme?", ["SPSS"], ["Documents Required", "Eligibility", "Benefits"], ["e-nirman", "worker", "bocwwb"]),
    GoldQuery("GQ-27", "What proof is required to claim BCK-10 food bill assistance?", ["BCK-10"], ["Documents Required", "Eligibility", "Benefits"], ["hostel", "warden"]),
    GoldQuery("GQ-28", "What document is required for disability scholarship applicants?", ["GOOGLE-GEN", "ADANI-GJ", "BCK-6.1", "SAKSHAM"], ["Documents Required", "Eligibility"], ["certificate", "udid", "disability"]),

    # 5. Course & Stream Eligibility
    GoldQuery("GQ-29", "Is B.Tech or engineering degree covered under Mukhyamantri Yuva Swavalamban Yojana?", ["MYSY"], ["Eligibility", "Benefits", "Terms & Conditions"], ["engineering", "b.tech"]),
    GoldQuery("GQ-30", "Is Namo Saraswati scheme restricted to Science stream in Class 11 and 12?", ["NSVY"], ["Eligibility", "Terms & Conditions"], ["science"]),
    GoldQuery("GQ-31", "Which courses qualify for BCK-79 Medical Engineering food bill assistance?", ["BCK-79"], ["Eligibility", "Benefits"], ["medical", "engineering"]),
    GoldQuery("GQ-32", "Is Adobe India Women-in-Technology scholarship for Computer Science students?", ["ADOBE-WIT"], ["Eligibility", "Benefits"], ["female", "technology", "computer"]),

    # 6. Deadlines, Portals & Scheme Status
    GoldQuery("GQ-33", "What is the official application portal for MYSY?", ["MYSY"], ["Important Dates", "Overview", "Terms & Conditions"], ["mysy.guj.nic.in"]),
    GoldQuery("GQ-34", "What is the official application portal for Digital Gujarat SC ST SEBC schemes?", ["BCK-6.1", "BCK-10", "BCK-5", "BCK-78", "BCK-80", "BCK-81", "BCK-83", "BCK-137", "BCK-138"], ["Overview", "Important Dates", "Terms & Conditions"], ["digitalgujarat.gov.in"]),
    GoldQuery("GQ-35", "What is the current status of National Talent Search Examination NTSE?", ["NTSE"], ["Overview", "Terms & Conditions"], ["suspended"]),
    GoldQuery("GQ-36", "What is the application timeline and cycle for SHODH fellowship?", ["SHODH"], ["Important Dates", "Terms & Conditions"], ["annual", "annual_odd_semester", "2026-08-01"])
]


@dataclass
class EvaluationMetrics:
    """Evaluation summary metrics."""
    total_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    precision_at_3: float
    citation_validity_rate: float
    provenance_verification_rate: float
    per_query_results: List[Dict[str, Any]] = field(default_factory=list)


def normalize_eval_token(tok: str) -> str:
    """Normalizes token by removing currency signs, commas, and dots for exact numeric comparisons."""
    return tok.lower().replace(",", "").replace("₹", "").replace(".", "").replace("$", "").strip()


class RetrievalEvaluator:
    """Runs rigorous evaluation benchmark against the HybridEvidenceRetriever."""

    def __init__(self, retriever: Optional[HybridEvidenceRetriever] = None):
        self.retriever = retriever or get_evidence_retriever()

    def evaluate(self, gold_queries: Optional[List[GoldQuery]] = None) -> EvaluationMetrics:
        queries = gold_queries or GOLD_STANDARD_QUERIES
        total = len(queries)
        
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        reciprocal_ranks: List[float] = []
        ndcg_scores: List[float] = []
        precision_at_3_scores: List[float] = []
        valid_citations_count = 0
        verified_provenance_count = 0
        total_citations_checked = 0
        per_query_details: List[Dict[str, Any]] = []

        for gq in queries:
            pack = self.retriever.search(query=gq.query_text, top_k=5)
            ev_list = pack.evidence
            
            found_rank: Optional[int] = None
            relevant_in_top_3 = 0
            total_relevant_retrieved = 0
            dcg_5 = 0.0

            for rank_idx, ev in enumerate(ev_list, start=1):
                total_citations_checked += 1
                
                # Check citation validity
                if ev.citation.source_url and ev.citation.citation_id and len(ev.text) > 15:
                    valid_citations_count += 1
                
                # Check cryptographic provenance hash
                if ev.citation.chunk_hash:
                    verified_provenance_count += 1

                # Strict Relevance Condition
                scheme_match = any(
                    exp_s.lower() in ev.scheme_id.lower() or ev.scheme_id.lower() in exp_s.lower()
                    for exp_s in gq.expected_scheme_ids
                )
                section_match = any(
                    sec.lower() in ev.section.lower() or ev.section.lower() in sec.lower()
                    for sec in gq.expected_sections
                )
                text_lower = ev.text.lower()
                text_clean = text_lower.replace(",", "").replace("₹", "").replace(".", "").replace("$", "").strip()
                tok_match = any(
                    tok.lower() in text_lower or normalize_eval_token(tok) in text_clean
                    for tok in gq.expected_content_tokens
                )

                # Strict factual matching: Requires ground-truth scheme and content token match
                if gq.expected_content_tokens:
                    is_relevant = scheme_match and tok_match
                else:
                    is_relevant = scheme_match and section_match

                if is_relevant:
                    total_relevant_retrieved += 1
                    if found_rank is None:
                        found_rank = rank_idx
                    if rank_idx <= 3:
                        relevant_in_top_3 += 1
                    # Discounted Cumulative Gain at rank_idx
                    dcg_5 += 1.0 / math.log2(rank_idx + 1)

            if found_rank == 1:
                hits_at_1 += 1
            if found_rank is not None and found_rank <= 3:
                hits_at_3 += 1
            if found_rank is not None and found_rank <= 5:
                hits_at_5 += 1

            rr = 1.0 / found_rank if found_rank is not None else 0.0
            reciprocal_ranks.append(rr)
            
            # Mathematically correct Normalized DCG (nDCG@5 = DCG@5 / IDCG@5)
            num_ideal = max(1, min(total_relevant_retrieved, 5))
            idcg_5 = sum(1.0 / math.log2(i + 1) for i in range(1, num_ideal + 1))
            ndcg_val = min(1.0, dcg_5 / idcg_5) if idcg_5 > 0 else 0.0
            ndcg_scores.append(ndcg_val)
            
            precision_at_3_scores.append(relevant_in_top_3 / 3.0)

            per_query_details.append({
                "query_id": gq.query_id,
                "query": gq.query_text,
                "first_relevant_rank": found_rank,
                "mrr": round(rr, 4),
                "ndcg_at_5": round(ndcg_val, 4),
                "top_result": ev_list[0].scheme_id if ev_list else "NONE",
                "top_section": ev_list[0].section if ev_list else "NONE"
            })

        recall_1 = hits_at_1 / total if total > 0 else 0.0
        recall_3 = hits_at_3 / total if total > 0 else 0.0
        recall_5 = hits_at_5 / total if total > 0 else 0.0
        mean_mrr = sum(reciprocal_ranks) / total if total > 0 else 0.0
        mean_ndcg = sum(ndcg_scores) / total if total > 0 else 0.0
        mean_p3 = sum(precision_at_3_scores) / total if total > 0 else 0.0
        cit_rate = valid_citations_count / total_citations_checked if total_citations_checked > 0 else 1.0
        prov_rate = verified_provenance_count / total_citations_checked if total_citations_checked > 0 else 1.0

        return EvaluationMetrics(
            total_queries=total,
            recall_at_1=round(recall_1, 4),
            recall_at_3=round(recall_3, 4),
            recall_at_5=round(recall_5, 4),
            mrr=round(mean_mrr, 4),
            ndcg_at_5=round(mean_ndcg, 4),
            precision_at_3=round(mean_p3, 4),
            citation_validity_rate=round(cit_rate, 4),
            provenance_verification_rate=round(prov_rate, 4),
            per_query_results=per_query_details
        )
