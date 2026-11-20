"""
ScholarBridge Hardened Deterministic Rule Matcher (V1.1.1 Correctness Patch)
=============================================================================
Evaluates student profiles against knowledge base eligibility rules with:
1. Strict Benchmark Disability (>=40%) evaluation semantics
2. Full operator & value compliance for all Boolean fields (=, !=, IN, NOT_IN)
3. Strict normalized alias matching for institutions, entrance exams, and courses
4. Centralized deterministic operator engine (=, !=, <, <=, >, >=, IN, NOT_IN, BETWEEN)
5. Canonical stream and course clustering
6. 3-state deterministic outcome: ELIGIBLE, CONDITIONAL, INELIGIBLE, and SUSPENDED.
"""

from typing import List, Dict, Tuple, Any, Set, Optional
from .models import (
    StudentProfile, Scholarship, EligibilityRule, RuleEvaluation,
    MatchResult, EligibilityStatus
)


# Explicit set of all supported machine rule fields in the engine
SUPPORTED_RULE_FIELDS: Set[str] = {
    "annual_family_income",
    "social_category",
    "gender",
    "domicile_state",
    "study_state",
    "study_institution",
    "education_level",
    "stream",
    "course_name",
    "marks_percentage",
    "marks_percentile",
    "entrance_exam_qualified",
    "hostel_status",
    "is_pwd",
    "parent_occupation",
    "is_construction_worker_ward",
    "is_defence_ward",
    "is_orphan_or_martyr_ward",
    "is_sports_medalist",
    "age"
}

# Explicit set of all supported mathematical / comparison operators
SUPPORTED_OPERATORS: Set[str] = {
    "=", "==", "!=", "<", "<=", ">", ">=", "IN", "NOT_IN", "BETWEEN"
}


# Canonical Stream Aliases (Controlled Normalization)
STREAM_CLUSTERS: Dict[str, Set[str]] = {
    "ENGINEERING": {
        "ENGINEERING", "B.TECH", "B.E.", "BTECH", "BE", "DIPLOMA ENGINEERING",
        "POLYTECHNIC", "CIVIL", "MECHANICAL", "ELECTRICAL", "ELECTRONICS",
        "CHEMICAL", "COMPUTER ENGINEERING", "AERONAUTICAL", "AUTOMOBILE"
    },
    "TECHNICAL": {
        "TECHNICAL", "ENGINEERING", "TECHNOLOGY", "POLYTECHNIC", "PHARMACY",
        "ARCHITECTURE", "COMPUTER SCIENCE", "IT", "INFORMATION TECHNOLOGY",
        "DATA SCIENCE", "AI", "ARTIFICIAL INTELLIGENCE", "CIVIL", "MECHANICAL",
        "ELECTRICAL", "ELECTRONICS", "CHEMICAL", "B.TECH", "B.E.", "DIPLOMA", "BCA", "MCA"
    },
    "MEDICAL": {
        "MEDICAL", "MBBS", "BDS", "BAMS", "BHMS", "NURSING", "PHYSIOTHERAPY",
        "DENTAL", "AYUSH", "PHARMA", "PHARMACY", "ALLOPATHY", "HOMOEOPATHY", "AYURVEDA"
    },
    "SCIENCE": {
        "SCIENCE", "PHYSICS", "CHEMISTRY", "MATHEMATICS", "BIOLOGY",
        "BIOTECHNOLOGY", "BOTANY", "ZOOLOGY", "MICROBIOLOGY", "BIOCHEMISTRY",
        "BSC", "B.SC", "MSC", "M.SC", "NATURAL SCIENCES", "BASIC SCIENCES"
    },
    "COMMERCE": {
        "COMMERCE", "ACCOUNTING", "FINANCE", "BANKING", "BUSINESS", "BBA",
        "BCOM", "B.COM", "MCOM", "M.COM", "CA", "CS", "CMA"
    },
    "ARTS": {
        "ARTS", "HUMANITIES", "SOCIAL SCIENCES", "LITERATURE", "HISTORY",
        "ECONOMICS", "SOCIOLOGY", "POLITICAL SCIENCE", "PSYCHOLOGY", "BA", "B.A.", "MA", "M.A."
    },
    "LAW": {
        "LAW", "LEGAL", "LL.B", "LLB", "BALLB", "B.A. LL.B", "BBALLB", "B.B.A. LL.B", "LLM", "LL.M"
    },
    "AGRICULTURE": {
        "AGRICULTURE", "AGRI", "HORTICULTURE", "FORESTRY", "VETERINARY",
        "DAIRY", "FISHERIES", "AGRIBUSINESS", "B.SC AGRI", "BSC AGRICULTURE"
    },
    "MANAGEMENT": {
        "MANAGEMENT", "MBA", "BBA", "HOTEL MANAGEMENT", "POSTGRADUATE DIPLOMA IN MANAGEMENT", "PGDM"
    }
}

# Canonical Course Aliases
COURSE_CLUSTERS: Dict[str, Set[str]] = {
    "B.TECH": {"B.TECH", "BTECH", "B.E.", "BE", "BACHELOR OF TECHNOLOGY", "BACHELOR OF ENGINEERING", "B TECH", "B E"},
    "M.TECH": {"M.TECH", "MTECH", "M.E.", "ME", "MASTER OF TECHNOLOGY", "MASTER OF ENGINEERING", "M TECH", "M E"},
    "MBBS": {"MBBS", "BACHELOR OF MEDICINE", "BACHELOR OF SURGERY"},
    "BDS": {"BDS", "BACHELOR OF DENTAL SURGERY"},
    "BCA": {"BCA", "BACHELOR OF COMPUTER APPLICATIONS"},
    "MCA": {"MCA", "MASTER OF COMPUTER APPLICATIONS"},
    "B.SC": {"B.SC", "BSC", "BACHELOR OF SCIENCE", "BS"},
    "M.SC": {"M.SC", "MSC", "MASTER OF SCIENCE", "MS"},
    "B.COM": {"B.COM", "BCOM", "BACHELOR OF COMMERCE"},
    "M.COM": {"M.COM", "MCOM", "MASTER OF COMMERCE"},
    "B.A.": {"B.A.", "BA", "BACHELOR OF ARTS"},
    "M.A.": {"M.A.", "MA", "MASTER OF ARTS"},
    "DIPLOMA": {"DIPLOMA", "POLYTECHNIC", "DIPLOMA ENGINEERING", "D2D"},
    "ITI": {"ITI", "INDUSTRIAL TRAINING", "CERTIFICATE"},
    "PHD": {"PHD", "PH.D", "DOCTOR OF PHILOSOPHY", "DOCTORAL RESEARCH"}
}

# Canonical Institution Aliases
INSTITUTION_CLUSTERS: Dict[str, Set[str]] = {
    "IITGN": {"IIT GANDHINAGAR", "IITGN", "INDIAN INSTITUTE OF TECHNOLOGY GANDHINAGAR", "IIT G"},
    "IIMA": {"IIM AHMEDABAD", "IIMA", "INDIAN INSTITUTE OF MANAGEMENT AHMEDABAD"},
    "SVNIT": {"SVNIT", "SVNIT SURAT", "SARDAR VALLABHBHAI NATIONAL INSTITUTE OF TECHNOLOGY"},
    "MSU": {"MSU", "MSU BARODA", "MAHARAJA SAYAJIRAO UNIVERSITY", "MAHARAJA SAYAJIRAO UNIVERSITY OF BARODA"},
    "GTU": {"GTU", "GUJARAT TECHNOLOGICAL UNIVERSITY"},
    "GU": {"GUJARAT UNIVERSITY", "GU"},
    "PDEU": {"PDEU", "PDPU", "PANDIT DEENDAYAL ENERGY UNIVERSITY", "PANDIT DEENDAYAL PETROLEUM UNIVERSITY"},
    "DAIICT": {"DAIICT", "DA-IICT", "DHIRUBHAI AMBANI INSTITUTE OF INFORMATION AND COMMUNICATION TECHNOLOGY"},
    "GNLU": {"GNLU", "GUJARAT NATIONAL LAW UNIVERSITY"},
    "NIRMA": {"NIRMA", "NIRMA UNIVERSITY"},
    "AGRI_UNIV": {"AAU", "JAU", "NAU", "SDAU", "ANAND AGRICULTURAL UNIVERSITY", "JUNAGADH AGRICULTURAL UNIVERSITY", "NAVSARI AGRICULTURAL UNIVERSITY", "SARDARKRUSHINAGAR DANTIWADA AGRICULTURAL UNIVERSITY", "STATE AGRICULTURAL UNIVERSITIES"}
}

# Canonical Entrance Exam Aliases
ENTRANCE_EXAM_CLUSTERS: Dict[str, Set[str]] = {
    "GATE": {"GATE", "GRADUATE APTITUDE TEST IN ENGINEERING"},
    "GPAT": {"GPAT", "GRADUATE PHARMACY APTITUDE TEST"},
    "CAT": {"CAT", "COMMON ADMISSION TEST"},
    "CLAT": {"CLAT", "COMMON LAW ADMISSION TEST"},
    "NEET": {"NEET", "NEET UG", "NEET PG"},
    "JEE": {"JEE", "JEE MAIN", "JEE ADVANCED"},
    "UGC_NET": {"UGC-NET", "UGC NET", "UGC JRF", "NET-JRF"},
    "CSIR_NET": {"CSIR-NET", "CSIR NET", "CSIR JRF"},
    "NMMS": {"NMMS", "NATIONAL MEANS CUM MERIT SCHOLARSHIP EXAM"},
    "YASASVI": {"YET", "PM-YASASVI ENTRANCE TEST", "YASASVI"}
}


def normalize_clean_token(s: str) -> str:
    """Strips punctuation and whitespace for clean token matching."""
    return "".join(c for c in str(s).upper() if c.isalnum() or c in [" ", "_", "-", "/"])


def evaluate_numeric_operator(actual: float, op: str, expected_str: str) -> Tuple[bool, str]:
    """Deterministically evaluates numeric comparison operators."""
    op = op.strip().upper()
    if op == "BETWEEN":
        parts = [float(p.strip()) for p in expected_str.replace(":", "-").split("-") if p.strip()]
        if len(parts) == 2:
            min_v, max_v = min(parts), max(parts)
            passed = min_v <= actual <= max_v
            reason = f"Value {actual} is between {min_v} and {max_v}" if passed else f"Value {actual} is outside range [{min_v}, {max_v}]"
            return passed, reason
        return False, f"Invalid BETWEEN range format: {expected_str}"
        
    target = float(expected_str)
    if op in ["=", "=="]:
        passed = (actual == target)
        reason = f"Value {actual} equals required {target}" if passed else f"Value {actual} does not equal {target}"
    elif op == "!=":
        passed = (actual != target)
        reason = f"Value {actual} != {target}" if passed else f"Value cannot be {target}"
    elif op == "<":
        passed = (actual < target)
        reason = f"Value {actual} < {target}" if passed else f"Value {actual} is not strictly less than {target}"
    elif op == "<=":
        passed = (actual <= target)
        reason = f"Value {actual} <= {target}" if passed else f"Value {actual} exceeds maximum limit of {target}"
    elif op == ">":
        passed = (actual > target)
        reason = f"Value {actual} > {target}" if passed else f"Value {actual} is not strictly greater than floor of {target}"
    elif op == ">=":
        passed = (actual >= target)
        reason = f"Value {actual} >= {target}" if passed else f"Value {actual} is below minimum requirement of {target}"
    else:
        return False, f"Unsupported numeric operator: {op}"
        
    return passed, reason


def evaluate_set_operator(actual: str, op: str, expected_str: str) -> Tuple[bool, str]:
    """Deterministically evaluates set/string membership operators."""
    op = op.strip().upper()
    target_set = {normalize_clean_token(item) for item in expected_str.replace(", ", "|").replace("/", "|").split("|") if item.strip()}
    clean_actual = normalize_clean_token(actual)
    
    if "ALL" in target_set or "ANY" in target_set:
        return True, "Open to all values"
        
    if op in ["=", "==", "IN"]:
        passed = clean_actual in target_set
        reason = f"Value '{actual}' is eligible" if passed else f"Value '{actual}' not in required options ({expected_str})"
        return passed, reason
    elif op in ["!=", "NOT_IN"]:
        passed = clean_actual not in target_set
        reason = f"Value '{actual}' is not excluded" if passed else f"Value '{actual}' is explicitly excluded"
        return passed, reason
    else:
        return False, f"Unsupported set operator: {op}"


def evaluate_boolean_operator(actual_bool: bool, op: str, expected_str: str) -> Tuple[bool, str]:
    """Deterministically evaluates boolean fields with full operator support (=, !=, IN, NOT_IN)."""
    op = op.strip().upper()
    norm_expected = expected_str.strip().upper()
    
    expected_bool = norm_expected in ["YES", "TRUE", "1", "Y"]
    
    if op in ["=", "==", "IN"]:
        passed = (actual_bool == expected_bool)
        reason = f"Condition satisfied ({'Yes' if actual_bool else 'No'})" if passed else f"Condition requires {'Yes' if expected_bool else 'No'} (Actual: {'Yes' if actual_bool else 'No'})"
        return passed, reason
    elif op in ["!=", "NOT_IN"]:
        passed = (actual_bool != expected_bool)
        reason = f"Condition satisfied (not {'Yes' if expected_bool else 'No'})" if passed else f"Condition requires value != {'Yes' if expected_bool else 'No'}"
        return passed, reason
    else:
        return False, f"Unsupported boolean operator: {op}"


class RuleMatcher:
    """Evaluates rules against a student profile with full explainability."""

    @staticmethod
    def evaluate_atomic_rule(rule: EligibilityRule, profile: StudentProfile) -> RuleEvaluation:
        field = rule.field.strip()
        op = rule.operator.strip()
        val = rule.value.strip()
        
        # 0. Fail Loudly on Unmapped / Unknown Rule Fields
        if field not in SUPPORTED_RULE_FIELDS:
            return RuleEvaluation(
                rule=rule,
                passed=False,
                is_conditional=False,
                reason=f"CONFIG_ERROR: Rule field '{field}' is unmapped in the deterministic engine. Field must be added to SUPPORTED_RULE_FIELDS and handled explicitly.",
                actual_value=None,
                expected_value=val
            )

        # 1. Financial: Annual Family Income
        if field == "annual_family_income":
            actual_val = profile.annual_family_income
            passed, reason = evaluate_numeric_operator(actual_val, op, val)
            if not passed and op == ">":
                reason += " (covered under BCK-6.1 / Post-Matric scheme)"
            return RuleEvaluation(rule, passed, False, reason, actual_val, val)

        # 2. Demographics: Social Category
        if field == "social_category":
            actual_cat = profile.social_category
            passed, reason = evaluate_set_operator(actual_cat, op, val)
            # Special case for EWS priority under General
            if not passed and actual_cat.upper() == "EWS" and "GENERAL, EWS" in val.upper():
                passed = True
                reason = "EWS category eligible under General/EWS combined quota"
            return RuleEvaluation(rule, passed, False, reason, actual_cat, val)

        # 3. Demographics: Gender
        if field == "gender":
            actual_gen = profile.gender
            passed, reason = evaluate_set_operator(actual_gen, op, val)
            if not passed:
                reason = f"Scheme is restricted to {val} students only (Student is {profile.gender})"
            return RuleEvaluation(rule, passed, False, reason, profile.gender, val)

        # 4. Geography: Domicile State
        if field == "domicile_state":
            actual_dom = profile.domicile_state
            passed, reason = evaluate_set_operator(actual_dom, op, val)
            if not passed:
                reason = f"Requires {val} domicile (Student domicile is {actual_dom})"
            return RuleEvaluation(rule, passed, False, reason, actual_dom, val)

        # 5. Geography: Study State
        if field == "study_state":
            actual_study = profile.study_state
            passed, reason = evaluate_set_operator(actual_study, op, val)
            if not passed:
                reason = f"Requires studying in {val} (Student is studying in {actual_study})"
            return RuleEvaluation(rule, passed, False, reason, actual_study, val)

        # 6. Geography: Study Institution (Strict Cluster & Alias Matching)
        if field == "study_institution":
            if profile.institution_name:
                inst_actual = normalize_clean_token(profile.institution_name)
                target_insts = [normalize_clean_token(i) for i in val.replace(", ", "|").split("|") if i.strip()]
                
                # Direct match
                passed = inst_actual in target_insts
                
                # Canonical Institution Cluster lookup
                if not passed:
                    for target_token in target_insts:
                        for cluster_key, aliases in INSTITUTION_CLUSTERS.items():
                            clean_aliases = {normalize_clean_token(a) for a in aliases}
                            if target_token in clean_aliases and inst_actual in clean_aliases:
                                passed = True
                                break
                        if passed:
                            break
                            
                reason = f"Enrolled in eligible institution: {profile.institution_name}" if passed else f"Requires enrollment in {val} (Student is at {profile.institution_name})"
                return RuleEvaluation(rule, passed, False, reason, profile.institution_name, val)
            else:
                return RuleEvaluation(rule, False, True, f"Conditionally eligible — requires enrollment in {val}", None, val)

        # 7. Academics: Education Level
        if field == "education_level":
            actual_edu = profile.education_level
            passed, reason = evaluate_set_operator(actual_edu, op, val)
            return RuleEvaluation(rule, passed, False, reason, actual_edu, val)

        # 8. Academics: Stream (Canonical Cluster & Alias Matching)
        if field == "stream":
            actual_stream = normalize_clean_token(profile.stream)
            target_stream_tokens = [normalize_clean_token(s) for s in val.replace(", ", "|").replace("/", "|").split("|") if s.strip()]
            
            if "ALL" in target_stream_tokens or actual_stream == "ALL":
                return RuleEvaluation(rule, True, False, "Open to all academic streams", profile.stream, val)
                
            # Direct match
            passed = actual_stream in target_stream_tokens
            
            # Canonical Cluster Check
            if not passed:
                for target_token in target_stream_tokens:
                    if target_token in STREAM_CLUSTERS and actual_stream in STREAM_CLUSTERS[target_token]:
                        passed = True
                        break
                    for cluster_name, aliases in STREAM_CLUSTERS.items():
                        if cluster_name not in ["TECHNICAL"]:
                            if target_token in aliases and actual_stream in aliases:
                                passed = True
                                break
                    if passed:
                        break
                        
            reason = f"Academic stream '{profile.stream}' is eligible" if passed else f"Scheme restricted to {val} stream (Student is in {profile.stream})"
            return RuleEvaluation(rule, passed, False, reason, profile.stream, val)

        # 9. Academics: Course Name (Canonical Normalization & Aliases)
        if field == "course_name":
            if profile.course_name:
                actual_course = normalize_clean_token(profile.course_name)
                target_courses = [normalize_clean_token(c) for c in val.replace(", ", "|").replace("/", "|").split("|") if c.strip()]
                
                if "ALL" in target_courses or actual_course == "ALL":
                    return RuleEvaluation(rule, True, False, "Open to all courses", profile.course_name, val)
                    
                passed = actual_course in target_courses
                
                # Check Course Clusters (supports degree prefix e.g. B.Tech Computer Science -> B.Tech)
                if not passed:
                    for target_c in target_courses:
                        for cluster_key, aliases in COURSE_CLUSTERS.items():
                            clean_aliases = {normalize_clean_token(a) for a in aliases}
                            if target_c in clean_aliases:
                                if any(actual_course.startswith(a) or a in actual_course.split() or actual_course in a for a in clean_aliases):
                                    passed = True
                                    break
                        if passed:
                            break
                            
                reason = f"Enrolled course '{profile.course_name}' is eligible" if passed else f"Scheme restricted to {val} course (Student is in {profile.course_name})"
                return RuleEvaluation(rule, passed, False, reason, profile.course_name, val)
            else:
                return RuleEvaluation(rule, False, True, f"Conditionally eligible — requires enrollment in {val} course", None, val)

        # 10. Academics: Marks Percentage / Percentile
        if field in ["marks_percentage", "marks_percentile"]:
            is_d2d_rule = "d2d" in rule.rule_description.lower()
            
            # D2D-specific rule does not apply to non-D2D regular admissions
            if is_d2d_rule and not profile.is_d2d_admission:
                return RuleEvaluation(rule, False, False, "D2D criterion applies only to Diploma-to-Degree lateral admissions", None, val)
                
            # Direct 10th/12th board percentile rule does not apply to D2D admissions
            if not is_d2d_rule and profile.is_d2d_admission and field == "marks_percentile":
                return RuleEvaluation(rule, False, False, "Board percentile criterion applies to direct admissions, not D2D", None, val)

            actual_marks = profile.marks_percentile if field == "marks_percentile" else profile.marks_percentage
            if actual_marks is None:
                return RuleEvaluation(rule, False, True, f"Conditionally eligible — requires {field.replace('_', ' ')} to satisfy {op} {val}", None, val)
            passed, reason = evaluate_numeric_operator(actual_marks, op, val)
            return RuleEvaluation(rule, passed, False, reason, actual_marks, val)

        # 11. Academics: Entrance Exam Qualification (Strict Cluster Lookup)
        if field == "entrance_exam_qualified":
            if profile.entrance_exam_qualified:
                actual_exam = normalize_clean_token(profile.entrance_exam_qualified)
                target_exams = [normalize_clean_token(e) for e in val.replace(", ", "|").split("|") if e.strip()]
                
                passed = actual_exam in target_exams
                if not passed:
                    for target_e in target_exams:
                        for cluster_key, aliases in ENTRANCE_EXAM_CLUSTERS.items():
                            clean_aliases = {normalize_clean_token(a) for a in aliases}
                            if target_e in clean_aliases and actual_exam in clean_aliases:
                                passed = True
                                break
                        if passed:
                            break
                            
                reason = f"Cleared required entrance: {profile.entrance_exam_qualified}" if passed else f"Requires {val} qualification (Student cleared {profile.entrance_exam_qualified})"
                return RuleEvaluation(rule, passed, False, reason, profile.entrance_exam_qualified, val)
            else:
                return RuleEvaluation(rule, False, True, f"Conditionally eligible — requires qualification in {val}", None, val)

        # 12. Special Context: Hostel Status
        if field == "hostel_status":
            if profile.hostel_status is not None:
                actual_h = normalize_clean_token(profile.hostel_status)
                target_h = [normalize_clean_token(h) for h in val.replace(", ", "|").split("|") if h.strip()]
                passed = (actual_h in target_h) or (actual_h == "COLLEGE_AFFILIATED_HOSTEL" and "HOSTELLER" in target_h)
                reason = "Residing in recognized college hostel" if passed else f"Scheme is strictly for hosteller students ({val})"
                return RuleEvaluation(rule, passed, False, reason, profile.hostel_status, val)
            else:
                return RuleEvaluation(rule, False, True, "Conditionally eligible — requires college-affiliated hostel accommodation", None, val)

        # 13. Special Context: Person with Disability (Strict Benchmark Disability >= 40%)
        if field == "is_pwd":
            req_yes = val.strip().upper() in ["YES", "TRUE", "1", "Y"]
            if req_yes and op in ["=", "==", "IN"]:
                if not profile.is_pwd:
                    return RuleEvaluation(rule, False, False, "Scheme requires benchmark disability (>=40%)", "No", val)
                
                # Strict check on disability percentage
                if profile.disability_percentage > 0:
                    if profile.disability_percentage < 40.0:
                        return RuleEvaluation(
                            rule, False, False,
                            f"Disability percentage {profile.disability_percentage}% is below the statutory benchmark cutoff of 40%",
                            f"{profile.disability_percentage}%", ">=40%"
                        )
                    else:
                        return RuleEvaluation(
                            rule, True, False,
                            f"Statutory benchmark disability verified ({profile.disability_percentage}%)",
                            f"{profile.disability_percentage}%", val
                        )
                else:
                    # Unstated percentage: if UDID card present, pass; otherwise conditional
                    if profile.has_udid_card:
                        return RuleEvaluation(rule, True, False, "Benchmark disability verified via valid UDID card", "UDID Card", val)
                    return RuleEvaluation(rule, False, True, "Conditionally eligible — disability percentage must be verified >= 40% (UDID card required)", None, val)
            else:
                passed, reason = evaluate_boolean_operator(profile.is_pwd, op, val)
                return RuleEvaluation(rule, passed, False, reason, profile.is_pwd, val)

        # 14. Special Context: Construction Worker Ward
        if field == "is_construction_worker_ward":
            actual_bool = profile.is_construction_worker_ward or (bool(profile.parent_occupation) and "CONSTRUCTION" in profile.parent_occupation.upper())
            req_yes = val.strip().upper() in ["YES", "TRUE", "1", "Y"]
            
            if req_yes and op in ["=", "==", "IN"] and not actual_bool and profile.parent_occupation is None:
                # Unstated parent occupation -> Conditional
                return RuleEvaluation(rule, False, True, "Conditionally eligible — requires parent registered with BOCWWB / e-Nirman card", None, val)
                
            passed, reason = evaluate_boolean_operator(actual_bool, op, val)
            return RuleEvaluation(rule, passed, False, reason, "Yes" if actual_bool else "No", val)

        # 15. Special Context: Parent Occupation
        if field == "parent_occupation":
            if profile.parent_occupation:
                actual_occ = profile.parent_occupation.upper()
                if "CONSTRUCTION" in val.upper() and ("CONSTRUCTION" in actual_occ or profile.is_construction_worker_ward):
                    return RuleEvaluation(rule, True, False, "Parent construction worker verified", profile.parent_occupation, val)
                elif "BEEDI" in val.upper() and "BEEDI" in actual_occ:
                    return RuleEvaluation(rule, True, False, "Parent beedi/mine worker verified", profile.parent_occupation, val)
                passed, reason = evaluate_set_operator(actual_occ, op, val)
                return RuleEvaluation(rule, passed, False, reason, profile.parent_occupation, val)
            else:
                return RuleEvaluation(rule, False, True, f"Conditionally eligible — requires parent occupation: {val}", None, val)

        # 16. Special Context: Defence / Ex-Servicemen Ward
        if field == "is_defence_ward":
            actual_bool = profile.is_defence_ward or (bool(profile.parent_occupation) and any(o in profile.parent_occupation.upper() for o in ["DEFENCE", "EX-SERVICEMEN", "ESM", "CAPF", "ARMY", "NAVY", "AIR FORCE"]))
            req_yes = val.strip().upper() in ["YES", "TRUE", "1", "Y"]
            
            if req_yes and op in ["=", "==", "IN"] and not actual_bool and profile.parent_occupation is None:
                return RuleEvaluation(rule, False, True, "Conditionally eligible — requires ward of Ex-Servicemen or CAPF personnel", None, val)
                
            passed, reason = evaluate_boolean_operator(actual_bool, op, val)
            return RuleEvaluation(rule, passed, False, reason, "Yes" if actual_bool else "No", val)

        # 17. Special Context: Orphan / Martyr Ward
        if field == "is_orphan_or_martyr_ward":
            actual_bool = profile.is_orphan_or_martyr_ward
            req_yes = val.strip().upper() in ["YES", "TRUE", "1", "Y"]
            
            if req_yes and not actual_bool:
                return RuleEvaluation(rule, False, False, "Scheme requires orphan, martyr child, or single-parent loss status", "No", val)
                
            passed, reason = evaluate_boolean_operator(actual_bool, op, val)
            return RuleEvaluation(rule, passed, False, reason, "Yes" if actual_bool else "No", val)

        # 18. Special Context: Sports Medalist
        if field == "is_sports_medalist":
            actual_bool = profile.is_sports_medalist
            req_yes = val.strip().upper() in ["YES", "TRUE", "1", "Y"]
            
            if req_yes and not actual_bool:
                return RuleEvaluation(rule, False, False, "Scheme requires state/national sports medal achievement", "No", val)
                
            passed, reason = evaluate_boolean_operator(actual_bool, op, val)
            return RuleEvaluation(rule, passed, False, reason, "Yes" if actual_bool else "No", val)

        # 19. Demographics: Age Limit
        if field == "age":
            if profile.age is not None:
                passed, reason = evaluate_numeric_operator(float(profile.age), op, val)
                return RuleEvaluation(rule, passed, False, reason, profile.age, val)
            else:
                return RuleEvaluation(rule, False, True, f"Conditionally eligible — age must satisfy {op} {val}", None, val)

        # Fallback (never reached due to initial SUPPORTED_RULE_FIELDS check)
        return RuleEvaluation(rule, False, False, f"UNHANDLED_RULE_FIELD: {field}", None, val)

    def evaluate_scholarship(
        self, scholarship: Scholarship, rules: List[EligibilityRule], profile: StudentProfile
    ) -> MatchResult:
        """Evaluates one scholarship's rules against a student profile."""
        
        # Immediate check for Suspended / Historical schemes (e.g. NTSE)
        if scholarship.status == "Suspended":
            return MatchResult(
                scholarship=scholarship,
                status=EligibilityStatus.SUSPENDED,
                explanation="NOTICE: This scheme is currently suspended nationwide by NCERT/MoE and is not accepting applications for 2026-27.",
                exclusion_reason="Scheme is suspended nationwide"
            )

        passed_evals: List[RuleEvaluation] = []
        failed_evals: List[RuleEvaluation] = []
        conditional_evals: List[RuleEvaluation] = []

        # Group rules by Condition_Group (e.g., G1 = AND, G_ACAD = OR for MYSY)
        groups: Dict[str, List[EligibilityRule]] = {}
        for r in rules:
            groups.setdefault(r.condition_group, []).append(r)

        for grp_name, grp_rules in groups.items():
            # Check if this group uses OR logic (e.g. G_ACAD in MYSY for 10th vs 12th vs D2D)
            is_or_group = any(r.logical_operator == "OR" for r in grp_rules) or grp_name.startswith("G_OR") or grp_name == "G_ACAD"
            
            grp_evals = [self.evaluate_atomic_rule(r, profile) for r in grp_rules]
            
            if is_or_group:
                # Any passed rule in OR group satisfies the whole group!
                if any(ev.passed for ev in grp_evals):
                    passed_evals.extend([ev for ev in grp_evals if ev.passed])
                elif any(ev.is_conditional for ev in grp_evals):
                    conditional_evals.extend([ev for ev in grp_evals if ev.is_conditional])
                else:
                    failed_evals.extend(grp_evals)
            else:
                # AND group: every rule must pass
                for ev in grp_evals:
                    if ev.passed:
                        passed_evals.append(ev)
                    elif ev.is_conditional:
                        conditional_evals.append(ev)
                    else:
                        failed_evals.append(ev)

        # Determine Final Eligibility Status
        if failed_evals:
            status = EligibilityStatus.INELIGIBLE
            exclusion_reason = "; ".join(ev.reason for ev in failed_evals[:3])
            explanation = scholarship.ineligibility_exclusion_template or f"Ineligible: {exclusion_reason}"
        elif conditional_evals:
            status = EligibilityStatus.CONDITIONAL
            cond_reason = "; ".join(ev.reason for ev in conditional_evals[:2])
            explanation = f"Conditionally Eligible: {cond_reason}"
            exclusion_reason = ""
        else:
            status = EligibilityStatus.ELIGIBLE
            explanation = scholarship.eligibility_explanation_template or "All eligibility criteria satisfied."
            exclusion_reason = ""

        return MatchResult(
            scholarship=scholarship,
            status=status,
            passed_rules=passed_evals,
            failed_rules=failed_evals,
            conditional_rules=conditional_evals,
            explanation=explanation,
            exclusion_reason=exclusion_reason
        )
