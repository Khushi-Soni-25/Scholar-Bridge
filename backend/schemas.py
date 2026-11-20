from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator

class AuthRegister(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: EmailStr
    password: str = Field(min_length=6, max_length=100)

    @field_validator("password")
    @classmethod
    def password_must_fit_bcrypt(cls, value: str) -> str:
        # bcrypt supports at most 72 bytes, not necessarily 72 Unicode characters.
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be at most 72 bytes when encoded as UTF-8")
        return value

class AuthLogin(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_must_fit_bcrypt(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be at most 72 bytes when encoded as UTF-8")
        return value

class Profile(BaseModel):
    social_category: str
    gender: str
    annual_family_income: float = Field(gt=0)
    domicile_state: str = "Gujarat"
    study_state: str = "Gujarat"
    district: Optional[str] = None
    education_level: str = "Undergraduate"
    stream: str = "All"
    course_name: Optional[str] = None
    institution_name: Optional[str] = None
    marks_percentage: Optional[float] = Field(None, ge=0, le=100)
    marks_percentile: Optional[float] = Field(None, ge=0, le=100)
    is_d2d_admission: bool = False
    entrance_exam_qualified: Optional[str] = None
    is_pwd: bool = False
    disability_percentage: float = Field(0, ge=0, le=100)
    has_udid_card: bool = False
    hostel_status: Optional[str] = None
    parent_occupation: Optional[str] = None
    is_construction_worker_ward: bool = False
    is_defence_ward: bool = False
    is_orphan_or_martyr_ward: bool = False
    is_sports_medalist: bool = False
    age: Optional[int] = Field(None, ge=0, le=80)

class RecommendationRequest(BaseModel):
    profile: Profile

class SearchRequest(BaseModel):
    query: str = Field(min_length=3)
    top_k: int = Field(5, ge=1, le=20)
    scheme_id: Optional[str] = None

class ExplainRequest(BaseModel):
    profile: Profile
    scholarship_id_or_code: str
    user_query: Optional[str] = None

class SaveRequest(BaseModel):
    scholarship_id: str
