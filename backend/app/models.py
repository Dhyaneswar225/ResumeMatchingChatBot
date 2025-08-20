from pydantic import BaseModel
from typing import List

class MatchRequest(BaseModel):
    job_description: str

class MatchResponse(BaseModel):
    similarity_score: float
    matched_keywords: List[str]
    resume_skills: List[str]
    jd_skills: List[str]

class FeedbackRequest(BaseModel):
    removed_skills: List[str]
    suggested_skills: List[str] = []