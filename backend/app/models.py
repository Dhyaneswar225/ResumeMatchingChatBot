from pydantic import BaseModel
from typing import List, Dict

class MatchRequest(BaseModel):
    job_description: str

class MatchResponse(BaseModel):
    similarity_score: float
    matched_keywords: List[str]
    resume_skills: List[str]
    jd_skills: List[str]
    ats_score: float
    overall_score: float
    ats_issues: List[str]
    ats_suggestions: List[str]
    grammar_errors: List[Dict]
    quantifiable_pct: float
    action_verb_pct: float
    repeated_words: Dict[str, int]
    buzzwords_found: List[str]
    filler_found: List[str]
    rewrite_suggestions: List[str]
    hard_skills_suggestions: List[str]
    soft_skills_suggestions: List[str]
    sections_detected: List[str]
    suggestions: List[str]

class FeedbackRequest(BaseModel):
    removed_skills: List[str]
    suggested_skills: List[str] = []