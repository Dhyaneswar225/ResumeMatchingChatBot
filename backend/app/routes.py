from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.nlp import match_resume_with_jd, save_rejected_skills, load_rejected_skills
from app.models import MatchResponse, FeedbackRequest
import io

router = APIRouter()

@router.post("/match", response_model=MatchResponse)
async def match_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    try:
        if not resume.filename.endswith((".pdf", ".txt")):
            raise HTTPException(status_code=400, detail="Only PDF or text files are supported")

        content = await resume.read()
        if resume.filename.endswith(".pdf"):
            result = match_resume_with_jd(io.BytesIO(content), job_description)
        else:
            result = match_resume_with_jd(content.decode("utf-8"), job_description)

        return {
            "similarity_score": result["jd_match_pct"],
            "matched_keywords": result["matched_skills"],
            "resume_skills": result["resume_skills"],
            "jd_skills": result["jd_skills"],
            "ats_score": result["ats_score"],
            "overall_score": result["overall_score"],
            "ats_issues": result["ats_issues"],
            "ats_suggestions": result["ats_suggestions"],
            "grammar_errors": result["grammar_errors"],
            "quantifiable_pct": result["quantifiable_pct"],
            "action_verb_pct": result["action_verb_pct"],
            "repeated_words": result["repeated_words"],
            "buzzwords_found": result["buzzwords_found"],
            "filler_found": result["filler_found"],
            "rewrite_suggestions": result["rewrite_suggestions"],
            "hard_skills_suggestions": result["hard_skills_suggestions"],
            "soft_skills_suggestions": result["soft_skills_suggestions"],
            "sections_detected": result["sections_detected"],
            "suggestions": result["suggestions"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        removed_skills = set(skill.lower().strip() for skill in feedback.removed_skills if skill.strip())
        suggested_skills = set(skill.lower().strip() for skill in feedback.suggested_skills if skill.strip())
        if removed_skills or suggested_skills:
            save_rejected_skills(removed_skills, suggested_skills)
        message = f"Feedback recorded. Removed skills: {', '.join(removed_skills) if removed_skills else 'None'}."
        if suggested_skills:
            message += f" Suggested skills: {', '.join(suggested_skills)} added to suggested skills."
        rejected_skills = load_rejected_skills()
        message += f" Current rejected skills: {', '.join(rejected_skills) if rejected_skills else 'None'}."
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))