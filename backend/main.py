import spacy
import fitz  # PyMuPDF
import re
import json
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords as NLTK_STOPWORDS
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOPWORDS
from pydantic import BaseModel
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load custom NER model
MODEL_PATH = "./skill_ner_model"
try:
    nlp = spacy.load(MODEL_PATH)
    logger.info("Loaded custom skill NER model.")
except Exception as e:
    logger.warning(f"Falling back to en_core_web_sm: {str(e)}")
    nlp = spacy.load("en_core_web_sm")

# Load stopwords
try:
    with open("stopwords.json", "r") as f:
        CUSTOM_STOPWORDS = set(json.load(f))
except FileNotFoundError:
    logger.warning("stopwords.json not found; using empty set.")
    CUSTOM_STOPWORDS = set()

ALL_STOPWORDS = SPACY_STOPWORDS.union(set(NLTK_STOPWORDS.words('english'))).union(CUSTOM_STOPWORDS)

# Load rejected skills
try:
    with open("rejected_skills.json", "r") as f:
        REJECTED_SKILLS = set(json.load(f))
except FileNotFoundError:
    logger.warning("rejected_skills.json not found; using empty set.")
    REJECTED_SKILLS = set()

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for response
class MatchResponse(BaseModel):
    resume_sections: Dict[str, str]
    resume_skills: List[str]
    jd_skills: List[str]
    matched_skills: List[str]
    match_percentage: float

# Function to extract text from PDF
def extract_text_from_pdf(pdf_content):
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to extract resume text")

# Function to detect sections in resume text
def detect_sections(text):
    sections = {
        "contact": False,
        "summary": False,
        "experience": False,
        "education": False,
        "skills": False,
        "projects": False,
        "certifications": False,
        "awards": False,
        "publications": False,
        "volunteer": False,
        "languages": False
    }
    
    lines = text.lower().split('\n')
    current_section = None
    content = {k: [] for k in sections}
    
    section_patterns = {
        "contact": r"(contact|info|information|address|phone|email)",
        "summary": r"(summary|objective|profile|about me)",
        "experience": r"(experience|work|employment|professional|history)",
        "education": r"(education|academic|qualifications|degrees)",
        "skills": r"(skills|competencies|abilities|expertise)",
        "projects": r"(projects|portfolio|works)",
        "certifications": r"(certifications|licenses|credentials)",
        "awards": r"(awards|honors|achievements|recognitions)",
        "publications": r"(publications|papers|articles|books)",
        "volunteer": r"(volunteer|community|extracurricular)",
        "languages": r"(languages|linguistic)"
    }
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        for sec, pattern in section_patterns.items():
            if re.search(pattern, line):
                current_section = sec
                sections[sec] = True
                break
                
        if current_section:
            content[current_section].append(line)
    
    for sec in content:
        content[sec] = ' '.join(content[sec])
    
    return content

# Function to extract skills
def extract_skills(text, rejected_skills=REJECTED_SKILLS):
    try:
        sentences = sent_tokenize(text)
        all_skills = set()
        
        for sentence in sentences:
            doc = nlp(sentence)
            for ent in doc.ents:
                if ent.label_ == "SKILL":
                    skill = ent.text.lower().strip()
                    if (skill and len(skill) > 1 and 
                        skill not in ALL_STOPWORDS and 
                        skill not in rejected_skills and 
                        not re.match(r'^\d+$', skill) and 
                        not re.match(r'^[a-z]$', skill)):
                        all_skills.add(skill)
        
        return sorted(list(all_skills))
    except Exception as e:
        logger.error(f"Error extracting skills: {str(e)}")
        return []

@app.post("/match", response_model=MatchResponse)
async def match_resume(resume: UploadFile = File(...), job_description: str = Form(...)):
    try:
        if not resume.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        content = await resume.read()
        resume_text = extract_text_from_pdf(content)
        resume_sections = detect_sections(resume_text)
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(job_description)
        matched_skills = sorted(list(set(resume_skills).intersection(set(jd_skills))))
        match_percentage = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 0
        
        return {
            "resume_sections": resume_sections,
            "resume_skills": resume_skills,
            "jd_skills": jd_skills,
            "matched_skills": matched_skills,
            "match_percentage": round(match_percentage, 2)
        }
    except Exception as e:
        logger.error(f"Error in match_resume: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
