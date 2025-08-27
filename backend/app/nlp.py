import spacy
import fitz  # PyMuPDF
import io
import re
import string
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from PyPDF2 import PdfReader
import nltk
import dateparser
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOPWORDS
from nltk.corpus import stopwords as NLTK_STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import language_tool_python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import os
import shutil
import transformers
from sklearn.metrics.pairwise import cosine_similarity
import sys
import unicodedata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress Transformers warnings for invalid flags
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Log transformers version
logger.info(f"Using transformers version: {transformers.__version__}")

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    logger.info("NLTK resources downloaded successfully.")
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {str(e)}")

# Initialize models
MODEL_PATH = "./skill_ner_model"
try:
    nlp = spacy.load(MODEL_PATH)
    logger.info("Loaded custom skill NER model.")
except Exception as e:
    logger.warning(f"Falling back to en_core_web_sm: {str(e)}")
    nlp = spacy.load("en_core_web_sm")

nlp_loc = spacy.load("en_core_web_sm")

try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Loaded sentence-transformer model.")
except Exception as e:
    logger.error(f"Could not load sentence-transformer model: {str(e)}")
    embedder = None

try:
    grammar_tool = language_tool_python.LanguageTool('en-US')
    logger.info("Initialized grammar tool.")
except Exception as e:
    logger.warning(f"Could not initialize grammar tool: {str(e)}")
    grammar_tool = None

# Initialize Gemma-3 model and tokenizer
try:
    logger.info("Loading tokenizer for google/gemma-3-1b-it")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    logger.info("Loading model google/gemma-3-1b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.warning(f"Could not load Gemma-3 model, falling back to Gemma-2: {str(e)}")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()
    logger.info("Fallback Gemma-2 model loaded.")

# Load stopwords
try:
    with open("stopwords.json", "r", encoding="utf-8") as f:
        custom_stopwords = set(json.load(f))
except Exception as e:
    logger.warning(f"Could not load stopwords.json: {str(e)}")
    custom_stopwords = set()

try:
    with open("rejected_skills.json", "r", encoding="utf-8") as f:
        rejected_skills = set(json.load(f))
except Exception as e:
    logger.warning(f"Could not load rejected_skills.json: {str(e)}")
    rejected_skills = set()

# Utility functions
def extract_text_from_pdf(pdf_content):
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        logger.info(f"Extracted full PDF text: {text[:100]}...")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {str(e)}")
        return ""

def normalize_text(text):
    """Normalize text by removing special characters and standardizing encoding."""
    if not text:
        return ""
    # Convert to ASCII, replacing special chars (e.g., "S¨ udwestfalen" → "Sudwestfalen")
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s-]', ' ', text)  # Remove non-word chars except hyphen
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def extract_sections_from_pdf(pdf_content):
    """
    Extracts resume text into sections based on bold headers or text patterns.
    """
    expected_sections = [
        "Professional Summary", "Technical Skills", "Education", "Professional Experience",
        "Projects", "Certifications & Training", "Achievements & Key Strengths", "Languages"
    ]
    upper_expected = [s.upper() for s in expected_sections]
    
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        sections = {}
        current_section = None
        full_text = ""
        
        for page in doc:
            dict_page = page.get_text("dict")
            full_text += page.get_text("text") + "\n"
            for block in dict_page.get("blocks", []):
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    line_text = ""
                    is_bold = False
                    for span in line["spans"]:
                        line_text += span["text"]
                        if span["flags"] & 2:  # Bit 1 (value 2) indicates bold
                            is_bold = True
                    line_text = normalize_text(line_text.strip())
                    logger.debug(f"Processing line: {line_text} (bold: {is_bold})")
                    # Check for section headers (bold or partial match, case-insensitive)
                    if line_text:
                        matched_section = next(
                            (s for s in expected_sections if re.match(rf'^{re.escape(s)}(\s|$)', line_text, re.I) or line_text.lower().startswith(s.lower())),
                            None
                        )
                        if matched_section or (is_bold and any(line_text.upper().startswith(s) for s in upper_expected)):
                            current_section = matched_section or next(
                                (s for s in expected_sections if line_text.upper().startswith(s)),
                                None
                            )
                            if current_section and current_section not in sections:
                                sections[current_section] = ""
                                logger.info(f"Detected section: {current_section}")
                        elif current_section:
                            sections[current_section] += line_text + "\n"
        
        # Normalize and log section content
        for sec in sections:
            sections[sec] = normalize_text(sections[sec])
            logger.info(f"Extracted section '{sec}': {sections[sec][:100]}...")
        
        doc.close()
        if not sections:
            logger.warning("No sections detected; treating as full resume")
            full_text = normalize_text(full_text)
            logger.info(f"Full resume text: {full_text[:100]}...")
            return {"Full Resume": full_text}
        return sections
    except Exception as e:
        logger.error(f"Failed to extract sections from PDF: {str(e)}")
        full_text = normalize_text(extract_text_from_pdf(pdf_content))
        logger.info(f"Full resume text (fallback): {full_text[:100]}...")
        return {"Full Resume": full_text}

def clean_text(text):
    if not text:
        return ""
    text = normalize_text(text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stopwords = set(NLTK_STOPWORDS.words('english')).union(SPACY_STOPWORDS).union(custom_stopwords)
    tokens = [token for token in tokens if token not in stopwords and len(token) > 2]  # Exclude short tokens
    return ' '.join(tokens)

def extract_skills(text):
    if not text:
        return []
    doc = nlp(text)
    skills = [ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL"]
    logger.info(f"Extracted raw skills: {skills}")
    skills = list(set(skills) - rejected_skills)
    logger.info(f"Skills after rejecting: {skills}")
    return skills

def verify_skills(skills, model, tokenizer):
    """
    Uses Gemma-3 to batch-verify if extracted terms are valid professional skills.
    """
    if not skills:
        return []
    
    prompt = (
        "Determine if each of the following is a valid professional skill for a resume. "
        "Consider technical skills (e.g., programming languages like Python, C++; tools like TensorFlow, OpenCV; frameworks like Django, React; and domain-specific terms like computer vision, augmented reality) "
        "and soft skills (e.g., communication, teamwork, problem solving) as valid. "
        "Exclude general terms (e.g., certified, project, data, system) unless they are contextually specific (e.g., database optimization). "
        "Respond with 'Yes' or 'No' for each skill, separated by commas, in the same order as provided:\n" +
        ", ".join(skills)
    )
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cpu")
    
    try:
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=500,
            do_sample=False
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.debug(f"Skill verification response: {response}")
        answers = [ans.strip() for ans in response.split(',')]
        if len(answers) != len(skills):
            logger.warning(f"Skill verification response length mismatch: expected {len(skills)}, got {len(answers)}")
            answers = answers[:len(skills)] + ['No'] * (len(skills) - len(answers))
        verified = [skill for skill, ans in zip(skills, answers) if ans.lower().startswith('yes')]
        logger.info(f"Verified skills: {verified}")
        return verified
    except Exception as e:
        logger.error(f"Error in skill verification: {str(e)}")
        return skills  # Fallback to unverified skills

def classify_skills(skills, model, tokenizer):
    """
    Classifies skills as hard or soft using Gemma-3, avoiding hardcoded lists.
    """
    hard_skills = set()
    soft_skills = set()
    
    if not skills:
        return hard_skills, soft_skills
    
    prompt = (
        "Classify each of the following skills as 'Hard' (technical skills like programming languages, tools, frameworks, or domain-specific knowledge) "
        "or 'Soft' (interpersonal or professional skills like communication, teamwork). "
        "Respond with 'Hard' or 'Soft' for each skill, separated by commas, in the same order as provided:\n" +
        ", ".join(skills)
    )
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cpu")
    
    try:
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=500,
            do_sample=False
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.debug(f"Skill classification response: {response}")
        classifications = [cls.strip() for cls in response.split(',')]
        if len(classifications) != len(skills):
            logger.warning(f"Skill classification response length mismatch: expected {len(skills)}, got {len(classifications)}")
            classifications = classifications[:len(skills)] + ['Soft'] * (len(skills) - len(classifications))
        
        for skill, cls in zip(skills, classifications):
            if cls.lower().startswith('hard'):
                hard_skills.add(skill)
            else:
                soft_skills.add(skill)
        
        logger.info(f"Classified hard skills: {hard_skills}")
        logger.info(f"Classified soft skills: {soft_skills}")
        return hard_skills, soft_skills
    except Exception as e:
        logger.error(f"Error in skill classification: {str(e)}")
        # Fallback: assume technical-sounding skills are hard, others are soft
        for skill in skills:
            if any(term in skill.lower() for term in ['programming', 'language', 'framework', 'tool', 'software', 'technology']):
                hard_skills.add(skill)
            else:
                soft_skills.add(skill)
        return hard_skills, soft_skills

def compute_similarity(resume_text, jd_text):
    if not embedder or not resume_text or not jd_text:
        logger.warning("Cannot compute similarity: embedder not loaded or empty input")
        return 0.0
    try:
        embeddings = embedder.encode([resume_text, jd_text])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] * 100
        logger.info(f"Computed similarity score: {similarity}")
        return similarity
    except Exception as e:
        logger.error(f"Error in computing similarity: {str(e)}")
        return 0.0

def analyze_resume(resume_text):
    if not resume_text:
        return {
            "sections": {s: False for s in [
                "Professional Summary", "Technical Skills", "Education", "Professional Experience",
                "Projects", "Certifications & Training", "Achievements & Key Strengths", "Languages"
            ]},
            "quantifiable_pct": 0.0,
            "action_verb_pct": 0.0,
            "repeated_words": {},
            "buzzwords_found": [],
            "filler_found": []
        }
    
    sections = {
        "Professional Summary": False,
        "Technical Skills": False,
        "Education": False,
        "Professional Experience": False,
        "Projects": False,
        "Certifications & Training": False,
        "Achievements & Key Strengths": False,
        "Languages": False
    }
    for section in sections:
        if section.lower() in resume_text.lower():
            sections[section] = True
    
    tokens = word_tokenize(resume_text.lower())
    quantifiable = sum(1 for token in tokens if re.match(r'\d+', token)) / len(tokens) * 100 if tokens else 0
    action_verbs = {"developed", "led", "built", "designed", "implemented", "analyzed"}
    action_verb_pct = sum(1 for token in tokens if token in action_verbs) / len(tokens) * 100 if tokens else 0
    
    repeated_words = {}
    for token in tokens:
        if token not in NLTK_STOPWORDS.words('english'):
            repeated_words[token] = repeated_words.get(token, 0) + 1
    repeated_words = {k: v for k, v in repeated_words.items() if v > 1}
    
    buzzwords = {"innovative", "dynamic", "proactive", "synergy"}
    filler_words = {"very", "really", "just"}
    buzzwords_found = [w for w in tokens if w in buzzwords]
    filler_found = [w for w in tokens if w in filler_words]
    
    return {
        "sections": sections,
        "quantifiable_pct": quantifiable,
        "action_verb_pct": action_verb_pct,
        "repeated_words": repeated_words,
        "buzzwords_found": buzzwords_found,
        "filler_found": filler_found
    }

def get_huggingface_suggestions(resume_text, jd_text, resume_skills, jd_skills):
    if not resume_text or not jd_text:
        return {
            "ats_suggestions": [],
            "rewrite_suggestions": [],
            "hard_skills_suggestions": [],
            "soft_skills_suggestions": []
        }
    
    prompt = (
        f"Resume: {resume_text[:500]}...\nJob Description: {jd_text[:500]}...\n"
        f"Resume Skills: {', '.join(resume_skills)}\nJD Skills: {', '.join(jd_skills)}\n"
        "Provide ATS improvement suggestions and rewrite suggestions for the resume. Ensure suggestions are specific and actionable."
    )
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cpu")
    
    try:
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=500,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.debug(f"Suggestions response: {response}")
        
        ats_suggestions = [f"Add keywords: {s}" for s in jd_skills if s not in resume_skills]
        rewrite_suggestions = [line.strip() for line in response.split('\n') if line.strip()]
        hard_skills_suggestions = [s for s in jd_skills if s not in resume_skills]
        soft_skills_suggestions = []
        
        return {
            "ats_suggestions": ats_suggestions,
            "rewrite_suggestions": rewrite_suggestions,
            "hard_skills_suggestions": hard_skills_suggestions,
            "soft_skills_suggestions": soft_skills_suggestions
        }
    except Exception as e:
        logger.error(f"Error in getting suggestions: {str(e)}")
        return {
            "ats_suggestions": [],
            "rewrite_suggestions": [],
            "hard_skills_suggestions": [],
            "soft_skills_suggestions": []
        }

def get_huggingface_ats_score(resume_text, jd_text):
    if not resume_text or not jd_text:
        return {"ats_score": 0.0, "ats_issues": ["Empty or invalid resume content"]}
    
    prompt = (
        f"Resume: {resume_text[:500]}...\nJob Description: {jd_text[:500]}...\n"
        "Estimate ATS compatibility score (0-100) and list specific issues affecting the score."
    )
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cpu")
    
    try:
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=500,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        logger.debug(f"ATS score response: {response}")
        
        score_match = re.search(r'\b(\d{1,3})\b', response)
        score = float(score_match.group(1)) if score_match else 50.0
        issues = [line.strip() for line in response.split('\n') if line.strip() and not line.startswith(str(int(score)))]
        
        return {"ats_score": score, "ats_issues": issues}
    except Exception as e:
        logger.error(f"Error in ATS scoring: {str(e)}")
        return {"ats_score": 50.0, "ats_issues": ["Failed to generate ATS score"]}

def filter_grammar_errors(errors, resume_text):
    """Filter out grammar errors caused by proper nouns or encoding artifacts."""
    proper_nouns = {"DHYANESWAR", "BACHU", "Hagen", "Germany", "Fachhochschule", "Sudwestfalen", "ValueLabs", "Shutterfly", "LumenPhish"}
    filtered_errors = []
    for err in errors:
        if "Possible spelling mistake" in err.message:
            context = err.context.lower()
            if any(pn.lower() in context for pn in proper_nouns) or any(c in context for c in ['¨', '', '#', 'ï', '§']):
                continue
        if "Unpaired symbol" in err.message and '”' in err.context:
            continue
        filtered_errors.append(err)
    return filtered_errors

def save_rejected_skills(removed_skills, suggested_skills):
    try:
        with open("rejected_skills.json", "r", encoding="utf-8") as f:
            current_rejected = set(json.load(f))
    except:
        current_rejected = set()
    
    current_rejected.update(removed_skills)
    
    with open("rejected_skills.json", "w", encoding="utf-8") as f:
        json.dump(list(current_rejected), f, indent=2)
    
    try:
        with open("suggested_skills.json", "r", encoding="utf-8") as f:
            current_suggested = set(json.load(f))
    except:
        current_suggested = set()
    
    current_suggested.update(suggested_skills)
    
    with open("suggested_skills.json", "w", encoding="utf-8") as f:
        json.dump(list(current_suggested), f, indent=2)
    
    with open("feedback_log.json", "a", encoding="utf-8") as f:
        json.dump({"removed_skills": list(removed_skills), "suggested_skills": list(suggested_skills)}, f)
        f.write("\n")

def load_rejected_skills():
    try:
        with open("rejected_skills.json", "r", encoding="utf-8") as f:
            return set(json.load(f))
    except:
        return set()

def match_resume_with_jd(resume_content, job_description):
    try:
        logger.info("Starting resume and JD matching process.")
        
        # Validate inputs
        if not job_description:
            raise ValueError("Job description cannot be empty")
        
        # Extract resume sections (handle PDF or text input)
        if isinstance(resume_content, io.BytesIO):
            resume_text = extract_text_from_pdf(resume_content.getvalue())
            if not resume_text:
                raise ValueError("Failed to extract text from resume PDF")
            resume_sections = extract_sections_from_pdf(resume_content.getvalue())
        else:
            if not resume_content:
                raise ValueError("Resume content cannot be empty")
            resume_sections = {"Full Resume": resume_content}
            resume_text = resume_content
        
        # Extract and verify skills per section
        resume_skills_by_section = {}
        all_resume_skills = set()
        for section, text in resume_sections.items():
            cleaned_text = clean_text(text)
            raw_skills = extract_skills(cleaned_text)
            verified_skills = verify_skills(raw_skills, model, tokenizer)
            resume_skills_by_section[section] = sorted(list(set(verified_skills)))
            all_resume_skills.update(verified_skills)
        
        # JD skills (overall, no sections)
        cleaned_jd = clean_text(job_description)
        raw_jd_skills = extract_skills(cleaned_jd)
        verified_jd_skills = verify_skills(raw_jd_skills, model, tokenizer)
        jd_skills_raw = set(verified_jd_skills)
        
        # Matching
        matched_skills = jd_skills_raw.intersection(all_resume_skills)
        jd_match_pct = (len(matched_skills) / len(jd_skills_raw)) * 100 if jd_skills_raw else 0
        
        # Similarity
        similarity_score = compute_similarity(resume_text, job_description)
        
        # Resume analysis
        resume_analysis = analyze_resume(resume_text)
        quantifiable_pct = resume_analysis["quantifiable_pct"]
        action_verb_pct = resume_analysis["action_verb_pct"]
        repeated_words = resume_analysis["repeated_words"]
        buzzwords_found = resume_analysis["buzzwords_found"]
        filler_found = resume_analysis["filler_found"]
        sections = resume_analysis["sections"]
        
        # Grammar check
        grammar_errors = []
        if grammar_tool and resume_text:
            raw_errors = grammar_tool.check(resume_text)
            grammar_errors = filter_grammar_errors(raw_errors, resume_text)
        
        # LLM-based suggestions and ATS score
        huggingface_sugg = get_huggingface_suggestions(resume_text, job_description, all_resume_skills, jd_skills_raw)
        huggingface_ats = get_huggingface_ats_score(resume_text, job_description)
        ats_score = huggingface_ats.get("ats_score", 50.0)
        overall_score = (jd_match_pct * 0.4 + ats_score * 0.4 + quantifiable_pct * 0.1 + action_verb_pct * 0.1)
        ats_issues = huggingface_ats.get("ats_issues", [])
        
        # Classify skills
        resume_hard_skills, resume_soft_skills = classify_skills(all_resume_skills, model, tokenizer)
        jd_hard_skills, jd_soft_skills = classify_skills(jd_skills_raw, model, tokenizer)
        
        response = {
            "jd_match_pct": round(jd_match_pct, 2),
            "resume_skills": sorted(list(all_resume_skills)),
            "jd_skills": sorted(list(jd_skills_raw)),
            "matched_skills": sorted(list(matched_skills)),
            "ats_score": round(ats_score, 2),
            "overall_score": round(overall_score, 2),
            "ats_issues": ats_issues,
            "ats_suggestions": huggingface_sugg["ats_suggestions"],
            "grammar_errors": [{"message": err.message, "context": err.context, "replacements": err.replacements} for err in grammar_errors] if grammar_errors else [],
            "quantifiable_pct": round(quantifiable_pct, 2),
            "action_verb_pct": round(action_verb_pct, 2),
            "repeated_words": repeated_words,
            "buzzwords_found": buzzwords_found,
            "filler_found": filler_found,
            "rewrite_suggestions": huggingface_sugg["rewrite_suggestions"],
            "sections_detected": [k for k, v in sections.items() if v],
            "suggestions": list(set(huggingface_sugg["ats_suggestions"] + huggingface_sugg["rewrite_suggestions"])),
            "resume_hard_skills": sorted(list(resume_hard_skills)),
            "resume_soft_skills": sorted(list(resume_soft_skills)),
            "jd_hard_skills": sorted(list(jd_hard_skills)),
            "jd_soft_skills": sorted(list(jd_soft_skills)),
            "hard_skills_suggestions": huggingface_sugg["hard_skills_suggestions"],
            "soft_skills_suggestions": huggingface_sugg["soft_skills_suggestions"],
            "resume_skills_by_section": resume_skills_by_section
        }
        logger.info(f"Final JSON response: {json.dumps(response, indent=2)}")
        logger.info("✅ Successfully processed resume and JD.")
        return response
    except Exception as e:
        logger.error(f"❌ Error in match_resume_with_jd: {str(e)}", exc_info=True)
        return {
            "error": "An internal error occurred during processing.",
            "details": str(e)
        }
