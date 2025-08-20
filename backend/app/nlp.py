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

# Initialize models
MODEL_PATH = "./skill_ner_model"
try:
    nlp = spacy.load(MODEL_PATH)
    print("✅ Loaded custom skill NER model.")
except Exception:
    print("⚠️ Falling back to en_core_web_sm.")
    nlp = spacy.load("en_core_web_sm")

nlp_loc = spacy.load("en_core_web_sm")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Loaded sentence-transformer model.")
except Exception as e:
    print(f"⚠️ Could not load sentence-transformer model: {e}")
    embedder = None

# Load stopwords
def load_stopwords(json_path="stopwords.json"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            words = json.load(f)
        return set(word.lower() for word in words)
    except Exception:
        print(f"⚠️ Stopwords file '{json_path}' not found. Using empty set.")
        return set()

STOP_WORDS = load_stopwords()

# Load rejected skills
REJECTED_SKILLS_FILE = "rejected_skills.json"
def load_rejected_skills():
    try:
        with open(REJECTED_SKILLS_FILE, "r", encoding="utf-8") as f:
            rejected = json.load(f)
        normalized_rejected = {clean_skill(skill) for skill in rejected if skill}
        print(f"✅ Loaded rejected skills: {normalized_rejected}")
        return normalized_rejected
    except Exception as e:
        print(f"⚠️ Failed to load rejected skills: {e}")
        return set()

# Load suggested skills
SUGGESTED_SKILLS_FILE = "suggested_skills.json"
def load_suggested_skills():
    try:
        with open(SUGGESTED_SKILLS_FILE, "r", encoding="utf-8") as f:
            suggested = json.load(f)
        normalized_suggested = {clean_skill(skill) for skill in suggested if skill}
        print(f"✅ Loaded suggested skills: {normalized_suggested}")
        return normalized_suggested
    except Exception:
        print(f"⚠️ Suggested skills file '{SUGGESTED_SKILLS_FILE}' not found. Using empty set.")
        return set()

def save_suggested_skills(new_suggested_skills):
    existing_suggested = load_suggested_skills()
    normalized_suggested = {clean_skill(skill) for skill in new_suggested_skills if skill}
    existing_suggested.update(normalized_suggested)
    try:
        with open(SUGGESTED_SKILLS_FILE, "w", encoding="utf-8") as f:
            json.dump(list(existing_suggested), f, indent=2)
        print(f"✅ Saved suggested skills: {existing_suggested}")
    except Exception as e:
        print(f"⚠️ Could not save suggested skills: {e}")

def save_rejected_skills(new_rejected_skills, new_suggested_skills=None):
    existing_rejected = load_rejected_skills()
    normalized_rejected = {clean_skill(skill) for skill in new_rejected_skills if skill}
    existing_rejected.update(normalized_rejected)
    try:
        with open(REJECTED_SKILLS_FILE, "w", encoding="utf-8") as f:
            json.dump(list(existing_rejected), f, indent=2)
        # Log feedback
        feedback_data = {"removed_skills": list(normalized_rejected)}
        if new_suggested_skills:
            normalized_suggested = {clean_skill(skill) for skill in new_suggested_skills if skill}
            save_suggested_skills(normalized_suggested)
            feedback_data["suggested_skills"] = list(normalized_suggested)
        with open("feedback_log.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_data) + "\n")
        print(f"✅ Saved rejected skills: {existing_rejected}")
    except Exception as e:
        print(f"⚠️ Could not save rejected skills: {e}")

# Common skills for both resumes and JDs
COMMON_SKILLS = {
    "python", "java", "django", "flask", "spring", "mongodb", "mysql", "aws",
    "tensorflow", "opencv", "javascript", "css3", "html5", "react", "angular",
    "node.js", "sql", "nosql", "git", "docker", "kubernetes", "backend",
    "frontend", "full-stack", "full stack", "web development",
    "software development", "database management", "cloud computing",
    "devops", "ci/cd", "api development", "rest api", "graphql",
    "machine learning", "data science", "data analysis", "ui/ux",
    "user interface", "user experience", "design", "deploy", "code",
    "testing", "automation", "security", "networking", "agile",
    "scrum", "project management", "microservices", "architecture"
}.union(load_suggested_skills())

def extract_text_from_pdf_file(file):
    text_fitz = ""
    text_pypdf2 = ""

    if isinstance(file, io.BytesIO):
        # Extract using PyMuPDF
        try:
            with fitz.open(stream=file, filetype="pdf") as doc:
                for page in doc:
                    page_text = page.get_text()
                    if page_text:
                        text_fitz += page_text + "\n"
        except Exception as e:
            print(f"⚠️ PyMuPDF failed: {e}")

        # Extract using PyPDF2
        try:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pypdf2 += page_text + "\n"
        except Exception as e:
            print(f"⚠️ PyPDF2 failed: {e}")
    else:
        # Assume file is text string
        return file

    # Combine and normalize
    combined_text = text_fitz + "\n" + text_pypdf2
    if combined_text.strip():
        sentences = nltk.sent_tokenize(combined_text)
        unique_sentences = list(dict.fromkeys(sentences))
        combined_text = " ".join(unique_sentences)
        combined_text = re.sub(r"\s+", " ", combined_text).strip()
    return combined_text

def clean_skill(skill):
    skill = skill.lower().strip()
    skill = skill.strip(string.punctuation)
    skill = re.sub(r'[-/]+$', '', skill)
    skill = re.sub(r'[,]+$', '', skill)
    return skill

UNWANTED_PATTERNS = [
    r"\b(bachelor|master|phd|degree|cgpa|certification|certified|udemy|coursera|project|experience|award|proficiency|fluent|native|born|phone|email|@|\+49|[\d]{10,})\b",
    r"\b(and|with|in|of|for|on|the|by|to|at|from|as|including|such as|like|also|that|this)\b",
    r"[\w\.-]+@[\w\.-]+\.\w+",  # Emails
    r"http[s]?://|www\.|linkedin\.com|github\.com",  # URLs
    r"^[/-]+|[/-]+$",  # Leading/trailing slashes or dashes
    r"\b\d{4}\b",  # Four-digit numbers (likely years)
    r"^[a-z]$",  # Single letters
    r"^[^\w\s]+$",  # Only punctuation
    r"\b(city|state|country|hyderabad|udwestfalen|university|college|institute)\b",  # Locations and institutions
    r"\b[\w-]*\d+[\w-]*\b",  # Terms with numbers (e.g., dhyaneswar-bachu-102a65202)
]

def is_unwanted_skill(skill, rejected_skills):
    skill = clean_skill(skill)
    if not skill or len(skill) < 2:
        return True
    if skill in SPACY_STOPWORDS or skill in STOP_WORDS:
        return True
    if skill in {"and", "or", "but", "with", "without"}:
        return True
    for pattern in UNWANTED_PATTERNS:
        if re.search(pattern, skill, re.IGNORECASE):
            return True
    if skill in rejected_skills:
        return True
    if skill in COMMON_SKILLS:
        return False
    if skill.startswith('-') or skill.endswith('-') or '/' in skill:
        return True
    return False

def extract_skills(text, is_jd=False, rejected_skills=None):
    if rejected_skills is None:
        rejected_skills = load_rejected_skills()
    doc = nlp(text)
    skills = set()
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            skill_text = clean_skill(ent.text)
            if not is_unwanted_skill(skill_text, rejected_skills):
                skills.add(skill_text)
    # Keyword matching for both resumes and JDs
    for skill in COMMON_SKILLS:
        if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE):
            if not is_unwanted_skill(skill, rejected_skills):
                skills.add(skill)
    print(f"Extracted skills (is_jd={is_jd}): {skills}")
    return skills

def cluster_skills(skills, eps=0.35, min_samples=1):
    if not skills:
        return []
    if embedder is None:
        return [[skill] for skill in skills]
    embeddings = embedder.encode(list(skills))
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    labels = clustering.labels_
    clusters = dict()
    for skill, label in zip(skills, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(skill)
    print(f"Clustered skills: {clusters}")
    return list(clusters.values())

def cluster_and_canonicalize(skills):
    clusters = cluster_skills(skills)
    canonical_dict = dict()
    for cluster in clusters:
        canonical = sorted(cluster, key=lambda x: (len(x), x))[0]
        canonical_dict[canonical] = cluster
    return canonical_dict

def canonicalize_skill(skill, canonical_dict):
    for canonical, synonyms in canonical_dict.items():
        if skill in synonyms:
            return canonical
    return skill

def canonicalize_skills(skills, canonical_dict):
    canonical_skills = set()
    for skill in skills:
        canonical = canonicalize_skill(skill, canonical_dict)
        canonical_skills.add(canonical)
    return canonical_skills

def extract_skills_with_clustering(text, rejected_skills=None, is_jd=False):
    if rejected_skills is None:
        rejected_skills = load_rejected_skills()
    ner_skills = extract_skills(text, is_jd, rejected_skills)
    print(f"NER skills (is_jd={is_jd}): {ner_skills}")
    sentences = nltk.sent_tokenize(text)
    candidate_skills = []
    for sentence in sentences:
        words = re.split(r"\s+", sentence.lower())
        for i in range(len(words)):
            for n in range(1, 3 if is_jd else 2):
                ngram = " ".join(words[i:i+n])
                ngram = clean_skill(ngram)
                if ngram and not is_unwanted_skill(ngram, rejected_skills) and len(ngram.split()) <= (3 if is_jd else 2):
                    candidate_skills.append(ngram)
    print(f"Candidate skills (is_jd={is_jd}): {candidate_skills}")
    combined_skills = list(set(ner_skills.union(candidate_skills)))
    canonical_dict = cluster_and_canonicalize(combined_skills)
    canonical_skills = canonicalize_skills(combined_skills, canonical_dict)
    filtered_skills = [skill for skill in canonical_skills if skill not in rejected_skills]
    print(f"Final skills (is_jd={is_jd}): {filtered_skills}")
    return sorted(filtered_skills)

def calculate_match(resume_skills, jd_skills):
    matched_skills = resume_skills & jd_skills
    jd_match_pct = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 0
    return matched_skills, jd_match_pct

def remove_countries_zip_addresses_institutions(text):
    doc = nlp_loc(text)
    cleaned_text = text
    for ent in reversed(doc.ents):
        if ent.label_ in ("GPE", "LOC", "DATE"):
            cleaned_text = cleaned_text[:ent.start_char] + cleaned_text[ent.end_char:]
    zip_patterns = [
        r"\b\d{5,}\b",
        r"\b\d{5}(-\d{4})?\b",
        r"\b[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z][ -]?\d[ABCEGHJ-NPRSTV-Z]\d\b",
    ]
    for pattern in zip_patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
    words = cleaned_text.split()
    filtered_words = [word for word in words if not dateparser.parse(word)]
    cleaned_text = " ".join(filtered_words)
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s\.\-/&]", " ", cleaned_text)
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text).strip()
    return cleaned_text

def match_resume_with_jd(resume_input, jd_text, rejected_skills=None):
    if rejected_skills is None:
        rejected_skills = load_rejected_skills()
    resume_text = extract_text_from_pdf_file(resume_input)
    resume_text_clean = remove_countries_zip_addresses_institutions(resume_text)
    jd_text_clean = remove_countries_zip_addresses_institutions(jd_text)
    resume_skills = set(extract_skills_with_clustering(resume_text_clean, rejected_skills, is_jd=False))
    jd_skills = set(extract_skills_with_clustering(jd_text_clean, rejected_skills, is_jd=True))
    matched_skills, jd_match_pct = calculate_match(resume_skills, jd_skills)
    return {
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "matched_skills": sorted(matched_skills),
        "jd_match_pct": round(jd_match_pct, 2)
    }