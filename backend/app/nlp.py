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

# Download NLTK resources if needed
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize models
MODEL_PATH = "./skill_ner_model"
try:
    nlp = spacy.load(MODEL_PATH)
    print("✅ Loaded custom skill NER model.")
except Exception:
    print("⚠️ Falling back to en_core_web_sm.")
    nlp = spacy.load("en_core_web_sm")

nlp_loc = spacy.load("en_core_web_sm")

try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Loaded sentence-transformer model.")
except Exception as e:
    print(f"⚠️ Could not load sentence-transformer model: {e}")
    embedder = None

# Initialize grammar tool
grammar_tool = language_tool_python.LanguageTool('en-US')

# Load stopwords
def load_stopwords(json_path="stopwords.json"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            words = json.load(f)
        return set(word.lower() for word in words)
    except Exception:
        print(f"⚠️ Stopwords file '{json_path}' not found. Using NLTK stopwords.")
        return set(NLTK_STOPWORDS.words('english'))

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

# Common action verbs for impact
ACTION_VERBS = {
    "achieved", "managed", "developed", "led", "created", "implemented", "improved", "optimized", "designed", "launched",
    "analyzed", "built", "collaborated", "coordinated", "delivered", "enhanced", "facilitated", "generated", "initiated",
    "mentored", "negotiated", "organized", "planned", "resolved", "streamlined", "trained", "transformed", "utilized",
    "validated", "wrote"
}

# Common buzzwords and filler words
BUZZWORDS = {"synergize", "leverage", "dynamic", "proactive", "think outside the box", "paradigm shift", "game-changer"}
FILLER_WORDS = {"very", "really", "just", "quite", "perhaps", "that", "things", "stuff"}

# Verb replacement mapping for impactful rewrites
VERB_REPLACEMENTS = {
    "worked on": "developed",
    "helped": "collaborated",
    "made": "created",
    "did": "executed",
    "used": "utilized",
    "was responsible for": "managed",
    "handled": "oversaw",
    "participated": "contributed"
}

# Metric suggestions based on bullet point context
METRIC_SUGGESTIONS = {
    "developed|created|built": ["increased efficiency by 15%", "reduced processing time by 20%", "improved accuracy by 10%"],
    "optimized|improved|enhanced": ["boosted performance by 25%", "decreased latency by 30%", "improved throughput by 20%"],
    "collaborated|worked with|team": ["aligned 5+ cross-functional teams", "coordinated with 10+ stakeholders", "streamlined team workflows by 15%"],
    "implemented|integrated": ["reduced integration time by 25%", "enabled seamless interoperability for 3+ systems", "cut integration costs by 20%"],
    "tested|validated": ["achieved 99% test coverage", "reduced bugs by 30%", "ensured 100% compliance with standards"],
    "documented|wrote": ["created documentation adopted by 50+ developers", "reduced onboarding time by 20%", "standardized processes for 3+ teams"]
}

# Whitelist for grammar checking
GRAMMAR_WHITELIST = {
    "dhyaneswar", "bachu", "hagen", "gmail.com", "github.com", "linkedin.com",
    "value labs", "shutterfly", "lumenphish", "fachhochschule", "sudwestfalen",
    "cmr institute", "coursera", "nvidia", "udemy", "mongodb university", "aws"
}

def extract_text_from_pdf_file(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text", flags=fitz.TEXTFLAGS_TEXT | fitz.TEXT_PRESERVE_WHITESPACE)
    return text

def clean_skill(skill):
    skill = skill.strip().lower()
    skill = re.sub(r"[^a-z0-9\s\-/&+]", "", skill)
    skill = re.sub(r"\s{2,}", " ", skill).strip()
    return skill

def is_unwanted_skill(skill, rejected_skills):
    if not skill:
        return True
    normalized_skill = clean_skill(skill)
    if len(normalized_skill) < 2 or normalized_skill in STOP_WORDS or normalized_skill in SPACY_STOPWORDS:
        return True
    if normalized_skill in rejected_skills:
        return True
    return False

def extract_skills(text, is_jd=False, rejected_skills=None):
    if rejected_skills is None:
        rejected_skills = load_rejected_skills()
    doc = nlp(text)
    skills = set()
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            cleaned = clean_skill(ent.text)
            if not is_unwanted_skill(cleaned, rejected_skills):
                skills.add(cleaned)
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
        if ent.label_ in ("GPE", "LOC", "DATE", "PERSON", "ORG"):
            cleaned_text = cleaned_text[:ent.start_char] + cleaned_text[ent.end_char:]
    zip_patterns = [
        r"\b\d{5,}\b",
        r"\b\d{5}(-\d{4})?\b",
        r"\b[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z][ -]?\d[ABCEGHJ-NPRSTV-Z]\d\b",
    ]
    email_url_pattern = r"\b[\w\.-]+@[\w\.-]+\.\w+\b|http[s]?://[\w\.-/]+|github\.com/[\w\.-/]+|linkedin\.com/[\w\.-/]+"
    for pattern in zip_patterns + [email_url_pattern]:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
    words = cleaned_text.split()
    filtered_words = [word for word in words if not dateparser.parse(word)]
    cleaned_text = " ".join(filtered_words)
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s\.\-/&]", " ", cleaned_text)
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text).strip()
    return cleaned_text

def check_ats_compatibility(pdf_stream):
    issues = []
    fonts = set()
    has_images = False
    try:
        with fitz.open(stream=pdf_stream) as doc:
            for page in doc:
                page_fonts = page.get_fonts()
                for font in page_fonts:
                    if len(font) > 3:
                        fonts.add(font[3])
                images = page.get_images()
                if images:
                    has_images = True
        standard_fonts = {'Arial', 'Calibri', 'TimesNewRoman', 'Times New Roman', 'Helvetica', 'Verdana', 'Georgia', 'Courier'}
        non_standard_fonts = fonts - standard_fonts
        if has_images:
            issues.append("Resume contains images/graphics, which may not be parsed correctly by ATS.")
        if non_standard_fonts:
            issues.append(f"Non-standard fonts used: {', '.join(non_standard_fonts)}. ATS may have trouble reading them.")
    except Exception as e:
        print(f"⚠️ Error checking ATS compatibility: {e}")
    return issues

def extract_sections(text):
    sections = {}
    patterns = {
        "skills": r"\b(skills|technical skills|key skills|proficiencies|technical proficiencies|core competencies|skill set|expertise|abilities)\b",
        "experience": r"\b(experience|work experience|professional experience|employment history|work history|career history|professional history|job history|career|employment|jobs)\b",
        "education": r"\b(education|qualifications|academic background|degrees|academic history|educational background|studies|academia|certifications|training|academic|schooling|learning)\b",
        "summary": r"\b(summary|professional summary|career summary|objective|profile)\b",
        "contact": r"\b(contact|contact information|personal information)\b",
        "projects": r"\b(projects|personal projects|portfolio)\b",
        "certifications": r"\b(certifications|certificates|licenses)\b"
    }
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    lines = text.split('\n')
    current_section = None
    for line in lines:
        lower_line = line.strip().lower()
        matched = False
        for sec, pat in patterns.items():
            if re.search(pat, lower_line, re.IGNORECASE):
                current_section = sec
                sections[current_section] = ""
                matched = True
                break
        if not matched and current_section and line.strip():
            sections[current_section] += line + "\n"
    # Fallback for education
    if "education" not in sections:
        education_keywords = r"\b(university|college|degree|bachelor|master|phd|diploma|certificate|institution|academy|school|graduate|undergraduate)\b"
        for line in lines:
            if re.search(education_keywords, line.lower(), re.IGNORECASE):
                sections["education"] = sections.get("education", "") + line + "\n"
    # Fallback for experience
    if "experience" not in sections:
        experience_keywords = r"\b(job|position|role|work|employment|internship|project|professional|company|organization|firm|responsibilities|duties)\b"
        for line in lines:
            if re.search(experience_keywords, line.lower(), re.IGNORECASE):
                sections["experience"] = sections.get("experience", "") + line + "\n"
    # Fallback for skills
    if "skills" not in sections:
        skills_keywords = r"\b(skills|technical|soft|hard|proficiencies)\b"
        for line in lines:
            if re.search(skills_keywords, line.lower(), re.IGNORECASE):
                sections["skills"] = sections.get("skills", "") + line + "\n"
    # Extract bullet points from relevant sections
    bullet_sections = ["experience", "projects"]
    bullet_points = []
    for section in bullet_sections:
        if section in sections:
            section_text = sections[section]
            bullets = re.split(r'\n-|\n\*|\n•', section_text)
            bullet_points.extend([bp.strip() for bp in bullets if bp.strip() and len(bp.split()) >= 5 and len(bp.split()) <= 30])
    return sections, bullet_points

def check_grammar_spelling(text):
    # Filter out contact info and whitelisted terms
    text = remove_countries_zip_addresses_institutions(text)
    for term in GRAMMAR_WHITELIST:
        text = re.sub(rf"\b{re.escape(term)}\b", "", text, flags=re.IGNORECASE)
    matches = grammar_tool.check(text)
    errors = []
    seen_contexts = set()
    for match in matches:
        if match.ruleId != "MORFOLOGIK_RULE_EN_US" and match.context not in seen_contexts:  # Ignore generic spelling errors and duplicates
            errors.append({
                "message": match.message,
                "context": match.context[:100],
                "replacements": match.replacements[:3]
            })
            seen_contexts.add(match.context)
    return errors[:5]  # Limit to 5 unique errors

def check_quantifiable_achievements(bullet_points):
    quantifiable = 0
    total = len(bullet_points)
    for bp in bullet_points:
        if re.search(r'\d+%|\d+\s|increased|reduced|saved|generated|achieved|improved', bp, re.IGNORECASE):
            quantifiable += 1
    pct = (quantifiable / total * 100) if total > 0 else 0
    return pct, quantifiable, total

def check_action_verbs(text):
    words = word_tokenize(text.lower())
    verbs_found = [word for word in words if word in ACTION_VERBS]
    return len(verbs_found) / len(words) * 100 if len(words) > 0 else 0, verbs_found

def check_brevity(bullet_points):
    long_bullets = [bp for bp in bullet_points if len(bp.split()) > 25]
    return len(long_bullets), len(bullet_points), long_bullets

def check_repetition(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word not in STOP_WORDS]
    word_freq = nltk.FreqDist(words)
    repeated = {word: freq for word, freq in word_freq.items() if freq >= 8}  # Higher threshold
    return repeated

def check_buzzwords_filler(text):
    words = word_tokenize(text.lower())
    buzz_found = list(set(word for word in words if word in BUZZWORDS))  # Deduplicate
    filler_found = list(set(word for word in words if word in FILLER_WORDS))
    return buzz_found, filler_found

def suggest_rewrites(bullet_points, jd_text=""):
    suggestions = []
    lemmatizer = WordNetLemmatizer()
    doc = nlp(jd_text.lower()) if jd_text else None
    jd_keywords = set(ent.text.lower() for ent in doc.ents if ent.label_ in ("SKILL", "ORG", "PRODUCT")) if doc else set()
    
    # Filter out invalid bullet points
    contact_patterns = [
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",  # Emails
        r"http[s]?://[\w\.-/]+|github\.com/[\w\.-/]+|linkedin\.com/[\w\.-/]+",  # URLs
        r"\b\d{5}(-\d{4})?\b",  # ZIP codes
        r"\b\d{10}\b",  # Phone numbers
        r"\b(lange hagen|fachhochschule|cmr institute)\b"  # Institutions
    ]
    
    for bp in bullet_points:
        if not bp.strip() or len(bp.split()) < 5:
            continue
        # Skip contact info or irrelevant text
        if any(re.search(pattern, bp.lower(), re.IGNORECASE) for pattern in contact_patterns):
            continue
        if re.search(r"^\b(skills|experience|education|summary|contact|projects|certifications)\b", bp.lower(), re.IGNORECASE):
            continue
        
        words = word_tokenize(bp.lower())
        # Remove filler words and buzzwords
        cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in FILLER_WORDS and word not in BUZZWORDS]
        cleaned_bp = " ".join(cleaned_words)
        
        # Identify main verb
        main_verb = None
        for word in cleaned_words:
            if word in ACTION_VERBS:
                main_verb = word
                break
            for weak_verb, strong_verb in VERB_REPLACEMENTS.items():
                if weak_verb in cleaned_bp:
                    main_verb = weak_verb
                    cleaned_bp = cleaned_bp.replace(weak_verb, strong_verb)
                    break
        
        # Determine context for metric suggestion
        metric = "achieved significant improvement"
        if main_verb:
            for pattern, metrics in METRIC_SUGGESTIONS.items():
                if any(v in main_verb for v in pattern.split('|')):
                    metric = metrics[0]
                    break
        
        # Incorporate JD keywords
        if jd_keywords:
            for keyword in jd_keywords:
                if keyword in COMMON_SKILLS and keyword not in cleaned_bp:
                    cleaned_bp += f" leveraging {keyword}"
                    break
        
        # Ensure brevity
        cleaned_words = cleaned_bp.split()
        if len(cleaned_words) > 20:
            cleaned_bp = " ".join(cleaned_words[:15]) + "..."
        
        rewrite = f"{cleaned_bp.capitalize()} ({metric})"
        suggestions.append(rewrite)
    
    return suggestions

def suggest_skills(jd_text):
    doc = nlp(jd_text)
    hard_skills = [ent.text.lower() for ent in doc.ents if ent.label_ in ("SKILL", "ORG", "PRODUCT")]
    soft_skills = ["communication", "teamwork", "problem-solving", "leadership", "adaptability"]
    return list(set(hard_skills)), soft_skills

def match_resume_with_jd(resume_input, jd_text, rejected_skills=None):
    if rejected_skills is None:
        rejected_skills = load_rejected_skills()
    is_pdf = isinstance(resume_input, io.BytesIO)
    if is_pdf:
        resume_input.seek(0)
        resume_text = extract_text_from_pdf_file(resume_input)
        resume_input.seek(0)
        ats_issues = check_ats_compatibility(resume_input.read())
    else:
        resume_text = resume_input
        ats_issues = []
    resume_text_clean = remove_countries_zip_addresses_institutions(resume_text)
    jd_text_clean = remove_countries_zip_addresses_institutions(jd_text)
    sections, bullet_points = extract_sections(resume_text_clean)
    resume_skills = set(extract_skills_with_clustering(resume_text_clean, rejected_skills, is_jd=False))
    jd_skills = set(extract_skills_with_clustering(jd_text_clean, rejected_skills, is_jd=True))
    matched_skills, jd_match_pct = calculate_match(resume_skills, jd_skills)
    grammar_errors = check_grammar_spelling(resume_text_clean)
    quantifiable_pct, quantifiable_count, total_bullets = check_quantifiable_achievements(bullet_points)
    action_verb_pct, verbs_found = check_action_verbs(resume_text_clean)
    long_bullets_count, total_bullets_brevity, long_bullets = check_brevity(bullet_points)
    repeated_words = check_repetition(resume_text_clean)
    buzzwords_found, filler_found = check_buzzwords_filler(resume_text_clean)
    rewrite_suggestions = suggest_rewrites(bullet_points, jd_text_clean)
    hard_skills_sugg, soft_skills_sugg = suggest_skills(jd_text_clean)
    overall_score = (jd_match_pct + quantifiable_pct + action_verb_pct + (100 - (long_bullets_count / total_bullets_brevity * 100 if total_bullets_brevity > 0 else 0))) / 4
    ats_bonus = sum(10 for sec in ["skills", "experience", "education"] if sec in sections)
    ats_penalty = 10 * len(ats_issues)
    ats_score = max(0, min(100, jd_match_pct + ats_bonus - ats_penalty))
    ats_suggestions = []
    suggestions = []
    if "skills" not in sections:
        ats_suggestions.append("Add a dedicated 'Skills' section with a clear header (e.g., 'Skills', 'Technical Skills').")
    if "experience" not in sections:
        ats_suggestions.append("Include a 'Work Experience' section with a clear header (e.g., 'Work Experience').")
    if "education" not in sections:
        ats_suggestions.append("Add an 'Education' section with a clear header (e.g., 'Education').")
    if "summary" not in sections:
        suggestions.append("Consider adding a 'Professional Summary' section to highlight your career overview.")
    for issue in ats_issues:
        ats_suggestions.append(issue + " Consider using standard fonts like Arial, Calibri, or Times New Roman.")
    if grammar_errors:
        suggestions.append("Fix grammar/spelling errors: " + "; ".join([f"{err['message']} in '{err['context'][:50]}...' (suggestions: {', '.join(err['replacements'])})" for err in grammar_errors]))
    if quantifiable_pct < 50:
        suggestions.append("Add quantifiable achievements to at least 50% of bullet points (e.g., 'Increased sales by 20%').")
    if action_verb_pct < 20:
        suggestions.append("Use more action verbs like 'achieved', 'managed', 'developed' to start bullet points.")
    if long_bullets_count > 0:
        suggestions.append(f"Shorten {long_bullets_count} bullet points exceeding 25 words: {', '.join([bp[:30] + '...' for bp in long_bullets[:3]])}")
    if repeated_words:
        suggestions.append("Reduce repetition of words: " + ", ".join([f"{word} ({freq} times)" for word, freq in repeated_words.items()]))
    if buzzwords_found or filler_found:
        suggestions.append("Avoid buzzwords/fillers: " + ", ".join(set(buzzwords_found + filler_found)))
    return {
        "resume_skills": sorted(resume_skills),
        "jd_skills": sorted(jd_skills),
        "matched_skills": sorted(matched_skills),
        "jd_match_pct": round(jd_match_pct, 2),
        "ats_score": round(ats_score, 2),
        "overall_score": round(overall_score, 2),
        "ats_issues": ats_issues,
        "ats_suggestions": ats_suggestions,
        "grammar_errors": grammar_errors,
        "quantifiable_pct": round(quantifiable_pct, 2),
        "action_verb_pct": round(action_verb_pct, 2),
        "repeated_words": repeated_words,
        "buzzwords_found": buzzwords_found,
        "filler_found": filler_found,
        "rewrite_suggestions": rewrite_suggestions,
        "hard_skills_suggestions": hard_skills_sugg,
        "soft_skills_suggestions": soft_skills_sugg,
        "sections_detected": list(sections.keys()),
        "suggestions": suggestions
    }