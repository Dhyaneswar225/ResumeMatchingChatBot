# Resume Matching Chatbot

## Overview
This application provides a chatbot interface for matching resumes to job descriptions using advanced NLP techniques. It extracts skills from resumes (PDF or text) and job descriptions using spaCy for Named Entity Recognition (NER), SentenceTransformers for skill embeddings, and DBSCAN for skill clustering. Users can provide feedback to remove incorrect skills, which are stored for future filtering. The backend is built with FastAPI, and the frontend uses Express.js to serve a conversational HTML/CSS/JS interface.

## Features
- Chatbot UI for uploading resumes (PDF or text) and entering job descriptions.
- Extracts and clusters skills using a custom or fallback spaCy NER model.
- Computes a matching percentage based on overlapping skills.
- Displays results and allows users to review and remove incorrect skills via the chat interface.
- Persists rejected skills in `rejected_skills.json` and logs feedback in `feedback_log.json`.
- Resets state after feedback to allow new resume uploads.

## Project Structure
```
chatbot-app/
├── backend/                     # FastAPI (Python) backend
│   ├── main.py                  # FastAPI entry point
│   ├── requirements.txt         # Python dependencies
│   ├── train_skill_ner.py       # Script to train custom NER model
│   ├── stopwords.json           # Custom stopwords for skill filtering
│   ├── rejected_skills.json     # Persistent storage for rejected skills
│   └── app/                     # Backend app code
│       ├── __init__.py
│       ├── models.py            # Pydantic models
│       ├── routes.py            # API routes
│       └── nlp.py               # NLP logic (skill extraction, clustering)
├── frontend/                    # Express.js + HTML frontend
│   ├── server.js                # Express.js entry point
│   ├── package.json             # Node.js dependencies
│   └── public/                  # Static files
│       ├── index.html           # Chatbot UI
│       ├── style.css            # Styling
│       └── script.js            # Frontend logic
└── README.md                    # Project documentation
```

## Prerequisites
- **Python 3.9+**: Ensure Python is installed (3.9 or higher recommended).
- **Node.js 16+**: Required for the frontend.
- **pip** and **npm**: Package managers for Python and Node.js dependencies.
- **Optional**: A `train_data.json` file for training a custom spaCy NER model.

## Setup Instructions

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Install the spaCy English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. (Optional) Train a custom NER model:
   - Prepare a `train_data.json` file in the `backend/` directory with the format:
     ```json
     [
       {"text": "I know Python and Java", "entities": [[7, 13, "SKILL"], [18, 22, "SKILL"]]},
       ...
     ]
     ```
   - Run the training script:
     ```bash
     python train_skill_ner.py
     ```
   - This generates a `skill_ner_model` directory in `backend/`.
5. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```
   - The backend runs on `http://localhost:8000`.

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the frontend server:
   ```bash
   node server.js
   ```
   - The frontend runs on `http://localhost:3000`.

## Usage
1. Ensure the backend server is running (`uvicorn main:app --reload`).
2. Start the frontend server (`node server.js`).
3. Open `http://localhost:3000` in a browser.
4. Interact with the chatbot:
   - Click "Upload Resume" to select a PDF or text file.
   - Type the job description in the input box and click "Send" or press Enter.
   - Review the match results (similarity score and matched skills).
   - Review extracted skills and enter comma-separated incorrect skills to remove, or press Enter to skip.
   - The bot confirms feedback and prompts to upload a new resume for the next match.

## Troubleshooting
- **Cannot Upload Resume After Feedback**:
  - Ensure `frontend/public/script.js` is updated to reset the resume file input after feedback.
  - Verify the browser console for errors and check that the backend `/match` endpoint is accessible.
- **Backend Errors**:
  - **ImportError with `sentence_transformers`**: Ensure `sentence-transformers>=2.7.0` and `huggingface_hub>=0.17.0` are installed:
    ```bash
    pip uninstall sentence-transformers huggingface_hub
    pip install sentence-transformers>=2.7.0 huggingface_hub>=0.17.0
    ```
  - **spaCy Model Issues**: Verify that `en_core_web_sm` is installed or `skill_ner_model` is in `backend/`.
- **Frontend Errors**:
  - **ECONNREFUSED**: Ensure the backend is running on `http://localhost:8000` before starting the frontend. Check for port conflicts:
    ```bash
    netstat -aon | findstr :8000  # On Windows
    lsof -i :8000  # On Linux/Mac
    ```
  - **Deprecation Warning**: Using `http-proxy-middleware>=3.0.0` avoids `util._extend` warnings.
- **Port Conflicts**: Change ports in `backend/main.py` or `frontend/server.js` if `8000` or `3000` are in use.

## Notes
- **Stopwords**: Customize `backend/stopwords.json` to filter unwanted skills.
- **Rejected Skills**: Stored in `backend/rejected_skills.json` and updated via user feedback.
- **Feedback Log**: Saved in `backend/feedback_log.json` for tracking removed skills.
- **Custom NER Model**: The app falls back to `en_core_web_sm` if `skill_ner_model` is unavailable.
- **Dependencies**: Requires Python 3.9+ and Node.js 16+.

## Dependencies
- **Backend**: FastAPI, spaCy, SentenceTransformers, PyMuPDF, PyPDF2, NLTK, scikit-learn, numpy, dateparser.
- **Frontend**: Express.js, http-proxy-middleware.

For further assistance, contact the developer or refer to the project repository.
