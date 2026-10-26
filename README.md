# GapFinder AI — RAG Learning Platform

A VS Code-friendly student prototype with a responsive dashboard, login/signup, PDF/TXT/DOCX upload, Retrieval-Augmented Generation (RAG), source passages, quiz practice, learning analytics, revision topics, and YouTube topic search.

## Run locally
1. Install Python 3.10 or newer and open this folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Create and activate a virtual environment (Windows PowerShell):
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   If `py` is unavailable, use `python -m venv .venv`.
4. Install packages:
   ```powershell
   pip install -r requirements.txt
   ```
5. Start the server:
   ```powershell
   python app.py
   ```
6. Open **http://127.0.0.1:5001** in your browser and create an account. This address is local to your computer and is not a public website link.
7. Upload readable PDF, TXT, or DOCX study notes, then ask questions in AI Tutor.

## How RAG works in this project
1. Extract text from the uploaded file.
2. Split it into overlapping passages.
3. Use TF-IDF and cosine similarity to retrieve passages relevant to a question.
4. Return the relevant source passages. If `OPENAI_API_KEY` is configured, send only retrieved passages as context to the model for a grounded answer / quiz.

### Optional generated AI answers and quizzes
Without an API key, local retrieval still works and the tutor displays the matching passages. To enable generated responses, set your key in the same terminal before starting Flask:
```powershell
$env:OPENAI_API_KEY="your_api_key_here"
python app.py
```
Never put a real API key in frontend JavaScript or commit it to a public repository.

## Notes / limitations
- This is a local demo, not a production deployment. It currently stores uploaded material in a shared local dataset; do not upload sensitive files.
- Local demo login hashes passwords, but production deployment requires per-user data isolation, HTTPS, CSRF protections, rate limiting, and a strong secret key.
- Scanned PDFs may need OCR before upload.
- Video suggestions open YouTube search results; YouTube Data API integration is not included.
