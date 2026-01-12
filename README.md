# Email Manager

A Gmail email classifier bot powered by Google APIs and Gemini AI. Automatically classifies emails, extracts text from attachments, and detects urgent messages.

## Features
- AI-powered email classification
- Gmail label management
- Attachment text extraction (PDF, DOCX, images, TXT)
- Urgent email detection
- Chat notification integration (Telegram, Slack, Discord)
- Designed for 24/7 cloud operation (GCP, headless)

## Setup
1. Clone the repo
2. Add your `credentials.json` (Google OAuth)
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python email_classifier.py`

## Deployment
- Remove sensitive files before pushing to GitHub
- Use `.gitignore` to exclude credentials and logs

## License
MIT
