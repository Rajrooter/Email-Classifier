"""
Email Classifier Bot - Configuration
All sensitive values should come from environment variables
"""

import os
from pathlib import Path

# ── Security first ──────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Critical: GEMINI_API_KEY environment variable is not set!")

# ── Gemini model selection ──────────────────────────────────────
# Updated to use the recommended model for higher quota
GEMINI_MODEL = "gemini-2.0-flash-001"

# ── Classification settings ─────────────────────────────────────
EMAIL_LABELS = {
    "Jobs":         "Job offers, recruiter messages, interviews, career opportunities",
    "Finance":      "Invoices, bills, bank statements, payments, taxes",
    "Promotions":   "Ads, marketing, discounts, newsletters (commercial)",
    "Newsletters":  "Content updates, blogs, industry news (non-promotional)",
    "Social":       "Social network notifications, event invites",
    "Personal":     "Direct human-to-human communication",
    "IMPORTANT":    "Deadlines, confirmations, official time-sensitive matters",
    "SPAM":         "Unsolicited / suspicious / phishing attempts",
    "Verification": "OTP, verification codes, account confirmations",
    "Banking":      "Bank statements, transactions, alerts, loans",
    "Others":       "Everything else that doesn't fit well"
}

# ── Processing limits & behaviour ───────────────────────────────
MAX_EMAILS_PER_RUN         = 40
EMAIL_BODY_PREVIEW_LENGTH  = 800
CHECK_INTERVAL_SECONDS     = 900          # 15 minutes
CONTINUOUS_MODE            = True

MARK_AS_READ               = False
ARCHIVE_AFTER_CLASSIFICATION = False

# ── Retry configuration ─────────────────────────────────────────
MAX_RETRIES                = 3
RETRY_DELAY                = 2            # seconds between retries
RATE_LIMIT_DELAY           = 1            # seconds between email processing

# ── Logging ─────────────────────────────────────────────────────
LOG_LEVEL      = "INFO"
LOG_FILE       = str(Path("logs/classifier.log"))
LOG_TO_CONSOLE = True

# ── Gmail API ───────────────────────────────────────────────────
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
