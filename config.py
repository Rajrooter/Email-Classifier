"""
Configuration file for Email Classifier Bot
Modify these settings according to your needs
"""

# ============================================
# GEMINI API CONFIGURATION
# ============================================
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY = "GEMINI_API_KEY"  # ⚠️ REPLACE THIS

# Gemini model to use
GEMINI_MODEL = "gemini-2.5-pro"  # Fast and cost-effective

# ============================================
# EMAIL CLASSIFICATION SETTINGS
# ============================================
# Labels for email classification
EMAIL_LABELS = {
    "Jobs": "Job postings, recruiter messages, interview invitations, career opportunities",
    "Finance": "Bills, bank statements, invoices, payment confirmations, tax documents",
    "Promotions": "Marketing emails, sales, discount offers, advertisements",
    "Newsletters": "Subscriptions, digest emails, content updates, blogs",
    "Social": "Social media notifications, event invitations, personal networking",
    "Personal": "Direct correspondence from individuals, family, friends",
    "SPAM": "Unsolicited bulk mail, phishing attempts, suspicious content",
    "IMPORTANT": "Time-sensitive matters, confirmations, official documents",
    "Others": "Miscellaneous emails that don't fit other categories",
    "Verification":"Account verification emails,Verification codes(otp,pins),Account verification confirmation mails",
    "Banking": "Bank statements, transaction alerts, payment confirmations,account statements,loans"
}

# Maximum number of emails to process per run
MAX_EMAILS_PER_RUN = 50

# How many characters of email body to analyze
EMAIL_BODY_PREVIEW_LENGTH = 500

# ============================================
# AUTOMATION SETTINGS
# ============================================
# Check for new emails every X seconds
CHECK_INTERVAL_SECONDS = 900  # 15 minutes

# Enable continuous running mode
CONTINUOUS_MODE = True  # Set to False for single run

# ============================================
# LOGGING SETTINGS
# ============================================
# GEMINI API TEST FUNCTION
# ============================================
def test_gemini_api_key_and_model():
    """Test Gemini API key and model configuration"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content("Say hello!")
        print(f"Gemini API test successful. Model: {GEMINI_MODEL}")
        print(f"Response: {response.text.strip()}")
    except Exception as e:
        print(f"Gemini API test failed: {e}")
# ============================================
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FILE = "logs/classifier.log"
LOG_TO_CONSOLE = True

# ============================================
# ADVANCED SETTINGS
# ============================================
# Mark emails as read after classification
MARK_AS_READ = False  # Set to True if you want emails marked as read

# Archive emails after classification
ARCHIVE_AFTER_CLASSIFICATION = False

# Maximum retries for API failures
MAX_RETRIES = 3

# Delay between retries (seconds)
RETRY_DELAY = 5

# Gmail API scopes required
SCOPES = ['API SCOPE REQUIRED ']

# ============================================
# MAIN ENTRY POINT FOR TESTING
# ============================================
if __name__ == "__main__":
    print("Testing Gemini API key and model...")

    test_gemini_api_key_and_model()

