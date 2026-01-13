"""
Gmail Email Classifier Bot using Gemini AI
Secure, robust, production-ready version
"""

import os
import sys
import base64
import pickle
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("❌ python-dotenv is not installed. Run: pip install python-dotenv")
    sys.exit(1)

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    # Use the correct Gemini API package (new)
    import google.generativeai as genai
    import google.generativeai as genai
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Run: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client google-generativeai")
    sys.exit(1)

try:
    import config
except ImportError:
    print("❌ config.py not found! Please create it first.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    """Configure logging with file and console handlers"""
    log_dir = Path(config.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    handlers = []
    
    # File handler
    file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)
    
    # Console handler (optional)
    if config.LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger(__name__)


logger = setup_logging()


# ══════════════════════════════════════════════════════════════════════
# GMAIL AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════

class GmailAuthenticator:
    """
    Handles Gmail API authentication
    Supports both local development and cloud deployment
    """

    TOKEN_FILE = Path("token.pickle")
    CREDENTIALS_FILE = Path("credentials.json")

    def __init__(self):
        self.creds: Optional[Credentials] = None

    def _load_from_environment(self) -> bool:
        """Try to load credentials from environment variable (for cloud)"""
        env_token = os.getenv("GMAIL_TOKEN_PICKLE_B64")
        if not env_token:
            return False

        try:
            token_bytes = base64.b64decode(env_token)
            self.creds = pickle.loads(token_bytes)
            logger.info("✓ Loaded credentials from environment variable")
            return True
        except Exception as e:
            logger.warning(f"Failed to load credentials from environment: {e}")
            return False

    def _load_from_file(self) -> bool:
        """Try to load credentials from local token file"""
        if not self.TOKEN_FILE.exists():
            return False

        try:
            with open(self.TOKEN_FILE, 'rb') as f:
                self.creds = pickle.load(f)
            logger.info("✓ Loaded credentials from token.pickle")
            return True
        except Exception as e:
            logger.warning(f"Failed to load token file: {e}")
            return False

    def _refresh_credentials(self) -> bool:
        """Try to refresh expired credentials"""
        if not self.creds:
            return False

        if not self.creds.expired or not self.creds.refresh_token:
            return False

        try:
            logger.info("Refreshing expired credentials...")
            self.creds.refresh(Request())
            self._save_credentials()
            logger.info("✓ Credentials refreshed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to refresh credentials: {e}")
            return False

    def _interactive_auth(self):
        """Perform interactive OAuth flow (local development only)"""
        cred_path = Path(os.getenv("GOOGLE_CREDENTIALS_PATH", self.CREDENTIALS_FILE))
        
        if not cred_path.exists():
            raise FileNotFoundError(
                f"❌ Missing credentials file: {cred_path}\n"
                "Download credentials.json from Google Cloud Console:\n"
                "1. Go to APIs & Services → Credentials\n"
                "2. Download OAuth 2.0 Client JSON\n"
                "3. Save as credentials.json in project folder"
            )

        logger.info("Starting interactive authentication flow...")
        flow = InstalledAppFlow.from_client_secrets_file(
            str(cred_path), 
            config.SCOPES
        )
        
        self.creds = flow.run_local_server(port=0)
        self._save_credentials()
        logger.info("✓ Interactive authentication completed")

    def _save_credentials(self):
        """Save credentials to file (local development only)"""
        # Don't save in cloud environments
        if any(os.getenv(k) for k in ["RAILWAY_ENVIRONMENT", "RENDER", "FLY_APP_NAME", "HEROKU"]):
            logger.debug("Cloud environment detected - skipping token file save")
            return

        try:
            with open(self.TOKEN_FILE, 'wb') as f:
                pickle.dump(self.creds, f)
            logger.info("✓ Credentials saved to token.pickle")
        except Exception as e:
            logger.warning(f"Failed to save credentials: {e}")

    def authenticate(self) -> Any:
        """Main authentication method"""
        logger.info("Starting Gmail authentication...")

        # Try environment variable first (for cloud)
        if self._load_from_environment():
            if self.creds.valid:
                return self._build_service()
            elif self._refresh_credentials():
                return self._build_service()

        # Try local token file
        if self._load_from_file():
            if self.creds.valid:
                return self._build_service()
            elif self._refresh_credentials():
                return self._build_service()

        # Last resort: interactive flow
        self._interactive_auth()
        return self._build_service()

    def _build_service(self) -> Any:
        """Build and return Gmail service"""
        service = build('gmail', 'v1', credentials=self.creds)
        logger.info("✓ Gmail service initialized successfully")
        return service


# ══════════════════════════════════════════════════════════════════════
# EMAIL CLASSIFIER (GEMINI AI)
# ══════════════════════════════════════════════════════════════════════

class EmailClassifier:
    """Classifies emails using Google's Gemini AI"""

    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError(
                "❌ GEMINI_API_KEY not set!\n"
                "Set it as environment variable: export GEMINI_API_KEY='your-key'"
            )

        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.GEMINI_MODEL)
            logger.info(f"✓ Gemini AI model initialized: {config.GEMINI_MODEL}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini AI: {e}")

    def _build_prompt(self, email_data: Dict) -> str:
        """Build classification prompt for Gemini with reasoning"""
        label_list = "\n".join([
            f"{i+1}. {name}: {desc}"
            for i, (name, desc) in enumerate(config.EMAIL_LABELS.items())
        ])
        valid_labels = list(config.EMAIL_LABELS.keys())

        prompt = f"""You are an expert email classifier. Classify the email below into exactly ONE category.

ALLOWED CATEGORIES:
{label_list}

EMAIL TO CLASSIFY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Subject: {email_data['subject']}
From: {email_data['from']}
Body Preview: {email_data['body_preview']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLASSIFICATION RULES:
- Prioritize "Important" for time-sensitive content (deadlines, confirmations, appointments)
- Use "Spam" for clearly unsolicited or phishing attempts
- Use "Personal" for direct human correspondence from known contacts
- Use "Jobs" for genuine career opportunities (not job ads → those are Promotions)
- When multiple categories fit, choose the most specific one
- Use "Others" ONLY when nothing else fits
- Output MUST be exactly one of these: {', '.join(valid_labels)}

TASK:
1. Think step by step and write 2-3 sentences of reasoning for your choice.
2. On the LAST line, output only the category name (no explanation, no punctuation, just the label).

Example output:
Reasoning: ...
...
LabelName

Begin:
"""
        return prompt

    def classify(self, email_data: Dict) -> str:
        """
        Classify an email and return the label name
        Always returns a valid label from config.EMAIL_LABELS
        """
        try:
            prompt = self._build_prompt(email_data)
            
            # Call Gemini API with retry logic
            for attempt in range(config.MAX_RETRIES):
                try:
                    response = self.model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 0.1,
                            "max_output_tokens": 200,
                            "top_p": 0.8,
                            "top_k": 40,
                        }
                    )

                    # Extract and validate response
                    if not hasattr(response, 'text') or not response.text:
                        logger.warning(f"Empty response from Gemini (attempt {attempt + 1})")
                        if attempt < config.MAX_RETRIES - 1:
                            time.sleep(config.RETRY_DELAY)
                            continue
                        return "Others"

                    # Get the response text
                    full_response = response.text.strip()

                    # Extract last line as the classification
                    lines = [line.strip() for line in full_response.split('\n') if line.strip()]
                    if not lines:
                        logger.warning(f"No valid lines in response (attempt {attempt + 1})")
                        if attempt < config.MAX_RETRIES - 1:
                            time.sleep(config.RETRY_DELAY)
                            continue
                        return "Others"

                    classification = lines[-1].strip('."\'`*-:')

                    # Validate against allowed labels
                    if classification in config.EMAIL_LABELS:
                        logger.info(f"✓ Classified as: {classification}")
                        return classification

                    # Try case-insensitive match
                    for label in config.EMAIL_LABELS:
                        if classification.lower() == label.lower():
                            logger.info(f"✓ Classified as: {label} (case-corrected)")
                            return label

                    # Invalid classification
                    logger.warning(
                        f"Invalid classification '{classification}' (attempt {attempt + 1}). "
                        f"Response: {full_response[:100]}"
                    )

                    if attempt < config.MAX_RETRIES - 1:
                        time.sleep(config.RETRY_DELAY)
                        continue

                    # All retries exhausted
                    logger.error(f"All {config.MAX_RETRIES} attempts failed. Defaulting to 'Others'")
                    return "Others"
                except Exception as e:
                    logger.warning(f"Gemini API error (attempt {attempt + 1}): {e}")
                    if attempt < config.MAX_RETRIES - 1:
                        time.sleep(config.RETRY_DELAY)
                    else:
                        logger.error("All retry attempts failed")
                        return "Others"
        
        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            return "Others"


# ══════════════════════════════════════════════════════════════════════
# EMAIL MANAGER (GMAIL OPERATIONS)
# ══════════════════════════════════════════════════════════════════════

class EmailManager:
    """Handles all Gmail API operations"""

    def __init__(self, service):
        self.service = service
        self.label_cache: Dict[str, str] = {}
        self._initialize_labels()

    def _initialize_labels(self):
        """Create all required labels if they don't exist"""
        logger.info("Initializing Gmail labels...")
        
        try:
            # Get existing labels
            results = self.service.users().labels().list(userId='me').execute()
            existing_labels = {
                label['name']: label['id'] 
                for label in results.get('labels', [])
            }

            # Create or cache labels
            for label_name in config.EMAIL_LABELS:
                if label_name in existing_labels:
                    self.label_cache[label_name] = existing_labels[label_name]
                    logger.info(f"  ✓ Label exists: {label_name}")
                else:
                    # Create new label
                    label_body = {
                        'name': label_name,
                        'labelListVisibility': 'labelShow',
                        'messageListVisibility': 'show',
                        'color': self._get_label_color(label_name)
                    }
                    
                    created_label = self.service.users().labels().create(
                        userId='me',
                        body=label_body
                    ).execute()
                    
                    self.label_cache[label_name] = created_label['id']
                    logger.info(f"  ✓ Label created: {label_name}")

            logger.info(f"✓ All {len(self.label_cache)} labels initialized")

        except HttpError as e:
            logger.critical(f"Failed to initialize labels: {e}")
            raise

    def _get_label_color(self, label_name: str) -> Dict:
        """Get appropriate color for label"""
        color_map = {
            'Jobs': {'textColor': '#ffffff', 'backgroundColor': '#16a765'},
            'Finance': {'textColor': '#ffffff', 'backgroundColor': '#0b804b'},
            'Promotions': {'textColor': '#000000', 'backgroundColor': '#fad165'},
            'Newsletters': {'textColor': '#000000', 'backgroundColor': '#a4c2f4'},
            'Social': {'textColor': '#ffffff', 'backgroundColor': '#ac2b16'},
            'Personal': {'textColor': '#ffffff', 'backgroundColor': '#8e63ce'},
            'Spam': {'textColor': '#ffffff', 'backgroundColor': '#cf2100'},
            'Important': {'textColor': '#ffffff', 'backgroundColor': '#fb4c2f'},
            'Verification': {'textColor': '#ffffff', 'backgroundColor': '#42d692'},
            'Banking': {'textColor': '#ffffff', 'backgroundColor': '#285bac'},
            'Others': {'textColor': '#000000', 'backgroundColor': '#d3d3d3'},
        }
        return color_map.get(label_name, {'textColor': '#000000', 'backgroundColor': '#cccccc'})

    def get_unread_emails(self) -> List[Dict]:
        """Fetch unread emails from inbox"""
        logger.info(f"Fetching up to {config.MAX_EMAILS_PER_RUN} unread emails...")

        try:
            results = self.service.users().messages().list(
                userId='me',
                q='is:unread in:inbox',
                maxResults=config.MAX_EMAILS_PER_RUN
            ).execute()

            messages = results.get('messages', [])

            if not messages:
                logger.info("No unread emails found")
                return []

            logger.info(f"Found {len(messages)} unread emails")

            # Fetch full details for each message
            emails = []
            for msg in messages:
                email_data = self._get_message_details(msg['id'])
                if email_data:
                    emails.append(email_data)

            return emails

        except HttpError as e:
            logger.error(f"Failed to fetch emails: {e}")
            return []

    def _get_message_details(self, message_id: str) -> Optional[Dict]:
        """Get detailed information for a specific email"""
        try:
            message = self.service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()

            # Extract headers
            headers = message['payload'].get('headers', [])
            headers_dict = {h['name'].lower(): h['value'] for h in headers}

            # Extract body
            body = self._extract_body(message['payload'])
            body_preview = body[:config.EMAIL_BODY_PREVIEW_LENGTH] if body else ""

            return {
                'id': message_id,
                'subject': headers_dict.get('subject', '(No Subject)'),
                'from': headers_dict.get('from', '(Unknown Sender)'),
                'body_preview': body_preview,
                'snippet': message.get('snippet', '')
            }

        except HttpError as e:
            logger.warning(f"Failed to get message details for {message_id}: {e}")
            return None

    def _extract_body(self, payload: Dict) -> str:
        """Extract email body text from payload"""
        body = ""

        # Try to find text/plain part
        if 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain':
                    if 'data' in part.get('body', {}):
                        data = part['body']['data']
                        body = base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
                        break

            # If no text/plain, try text/html
            if not body:
                for part in payload['parts']:
                    if part.get('mimeType') == 'text/html':
                        if 'data' in part.get('body', {}):
                            data = part['body']['data']
                            body = base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
                            break

        # Try direct body data
        if not body and 'body' in payload:
            if 'data' in payload['body']:
                data = payload['body']['data']
                body = base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')

        return body

    def apply_label(self, message_id: str, label_name: str) -> bool:
        """Apply a label to an email"""
        # Validate label exists
        if label_name not in self.label_cache:
            logger.error(f"Label '{label_name}' not found in cache")
            return False

        label_id = self.label_cache[label_name]

        try:
            # Build modification request
            modify_body = {
                'addLabelIds': [label_id]
            }

            # Optionally mark as read
            if config.MARK_AS_READ:
                modify_body.setdefault('removeLabelIds', []).append('UNREAD')

            # Optionally archive
            if config.ARCHIVE_AFTER_CLASSIFICATION:
                modify_body.setdefault('removeLabelIds', []).append('INBOX')

            # Apply modifications
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body=modify_body
            ).execute()

            return True

        except HttpError as e:
            logger.error(f"Failed to apply label to {message_id}: {e}")
            return False


# ══════════════════════════════════════════════════════════════════════
# MAIN BOT ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════

class EmailClassifierBot:
    """Main orchestrator for email classification"""

    def __init__(self):
        self.authenticator = GmailAuthenticator()
        self.gmail_service = None
        self.classifier = None
        self.email_manager = None
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'by_label': {}
        }

    def initialize(self):
        """Initialize all components"""
        logger.info("═" * 70)
        logger.info("  🚀 EMAIL CLASSIFIER BOT - INITIALIZATION")
        logger.info("═" * 70)

        # Authenticate Gmail
        self.gmail_service = self.authenticator.authenticate()

        # Initialize Gemini classifier
        self.classifier = EmailClassifier()

        # Initialize email manager
        self.email_manager = EmailManager(self.gmail_service)

        logger.info("✓ All components initialized successfully")
        logger.info("═" * 70)

    def process_emails(self):
        """Process unread emails and classify them"""
        logger.info(f"\n📧 Processing cycle started: {datetime.now():%Y-%m-%d %H:%M:%S}")
        logger.info("─" * 70)

        # Fetch unread emails
        emails = self.email_manager.get_unread_emails()

        if not emails:
            logger.info("✓ No emails to process")
            return

        # Process each email
        for idx, email in enumerate(emails, 1):
            logger.info(f"\n📨 Email {idx}/{len(emails)}")
            logger.info(f"   Subject: {email['subject'][:65]}...")
            logger.info(f"   From: {email['from'][:50]}...")

            # Classify email
            classification = self.classifier.classify(email)

            # Apply label
            if classification and self.email_manager.apply_label(email['id'], classification):
                self.stats['successful'] += 1
                self.stats['by_label'][classification] = self.stats['by_label'].get(classification, 0) + 1
                logger.info(f"   ✓ Labeled as: {classification}")
            else:
                self.stats['failed'] += 1
                logger.error(f"   ✗ Failed to apply label")

            self.stats['total_processed'] += 1

            # Rate limiting
            time.sleep(config.RATE_LIMIT_DELAY)

        self._print_statistics()

    def _print_statistics(self):
        """Print processing statistics"""
        logger.info("\n" + "═" * 70)
        logger.info("  📊 PROCESSING STATISTICS")
        logger.info("═" * 70)
        logger.info(f"Total Processed  : {self.stats['total_processed']}")
        logger.info(f"Successful       : {self.stats['successful']}")
        logger.info(f"Failed           : {self.stats['failed']}")

        if self.stats['by_label']:
            logger.info("\nLabel Distribution:")
            for label, count in sorted(
                self.stats['by_label'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                logger.info(f"  {label:18} : {count}")

        logger.info("═" * 70)

    def run_once(self):
        """Run the bot once and exit"""
        self.initialize()
        self.process_emails()
        logger.info("\n✓ Single run completed successfully")

    def run_continuous(self):
        """Run the bot continuously with periodic checks"""
        self.initialize()

        logger.info(f"\n🔄 Continuous mode started")
        logger.info(f"   Check interval: {config.CHECK_INTERVAL_SECONDS} seconds")
        logger.info(f"   Press Ctrl+C to stop\n")

        try:
            while True:
                self.process_emails()
                
                logger.info(f"\n⏳ Next check in {config.CHECK_INTERVAL_SECONDS} seconds...")
                time.sleep(config.CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("\n\n🛑 Bot stopped by user")
            self._print_statistics()
            logger.info("Goodbye! 👋")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    try:
        # Validate configuration
        if not config.GEMINI_API_KEY:
            logger.critical("❌ GEMINI_API_KEY not set in environment!")
            return 1

        # Create and run bot
        bot = EmailClassifierBot()

        if config.CONTINUOUS_MODE:
            bot.run_continuous()
        else:
            bot.run_once()

        return 0

    except KeyboardInterrupt:
        logger.info("\n\n🛑 Interrupted by user")
        return 0

    except Exception as e:
        logger.critical(f"❌ Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
