"""
Gmail Email Classifier Bot using Gemini AI
Automatically classifies and labels emails based on content
"""


# Standard library imports
import os
import base64
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
import pickle
from pathlib import Path

# Third-party Google API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Gemini AI import
import google.generativeai as genai

# Local imports
import config
# (Optional) Attachment extraction utilities
# from attachment_text_extractor import extract_text_from_attachment


# ============================================
# LOGGING SETUP
# ============================================
def setup_logging():
    """Configure logging for the application"""
    # Create logs directory if it doesn't exist
    log_dir = Path(config.LOG_FILE).parent
    log_dir.mkdir(exist_ok=True)
    # Configure logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    # Set up file handler
    file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    # Set up console handler
    handlers = [file_handler]
    if config.LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(console_handler)
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        handlers=handlers
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================
# GMAIL API AUTHENTICATION
# ============================================
class GmailAuthenticator:
    """Handles Gmail API authentication"""

    def __init__(self):
        self.creds = None
        self.token_file = 'token.json'
        self.credentials_file = 'credentials.json'

    def authenticate(self):
        """Authenticate and return Gmail service"""
        # Railway: Write GOOGLE_CREDENTIALS env var to credentials.json if present
        if os.environ.get("GOOGLE_CREDENTIALS"):
            with open(self.credentials_file, "w") as f:
                f.write(os.environ["GOOGLE_CREDENTIALS"])
        logger.info("Starting Gmail authentication...")

        # Check if token.json exists with valid credentials
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                self.creds = pickle.load(token)

        # If credentials are invalid or don't exist, get new ones
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                logger.info("Refreshing expired credentials...")
                self.creds.refresh(Request())
            else:
                logger.info("Requesting new credentials...")
                if not os.path.exists(self.credentials_file):
                    logger.error(f"ERROR: {self.credentials_file} not found!")
                    logger.error("Please download OAuth credentials from Google Cloud Console")
                    raise FileNotFoundError(f"{self.credentials_file} not found")

                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, config.SCOPES
                )
                # Use console flow for headless/cloud environments
                self.creds = flow.run_console()

            # Save credentials for next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(self.creds, token)
            logger.info("OK credentials saved successfully")

        # Build and return Gmail service
        service = build('gmail', 'v1', credentials=self.creds)
        logger.info("OK Gmail authentication successful")
        return service


# ============================================
# GEMINI AI CLASSIFIER
# ============================================
class EmailClassifier:
    """Handles email classification using Gemini AI"""
    
    def __init__(self):
        """Initialize Gemini AI"""
        if config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            logger.error("ERROR: GEMINI_API_KEY not set in config.py!")
            raise ValueError("Please set your Gemini API key in config.py")
        
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        logger.info(f"OK Gemini AI initialized with model: {config.GEMINI_MODEL}")
    
    def build_classification_prompt(self, email_data: Dict) -> str:
        """Build the classification prompt for Gemini with Socratic method for deep reasoning"""
        labels_description = "\n".join([
            f"- {label}: {description}"
            for label, description in config.EMAIL_LABELS.items()
        ])

        prompt = f"""
You are an expert email classifier. Use the Socratic method: ask yourself probing questions about the email's intent, context, and content before deciding on a label. Do NOT rely only on keywords in the subject or body. Carefully analyze the full content, sender, and intent of the email. If the email is a job advertisement or promotion, but not a direct job offer or recruiter message, label it as 'Promotions' not 'Jobs'. If unsure, explain your reasoning and choose the most appropriate label.

ALLOWED LABELS:
{labels_description}

EMAIL TO CLASSIFY:
Subject: {email_data['subject']}
From: {email_data['from']}
Body Preview: {email_data['body_preview']}

CLASSIFICATION RULES:
1. Prioritize 'IMPORTANT' for time-sensitive content (confirmations, deadlines, appointments)
2. Choose 'SPAM' for suspicious or unsolicited bulk mail
3. Use 'Personal' for direct human correspondence from known contacts
4. Apply 'Jobs' for any career-related content (but NOT for job ads or resume services—those are 'Promotions')
5. When multiple labels could apply, select the most specific one
6. Use 'Others' only as a last resort
7. ONLY create a new label if NONE of the allowed labels fit the email's purpose after deep reasoning. Do NOT create unnecessary labels. If you must create a new label, follow Gmail label naming rules:
   - Max 225 characters
   - No special characters: / \\ * ? < > | {{ }}
   - Cannot be empty or start/end with spaces
   - Cannot use Gmail system labels (INBOX, SPAM, TRASH, UNREAD, STARRED, IMPORTANT, SENT, DRAFT, CATEGORY_PERSONAL, CATEGORY_SOCIAL, CATEGORY_PROMOTIONS, CATEGORY_UPDATES, CATEGORY_FORUMS)
   - Use clear, concise, descriptive names (1-2 words, CamelCase or Title Case preferred)

TASK:
First, use the Socratic method: ask yourself at least two questions about the email's true purpose and answer them. Then, explain your reasoning in 1-2 sentences. Ask yourself: "Does any existing label fit this email?" If yes, use that label. Only if no existing label fits, propose a new label name with a short justification. Finally, on a new line, output ONLY the label name (no punctuation, no explanation, just the label name)."""
        return prompt
    
    def classify_email(self, email_data: Dict) -> Optional[str]:
        """Classify an email using Gemini AI, robustly extracting the label from the response. If the label does not exist, create a new one using reasoning."""
        try:
            prompt = self.build_classification_prompt(email_data)
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            import string
            lines = [line.strip() for line in response_text.splitlines() if line.strip()]
            if lines:
                label_candidate = lines[-1].strip().strip(string.punctuation)
            else:
                label_candidate = ""

            if label_candidate in config.EMAIL_LABELS:
                logger.info(f"OK Classified as: {label_candidate}")
                return label_candidate
            else:
                # Use reasoning from Socratic method to create a new label
                reasoning = " ".join(lines[:-1]) if len(lines) > 1 else "No reasoning provided."
                logger.info(f"Creating new label '{label_candidate}' with reasoning: {reasoning}")
                # Optionally, you could add the new label to config.EMAIL_LABELS here
                # config.EMAIL_LABELS[label_candidate] = f"Created by AI reasoning: {reasoning}"
                return label_candidate
        except Exception as e:
            logger.error(f"ERROR: Classification error: {str(e)}")
            return "Others"


# ============================================
# GMAIL EMAIL MANAGER
# ============================================
class EmailManager:
    """Manages Gmail operations - fetching, labeling, etc."""
    
    # Gmail system labels that cannot be created
    GMAIL_SYSTEM_LABELS = {
        'INBOX', 'SPAM', 'TRASH', 'UNREAD', 'STARRED', 'IMPORTANT', 'SENT', 'DRAFT',
        'CATEGORY_PERSONAL', 'CATEGORY_SOCIAL', 'CATEGORY_PROMOTIONS', 'CATEGORY_UPDATES', 'CATEGORY_FORUMS'
    }
    # Invalid characters for Gmail labels
    INVALID_LABEL_CHARS = set('/\\*?"<>|{}')

    @staticmethod
    def sanitize_label_name(label_name: str) -> Optional[str]:
        """Validate and sanitize a label name according to Gmail rules. Returns sanitized name or None if invalid."""
        if not label_name or not label_name.strip():
            return None
        label_name = label_name.strip()
        # Remove invalid characters
        label_name = ''.join(c for c in label_name if c not in EmailManager.INVALID_LABEL_CHARS)
        # Truncate to 225 chars
        label_name = label_name[:225]
        # Cannot be a system label
        if label_name.upper() in EmailManager.GMAIL_SYSTEM_LABELS:
            return None
        if not label_name:
            return None
        return label_name

    def __init__(self, service):
        self.service = service
        self.label_cache = {}
        self._initialize_labels()
    
    def _initialize_labels(self):
        """ate labels if they don't exist and cache label IDs"""
        logger.info("Initializing Gmail labels...")
        
        try:
            # Get existing labels
            results = self.service.users().labels().list(userId='me').execute()
            existing_labels = {label['name']: label['id'] for label in results.get('labels', [])}
            
            # ate missing labels
            for label_name in config.EMAIL_LABELS.keys():
                logger.info(f"Attempting to create/check label: '{label_name}'")
                if label_name in existing_labels:
                    self.label_cache[label_name] = existing_labels[label_name]
                    logger.info(f"  OK Label exists: {label_name}")
                else:
                    try:
                        label_object = {
                            'name': label_name,
                            'labelListVisibility': 'labelShow',
                            'messageListVisibility': 'show'
                        }
                        created_label = self.service.users().labels().create(
                            userId='me',
                            body=label_object
                        ).execute()
                        self.label_cache[label_name] = created_label['id']
                        logger.info(f"  OK Label created: {label_name}")
                    except Exception as label_error:
                        logger.error(f"ERROR: Failed to create label '{label_name}': {label_error}")
            
            logger.info("OK All labels initialized successfully")
        
        except HttpError as error:
            logger.error(f"ERROR: Error initializing labels: {error}")
            raise
    
    def get_unread_emails(self, max_results: int = 50) -> List[Dict]:
        """Fetch unread emails from inbox"""
        logger.info(f"Fetching up to {max_results} unread emails...")
        
        try:
            # Query for unread emails in inbox
            results = self.service.users().messages().list(
                userId='me',
                q='is:unread in:inbox',
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                logger.info("No unread emails found")
                return []
            
            logger.info(f"Found {len(messages)} unread emails")
            
            # Fetch full email details
            emails = []
            for message in messages:
                email_data = self._get_email_details(message['id'])
                if email_data:
                    emails.append(email_data)
            
            return emails
        
        except HttpError as error:
            logger.error(f"ERROR: Error fetching emails: {error}")
            return []
    
    def _get_email_details(self, message_id: str) -> Optional[Dict]:
        """Get detailed information for a specific email"""
        try:
            message = self.service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()
            
            # Extract headers
            headers = message['payload'].get('headers', [])
            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
            from_email = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown')
            
            # Extract body
            body = self._extract_body(message['payload'])
            body_preview = body[:config.EMAIL_BODY_PREVIEW_LENGTH] if body else "No content"
            
            return {
                'id': message_id,
                'subject': subject,
                'from': from_email,
                'body_preview': body_preview,
                'snippet': message.get('snippet', '')
            }
        
        except HttpError as error:
            logger.error(f"ERROR: Error getting email details for {message_id}: {error}")
            return None
    
    def _extract_body(self, payload: Dict) -> str:
        """Extract email body from payload"""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
                elif part['mimeType'] == 'text/html' and not body:
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
        elif 'body' in payload and 'data' in payload['body']:
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        
        return body
    
    def create_label_if_not_exists(self, label_name: str) -> Optional[str]:
        """Create a new Gmail label if it does not exist. Returns label ID or None."""
        # Sanitize and validate label name
        sanitized_label = self.sanitize_label_name(label_name)
        if not sanitized_label:
            logger.error(f"ERROR: Invalid label name (does not comply with Gmail rules): '{label_name}'")
            return None
        label_name = sanitized_label
        if label_name in self.label_cache:
            return self.label_cache[label_name]
        try:
            # Check if label already exists in Gmail
            results = self.service.users().labels().list(userId='me').execute()
            existing_labels = {label['name']: label['id'] for label in results.get('labels', [])}
            if label_name in existing_labels:
                self.label_cache[label_name] = existing_labels[label_name]
                logger.info(f"OK Label already exists in Gmail: {label_name}")
                return existing_labels[label_name]
            # Create new label
            label_object = {
                'name': label_name,
                'labelListVisibility': 'labelShow',
                'messageListVisibility': 'show'
            }
            created_label = self.service.users().labels().create(
                userId='me',
                body=label_object
            ).execute()
            self.label_cache[label_name] = created_label['id']
            logger.info(f"OK New label created by AI: {label_name}")
            return created_label['id']
        except Exception as e:
            logger.error(f"ERROR: Failed to create label '{label_name}': {e}")
            return None

    def apply_label(self, message_id: str, label_name: str) -> bool:
        """Apply a label to an email. Creates the label if it does not exist."""
        try:
            label_id = self.label_cache.get(label_name)
            if not label_id:
                # Try to create the label dynamically
                label_id = self.create_label_if_not_exists(label_name)
            if not label_id:
                logger.error(f"ERROR: Label not found and could not be created: {label_name}")
                return False
            
            # Prepare modification request
            modify_request = {
                'addLabelIds': [label_id]
            }
            
            # Add mark as read if configured
            if config.MARK_AS_READ:
                modify_request['removeLabelIds'] = ['UNREAD']
            
            # Add archive if configured
            if config.ARCHIVE_AFTER_CLASSIFICATION:
                modify_request['removeLabelIds'] = modify_request.get('removeLabelIds', []) + ['INBOX']
            
            # Apply modifications
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body=modify_request
            ).execute()
            
            return True
        
        except HttpError as error:
            logger.error(f"ERROR: Error applying label to {message_id}: {error}")
            return False


# ============================================
# MAIN BOT ORCHESTRATOR
# ============================================
class EmailClassifierBot:
    """Main orchestrator for the email classification bot"""
    
    def __init__(self):
        self.authenticator = GmailAuthenticator()
        self.gmail_service = None
        self.classifier = None
        self.email_manager = None
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'by_label': {}
        }
    
    def initialize(self):
        """Initialize all components"""
        logger.info("=" * 60)
        logger.info("START Initializing Email Classifier Bot")
        logger.info("=" * 60)
        
        # Authenticate Gmail
        self.gmail_service = self.authenticator.authenticate()
        
        # Initialize classifier
        self.classifier = EmailClassifier()
        
        # Initialize email manager
        self.email_manager = EmailManager(self.gmail_service)
        
        logger.info("OK Bot initialization complete!")
        logger.info("=" * 60)
    
    def process_emails(self):
        """Process unread emails and classify them"""
        logger.info("\n" + "=" * 60)
        logger.info(f"EMAIL: Starting email processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        # Fetch unread emails
        emails = self.email_manager.get_unread_emails(config.MAX_EMAILS_PER_RUN)

        if not emails:
            logger.info("OK No emails to process")
            return

        # Process each email
        for idx, email in enumerate(emails, 1):
            logger.info(f"\nProcessing email {idx}/{len(emails)}")
            logger.info(f"  Subject: {email['subject'][:60]}...")
            logger.info(f"  From: {email['from']}")

            # Classify email
            classification = self.classifier.classify_email(email)

            if classification:
                # Apply label
                success = self.email_manager.apply_label(email['id'], classification)

                if success:
                    self.stats['successful'] += 1
                    self.stats['by_label'][classification] = self.stats['by_label'].get(classification, 0) + 1
                    logger.info(f"  ✓ Labeled as: {classification}")

                    # (Urgent notification integrations removed)
                else:
                    self.stats['failed'] += 1
                    logger.error(f"  ❌ Failed to apply label")
            else:
                self.stats['failed'] += 1
                logger.error(f"  ❌ Classification failed")

            self.stats['total_processed'] += 1

            # Small delay to avoid rate limits
            time.sleep(0.5)

        self._print_stats()
    
    def _print_stats(self):
        """Print processing statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 Processing Statistics")
        logger.info("=" * 60)
        logger.info(f"Total Processed: {self.stats['total_processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        
        if self.stats['by_label']:
            logger.info("\nClassifications by Label:")
            for label, count in sorted(self.stats['by_label'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {label}: {count}")
        
        logger.info("=" * 60)
    
    def run_once(self):
        """Run the bot once"""
        self.initialize()
        self.process_emails()
        logger.info("\nOK Single run completed")
    
    def run_continuous(self):
        """Run the bot continuously"""
        self.initialize()
        
        logger.info(f"\nStarting continuous mode (checking every {config.CHECK_INTERVAL_SECONDS} seconds)")
        logger.info("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.process_emails()
                
                logger.info(f"\nWaiting {config.CHECK_INTERVAL_SECONDS} seconds until next check...")
                time.sleep(config.CHECK_INTERVAL_SECONDS)
        
        except KeyboardInterrupt:
            logger.info("\n\nBot stopped by user")
            self._print_stats()


# ============================================
# MAIN ENTRY POINT
# ============================================
def main():
    """Main entry point for the application"""
    try:
        bot = EmailClassifierBot()
        
        if config.CONTINUOUS_MODE:
            bot.run_continuous()
        else:
            bot.run_once()
    
    except Exception as e:
        logger.error(f"ERROR: Fatal error: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
