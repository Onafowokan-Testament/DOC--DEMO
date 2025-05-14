import logging
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from dotenv import load_dotenv
from medical_bot_core import get_medical_bot

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get the medical bot instance
medical_bot = get_medical_bot()

class TelegramMedicalBot:
    def __init__(self, token: str):
        self.token = token
        self.app = Application.builder().token(token).build()
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up all command and message handlers"""
        # Command handlers
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("clear", self.clear_conversation))
        self.app.add_handler(CommandHandler("emergency", self.emergency_info))
        self.app.add_handler(CommandHandler("disclaimer", self.medical_disclaimer))
        
        # Message handler for medical queries
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Callback handler for inline buttons
        self.app.add_handler(CallbackQueryHandler(self.button_callback))
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a message when the command /start is issued."""
        user = update.effective_user
        welcome_text = f"""
👋 Hello {user.first_name}! Welcome to the Medical Assistant Bot.

I'm here to help you with basic medical questions and provide general health information. 

⚠️ **Important Disclaimer**: This bot is not a substitute for professional medical advice. Always consult a healthcare provider for serious health concerns.

**What I can help with:**
• General health questions
• Symptom information
• Basic medical guidance
• Emergency contact information

**Commands:**
/help - Show available commands
/clear - Clear conversation history
/emergency - Emergency contact information
/disclaimer - Important medical disclaimers

**For medical emergencies, call 911 immediately!**

Please describe your symptoms or ask your health-related question.
"""
        await update.message.reply_text(welcome_text)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a list of available commands"""
        help_text = """
🔍 **Available Commands:**

/start - Start the bot and see welcome message
/help - Show this help message
/clear - Clear conversation history
/emergency - Get emergency contact information
/disclaimer - View medical disclaimers

**How to use:**
Simply type your symptoms or health questions, and I'll do my best to provide helpful information.

**Examples:**
- "I have a headache for 2 days"
- "What should I do about a fever?"
- "I'm experiencing chest pain"

⚠️ Remember: This bot is for informational purposes only. Always seek professional medical help for serious health concerns.
"""
        await update.message.reply_text(help_text)
    
    async def clear_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Clear the conversation history for the user"""
        user_id = str(update.effective_user.id)
        
        # Create inline keyboard for confirmation
        keyboard = [
            [
                InlineKeyboardButton("Yes, clear history", callback_data=f"clear_yes_{user_id}"),
                InlineKeyboardButton("No, keep history", callback_data=f"clear_no_{user_id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "Are you sure you want to clear your conversation history?",
            reply_markup=reply_markup
        )
    
    async def emergency_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Provide emergency contact information"""
        emergency_text = """
🚨 **EMERGENCY CONTACTS** 🚨

**Immediate Emergency:**
• 911 - General Emergency Services

**Crisis Hotlines:**
• 988 - National Suicide Prevention Lifeline
• 1-800-222-1222 - Poison Control
• Text HOME to 741741 - Crisis Text Line

**When to call 911:**
• Chest pain or difficulty breathing
• Severe bleeding or injuries
• Loss of consciousness
• Signs of stroke (FAST: Face, Arms, Speech, Time)
• Severe allergic reactions
• Suicidal thoughts or behavior

⚠️ **If you're experiencing a medical emergency, don't use this bot - call 911 immediately!**
"""
        await update.message.reply_text(emergency_text)
    
    async def medical_disclaimer(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show medical disclaimers"""
        disclaimer_text = """
⚕️ **MEDICAL DISCLAIMER** ⚕️

**This bot is NOT a substitute for professional medical advice, diagnosis, or treatment.**

**Important Points:**
• This AI assistant provides general health information only
• It cannot diagnose medical conditions
• It cannot prescribe medications
• It should not be used for emergencies

**Always consult a healthcare professional for:**
• Persistent or worsening symptoms
• Any concerning health issues
• Medication questions
• Serious health conditions

**For emergencies, call 911 immediately!**

By using this bot, you acknowledge that:
• The information is for educational purposes only
• You understand the limitations of AI medical advice
• You will seek professional help when needed
"""
        await update.message.reply_text(disclaimer_text)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages"""
        user_id = str(update.effective_user.id)
        message_text = update.message.text
        
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            # Process the message through the medical bot
            response = medical_bot.process_message(user_id, message_text)
            
            # Send the response
            await update.message.reply_text(response)
            
        except Exception as e:
            logger.error(f"Error processing message for user {user_id}: {e}")
            error_message = """
Sorry, I encountered an error while processing your message. Please try again.

If you're experiencing a medical emergency, please call 911 immediately instead of using this bot.
"""
            await update.message.reply_text(error_message)
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        # Parse callback data
        action = query.data
        
        if action.startswith("clear_yes_"):
            user_id = action.split("_")[-1]
            # Clear the conversation history
            medical_bot.clear_conversation(user_id)
            await query.edit_message_text("✅ Conversation history has been cleared.")
            
        elif action.startswith("clear_no_"):
            await query.edit_message_text("Conversation history kept unchanged.")
    
    def run(self):
        """Start the bot"""
        logger.info("Starting Medical Assistant Bot...")
        self.app.run_polling()


def main():
    """Main function to run the bot"""
    # Get Telegram bot token from environment
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not telegram_token:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables!")
        print("Please set TELEGRAM_BOT_TOKEN in your .env file")
        return
    
    # Create and run the bot
    bot = TelegramMedicalBot(telegram_token)
    bot.run()


if __name__ == "__main__":
    main()