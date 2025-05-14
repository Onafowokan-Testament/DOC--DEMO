import os

os.environ["STREAMLIT_SERVER_FILE_WATCHER"] = "false"
import logging
import uuid
from datetime import datetime
import json

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# Import our medical bot components
from graph import medical_bot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
.main-header {
    font-size: 2.5rem;
    color: #2E8B57;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
.user-message {
    background-color: #E8F4F8;
    border-left: 5px solid #2196F3;
}
.bot-message {
    background-color: #F0F8F0;
    border-left: 5px solid #4CAF50;
}
.emergency-alert {
    background-color: #FFE6E6;
    color: #D32F2F;
    padding: 1rem;
    border-radius: 5px;
    border: 2px solid #F44336;
    margin: 1rem 0;
}
.warning-alert {
    background-color: #FFF3E0;
    color: #F57C00;
    padding: 1rem;
    border-radius: 5px;
    border: 2px solid #FF9800;
    margin: 1rem 0;
}
.sidebar-info {
    background-color: #F5F5F5;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state with proper defaults
def initialize_session_state():
    """Initialize session state with proper defaults"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(
                content="Hello! I'm your medical assistant. How can I help you today?"
            )
        ]
    
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    
    if "bot_state" not in st.session_state:
        st.session_state.bot_state = {
            "messages": [],
            "current_input": "",
            "symptoms": [],
            "severity_level": "",
            "extracted_info": {},
            "emergency_keywords": [],
            "high_risk_symptoms": [],
            "documents": [],
            "medical_response": "",
            "follow_up_questions": [],
            "conversation_active": True,
        }
    
    # Ensure bot_state messages are in sync with session messages
    if "messages" in st.session_state.bot_state:
        st.session_state.bot_state["messages"] = st.session_state.messages.copy()

# Function to serialize messages for storage
def serialize_messages(messages):
    """Convert messages to JSON-serializable format"""
    serialized = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            serialized.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            serialized.append({"type": "ai", "content": msg.content})
    return serialized

# Function to deserialize messages from storage
def deserialize_messages(serialized_messages):
    """Convert JSON-serializable format back to messages"""
    messages = []
    for msg in serialized_messages:
        if msg["type"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
    return messages

# Initialize session state
initialize_session_state()

# Sidebar
with st.sidebar:
    st.markdown("### 🏥 AI Medical Assistant")
    st.markdown("---")

    st.markdown(
        """
    <div class="sidebar-info">
    <h4>Important Notice</h4>
    <p>This AI assistant is for informational purposes only and does not replace professional medical advice.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Emergency contacts
    st.markdown("### 🚨 Emergency Contacts")
    st.markdown(
        """
    - **Emergency**: 911
    - **Poison Control**: 1-800-222-1222
    - **Suicide Prevention**: 988
    - **Crisis Text**: Text HOME to 741741
    """
    )

    st.markdown("---")

    # Conversation management
    if st.button("🔄 New Conversation"):
        # Clear session state for new conversation
        st.session_state.messages = [
            AIMessage(
                content="Hello! I'm your medical assistant. How can I help you today?"
            )
        ]
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.bot_state = {
            "messages": st.session_state.messages.copy(),
            "current_input": "",
            "symptoms": [],
            "severity_level": "",
            "extracted_info": {},
            "emergency_keywords": [],
            "high_risk_symptoms": [],
            "documents": [],
            "medical_response": "",
            "follow_up_questions": [],
            "conversation_active": True,
        }
        st.rerun()

    if st.button("📋 Export Conversation"):
        if st.session_state.messages:
            conversation_text = ""
            for msg in st.session_state.messages:
                role = "Patient" if isinstance(msg, HumanMessage) else "AI Assistant"
                conversation_text += f"{role}: {msg.content}\n\n"

            st.download_button(
                label="Download Conversation",
                data=conversation_text,
                file_name=f"medical_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

    # Debug information (remove in production)
    with st.expander("🔍 Debug Info"):
        st.write(f"Conversation ID: {st.session_state.conversation_id}")
        st.write(f"Number of messages: {len(st.session_state.messages)}")
        st.write(f"Bot state active: {st.session_state.bot_state.get('conversation_active', True)}")
        
        # Show serialized state for debugging
        if st.button("Show Serialized State"):
            st.json(serialize_messages(st.session_state.messages))

    st.markdown("---")

    # App info
    st.markdown("### ℹ️ About")
    st.markdown(
        """
    - **Version**: 1.0.0
    - **Last Updated**: 2024
    - **Purpose**: Medical symptom discussion and guidance
    """
    )

# Main content
st.markdown(
    '<h1 class="main-header">🏥 AI Medical Assistant</h1>', unsafe_allow_html=True
)

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        st.markdown(
            f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {message.content}
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        # Check for emergency/warning content
        content = message.content
        if "🚨" in content or "EMERGENCY" in content.upper() or "911" in content:
            st.markdown(
                f"""
            <div class="emergency-alert">
                <strong>🚨 AI Assistant:</strong> {content}
            </div>
            """,
                unsafe_allow_html=True,
            )
        elif "⚠️" in content or "URGENT" in content.upper() or "RECOMMEND" in content:
            st.markdown(
                f"""
            <div class="warning-alert">
                <strong>⚠️ AI Assistant:</strong> {content}
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="chat-message bot-message">
                <strong>AI Assistant:</strong> {content}
            </div>
            """,
                unsafe_allow_html=True,
            )

# Chat input
user_input = st.chat_input("Describe your symptoms or health concerns...")

if user_input:
    # Add user message to session state
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Update bot state with new message - ensure proper sync
    st.session_state.bot_state["messages"] = st.session_state.messages.copy()
    st.session_state.bot_state["current_input"] = user_input

    # Show loading spinner
    with st.spinner("Analyzing your symptoms..."):
        try:
            # Create a copy of bot state for the invocation
            current_state = st.session_state.bot_state.copy()
            
            # Invoke the medical bot with current state
            result = medical_bot.invoke(
                current_state,
                config={
                    "configurable": {"thread_id": st.session_state.conversation_id},
                    "recursion_limit": 100,
                },
            )

            # Update the bot state with the result
            st.session_state.bot_state.update(result)

            # Extract new messages from the result
            if "messages" in result:
                # Get only new messages that aren't already in session state
                existing_message_contents = [msg.content for msg in st.session_state.messages]
                for msg in result["messages"]:
                    if isinstance(msg, AIMessage) and msg.content not in existing_message_contents:
                        st.session_state.messages.append(msg)

            # Update conversation active status
            if not result.get("conversation_active", True):
                st.session_state.messages.append(
                    AIMessage(
                        content="Thank you for chatting. Wishing you good health! If you have more questions later, feel free to ask."
                    )
                )

        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            st.error(
                "I'm sorry, I encountered an error processing your request. Please try again."
            )
            st.session_state.messages.append(
                AIMessage(
                    content="I'm sorry, I encountered an error. Could you please rephrase your question?"
                )
            )

    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666;">
    <p>🏥 AI Medical Assistant | For informational purposes only</p>
    <p>Always consult with healthcare professionals for medical advice</p>
</div>
""",
    unsafe_allow_html=True,
)

# Connection status (bottom right)
with st.sidebar:
    st.markdown("---")
    if st.button("🔍 Test Connection"):
        try:
            # Simple test to see if bot is available
            if medical_bot:
                st.success("✅ Bot connected successfully!")
            else:
                st.error("❌ Bot not initialized")
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")

# Usage instructions
with st.expander("📖 How to Use This Medical Assistant"):
    st.markdown(
        """
    ### Getting Started:
    1. **Describe your symptoms** in the chat box below
    2. **Be specific** about your concerns (location, duration, severity)
    3. **Follow the assistant's questions** for better assessment
    
    ### Emergency Situations:
    - If you have a **medical emergency**, call 911 immediately
    - The bot will recognize emergency keywords and provide instant guidance
    
    ### Important Notes:
    - This AI provides **information only**, not medical diagnoses
    - Always **consult healthcare professionals** for serious concerns
    - Keep your **medical information private** - don't share sensitive details
    
    ### Features:
    - Emergency keyword detection
    - Symptom analysis and guidance
    - Doctor referral recommendations
    - Conversation export for your records
    """
        )
