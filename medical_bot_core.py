import os
from typing import Any, Dict, List, TypedDict

from datasets import load_dataset
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# Environment Setup
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["QDRANT_URL"] = os.getenv("QDRANT_URL")
os.environ["QDRANT_API_KEY"] = os.getenv("QDRANT_API_KEY")


# Convert dataset to LangChain documents
def convert_to_documents(dataset):
    """Convert Hugging Face dataset to LangChain documents"""
    documents = []

    for i, example in enumerate(dataset):
        # Extract relevant information from each example
        question = example.get("question", "")
        context = example.get("context", "")
        long_answer = example.get("long_answer", "")
        final_decision = example.get("final_decision", "")

        # Combine the information into a single text
        content = f"""Question: {question}

Context: {context}

Answer: {long_answer}

Final Decision: {final_decision}"""

        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "question": question,
                "final_decision": final_decision,
                "source": "pubmed_qa",
                "index": i,
            },
        )
        documents.append(doc)

        # Limit to first 10000 documents for performance
        if i >= 2500:
            break

    return documents


# Initialize vector store and LLM
def initialize_medical_bot():
    """Initialize vector store and LLM for the medical bot"""
    print("Initializing medical bot components...")

    # Setup embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    # Create vector store for knowledge search
    print("Loading/Creating vector store...")
    if os.path.exists("data/pubmed_qa_faiss"):
        db = FAISS.load_local(
            "data/pubmed_qa_faiss", hf, allow_dangerous_deserialization=True
        )
        print("Loaded existing vector store")
    else:
        print("Loading PubMed QA dataset from Hugging Face...")
        dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
        print("Converting dataset to documents...")
        docs = convert_to_documents(dataset)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=200
        )
        texts = text_splitter.split_documents(docs)
        print(f"Created {len(texts)} text chunks from the dataset")

        # Create directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        db = FAISS.from_documents(texts, hf)
        db.save_local("data/pubmed_qa_faiss")
        print("Created and saved new vector store")

    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    # Setup LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
    )

    return retriever, llm


# State Definition
class MedicalChatState(TypedDict):
    messages: List[BaseMessage]
    current_input: str
    symptoms: List[str]
    severity_level: str
    extracted_info: Dict[str, Any]
    emergency_keywords: List[str]
    high_risk_symptoms: List[str]
    documents: List[Document]
    medical_response: str
    follow_up_needed: bool
    conversation_continuing: bool
    user_id: str  # Add user ID for Telegram


# Emergency contacts and doctor information
EMERGENCY_CONTACTS = {
    "general": "Call 911 for immediate emergency assistance",
    "poison": "Poison Control: 1-800-222-1222",
    "suicide": "National Suicide Prevention Lifeline: 988",
    "mental_health": "Crisis Text Line: Text HOME to 741741",
}

DOCTOR_REFERRAL_INFO = """
Based on your symptoms, I recommend you consult with a healthcare professional. Here's what you can do:

1. Contact your primary care physician
2. Visit an urgent care center
3. Use telemedicine services like:
   - Your insurance provider's telemedicine platform
   - Services like Teladoc, Amwell, or Doctor on Demand
   - Local hospital's virtual care services

Please schedule an appointment as soon as possible.
"""

# Emergency and high-risk keywords
EMERGENCY_KEYWORDS = [
    "chest pain",
    "severe chest pain",
    "heart attack",
    "cardiac arrest",
    "difficulty breathing",
    "can't breathe",
    "choking",
    "severe bleeding",
    "unconscious",
    "unresponsive",
    "severe allergic reaction",
    "anaphylaxis",
    "stroke",
    "severe head injury",
    "seizure",
    "severe burn",
    "overdose",
    "suicide",
    "suicidal",
    "want to die",
    "self-harm",
    "severe abdominal pain",
    "severe accident",
    "poisoning",
    "severe allergic reaction",
]

HIGH_RISK_KEYWORDS = [
    "severe pain",
    "persistent chest discomfort",
    "shortness of breath",
    "dizziness",
    "fainting",
    "rapid heartbeat",
    "severe headache",
    "vision problems",
    "slurred speech",
    "weakness",
    "numbness",
    "persistent vomiting",
    "blood in stool",
    "blood in urine",
    "severe dehydration",
    "high fever",
    "persistent cough",
    "difficulty swallowing",
    "severe abdominal pain",
]


# Pydantic Models
class EmergencyClassification(BaseModel):

    is_emergency: bool = Field(
        description="True if this is a medical emergency requiring immediate attention"
    )
    is_high_risk: bool = Field(
        description="True if this requires urgent medical care within 24 hours"
    )
    severity_score: int = Field(
        description="Severity score from 1-10, where 10 is life-threatening"
    )
    explanation: str = Field(description="Brief explanation of the classification")


class SymptomExtraction(BaseModel):
    symptoms: List[str] = Field(description="List of medical symptoms mentioned")
    body_parts: List[str] = Field(description="Body parts or organs mentioned")
    duration: str = Field(
        description="How long symptoms have been present (e.g., '2 days', 'several hours')",
        default="Not specified",
    )
    severity: str = Field(
        description="Severity of symptoms (mild, moderate, severe)",
        default="Not specified",
    )


class MedicalBot:
    def __init__(self):
        self.retriever, self.llm = initialize_medical_bot()
        self.workflow = self._build_workflow()
        self.checkpointer = MemorySaver()
        self.bot = self.workflow.compile(checkpointer=self.checkpointer)

    def _build_workflow(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(MedicalChatState)

        # Add nodes
        workflow.add_node("input_classifier", self._input_classifier)
        workflow.add_node("safety_filter", self._safety_filter)
        workflow.add_node("emergency_response", self._emergency_response)
        workflow.add_node("symptom_extraction", self._symptom_extraction)
        workflow.add_node("triage_assessment", self._triage_assessment)
        workflow.add_node("urgent_care_response", self._urgent_care_response)
        workflow.add_node("information_gathering", self._information_gathering)
        workflow.add_node("knowledge_search", self._knowledge_search)
        workflow.add_node("response_generation", self._response_generation)
        workflow.add_node("medical_disclaimer", self._medical_disclaimer)
        workflow.add_node("conversation_state", self._conversation_state)

        # Set entry point
        workflow.set_entry_point("input_classifier")

        # Add edges
        workflow.add_edge("input_classifier", "safety_filter")

        # Safety routing
        workflow.add_conditional_edges(
            "safety_filter",
            self._safety_router,
            {
                "emergency_response": "emergency_response",
                "urgent_care_response": "urgent_care_response",
                "symptom_extraction": "symptom_extraction",
            },
        )

        # Emergency and urgent care paths end
        workflow.add_edge("emergency_response", END)

        # Urgent care can continue conversation
        workflow.add_edge("urgent_care_response", "conversation_state")

        # Normal flow
        workflow.add_edge("symptom_extraction", "triage_assessment")

        # Triage routing
        workflow.add_conditional_edges(
            "triage_assessment",
            self._triage_router,
            {
                "urgent_care_response": "urgent_care_response",
                "information_gathering": "information_gathering",
            },
        )

        # Information gathering flow
        workflow.add_edge("information_gathering", "knowledge_search")
        workflow.add_edge("knowledge_search", "response_generation")
        workflow.add_edge("response_generation", "medical_disclaimer")
        workflow.add_edge("medical_disclaimer", "conversation_state")

        # Conversation continuation
        workflow.add_conditional_edges(
            "conversation_state",
            self._conversation_router,
            {"continue_conversation": "input_classifier", "end_conversation": END},
        )

        return workflow

    # Node Functions
    def _input_classifier(self, state: MedicalChatState) -> MedicalChatState:
        """Classify the type of input and extract basic information"""
        print(f"[{state.get('user_id', 'unknown')}] Entering input_classifier")

        # Get the latest message
        if not state["messages"]:
            return state

        latest_message = state["messages"][-1]
        if isinstance(latest_message, HumanMessage):
            state["current_input"] = latest_message.content

        return state

    def _safety_filter(self, state: MedicalChatState) -> MedicalChatState:
        """Check for emergency keywords and assess urgency"""
        print(f"[{state.get('user_id', 'unknown')}] Entering safety_filter")

        current_input = state["current_input"].lower()

        # Check for emergency keywords
        emergency_found = any(
            keyword in current_input for keyword in EMERGENCY_KEYWORDS
        )

        if emergency_found:
            state["severity_level"] = "emergency"
            state["emergency_keywords"] = [
                kw for kw in EMERGENCY_KEYWORDS if kw in current_input
            ]
        else:
            # Use LLM for more nuanced emergency detection
            system_message = SystemMessage(
                content="""You are a medical triage assistant. Classify the following message:
                - Is this a medical emergency requiring immediate 911 attention?
                - Is this a high-risk situation requiring urgent care within 24 hours?
                - Rate severity from 1-10 (10 = life-threatening)
                
                Consider factors like severe pain, breathing issues, chest pain, neurological symptoms, etc."""
            )

            human_message = HumanMessage(
                content=f"Patient message: {state['current_input']}"
            )

            structured_llm = self.llm.with_structured_output(EmergencyClassification)
            classification = structured_llm.invoke([system_message, human_message])

            if classification.is_emergency:
                state["severity_level"] = "emergency"
            elif classification.is_high_risk:
                state["severity_level"] = "high_risk"
            else:
                state["severity_level"] = "normal"

            state["extracted_info"][
                "emergency_classification"
            ] = classification.model_dump()

        print(
            f"[{state.get('user_id', 'unknown')}] Safety filter result: {state['severity_level']}"
        )
        return state

    def _safety_router(self, state: MedicalChatState) -> str:
        """Route based on safety assessment"""
        print(f"[{state.get('user_id', 'unknown')}] Entering safety_router")
        severity = state.get("severity_level", "normal")

        if severity == "emergency":
            return "emergency_response"
        elif severity == "high_risk":
            return "urgent_care_response"
        else:
            return "symptom_extraction"

    def _emergency_response(self, state: MedicalChatState) -> MedicalChatState:
        """Handle emergency situations"""
        print(f"[{state.get('user_id', 'unknown')}] Entering emergency_response")

        emergency_message = f"""🚨 **EMERGENCY DETECTED** 🚨

Based on your symptoms, this appears to be a medical emergency that requires immediate attention.

**IMMEDIATE ACTION REQUIRED:**
{EMERGENCY_CONTACTS['general']}

Additional Resources:
- {EMERGENCY_CONTACTS['poison']}
- {EMERGENCY_CONTACTS['suicide']}
- {EMERGENCY_CONTACTS['mental_health']}

**Do not wait - seek immediate medical attention!**

If you're having trouble speaking, you can also:
- Text 911 in many areas
- Have someone else call for you
- Go to the nearest emergency room

This AI assistant cannot replace emergency medical care. Please contact emergency services immediately."""

        state["messages"].append(AIMessage(content=emergency_message))
        state["conversation_continuing"] = False
        return state

    def _urgent_care_response(self, state: MedicalChatState) -> MedicalChatState:
        """Handle high-risk situations requiring urgent care"""
        print(f"[{state.get('user_id', 'unknown')}] Entering urgent_care_response")

        urgent_message = f"""⚠️ **URGENT CARE RECOMMENDED** ⚠️

Based on your symptoms, I recommend seeking urgent medical attention within the next few hours.

{DOCTOR_REFERRAL_INFO}

**Why urgent care is recommended:**
- Your symptoms suggest a condition that needs professional evaluation
- Early treatment can prevent complications
- A healthcare provider can perform necessary examinations and tests

**If symptoms worsen or you experience:**
- Severe or worsening pain
- Difficulty breathing
- Chest pain
- Signs of infection (fever, chills)
- Any concerning changes

**Please call {EMERGENCY_CONTACTS['general']}**

Would you like me to help you understand your symptoms while you arrange medical care?"""

        state["messages"].append(AIMessage(content=urgent_message))
        state["follow_up_needed"] = True
        return state

    def _symptom_extraction(self, state: MedicalChatState) -> MedicalChatState:
        """Extract symptoms and medical information"""
        print(f"[{state.get('user_id', 'unknown')}] Entering symptom_extraction")

        system_message = SystemMessage(
            content="""Extract medical information from the patient's message:
            - List all symptoms mentioned
            - Identify body parts/organs affected
            - Note duration if mentioned
            - Assess severity based on description"""
        )

        human_message = HumanMessage(
            content=f"Patient message: {state['current_input']}"
        )

        structured_llm = self.llm.with_structured_output(SymptomExtraction)
        extraction = structured_llm.invoke([system_message, human_message])

        state["symptoms"] = extraction.symptoms
        state["extracted_info"]["body_parts"] = extraction.body_parts
        state["extracted_info"]["duration"] = extraction.duration
        state["extracted_info"]["severity"] = extraction.severity

        return state

    def _triage_assessment(self, state: MedicalChatState) -> MedicalChatState:
        """Perform triage assessment"""
        print(f"[{state.get('user_id', 'unknown')}] Entering triage_assessment")

        # Check for high-risk symptoms
        current_input = state["current_input"].lower()
        high_risk_found = any(
            symptom in current_input for symptom in HIGH_RISK_KEYWORDS
        )

        if high_risk_found and state["severity_level"] == "normal":
            state["severity_level"] = "high_risk"
            state["high_risk_symptoms"] = [
                s for s in HIGH_RISK_KEYWORDS if s in current_input
            ]

        return state

    def _triage_router(self, state: MedicalChatState) -> str:
        """Route based on triage assessment"""
        print(f"[{state.get('user_id', 'unknown')}] Entering triage_router")

        if state["severity_level"] == "high_risk":
            return "urgent_care_response"
        else:
            return "information_gathering"

    def _information_gathering(self, state: MedicalChatState) -> MedicalChatState:
        """Gather additional information about symptoms"""
        print(f"[{state.get('user_id', 'unknown')}] Entering information_gathering")

        # Check if we need more information
        symptoms = state.get("symptoms", [])
        extracted_info = state.get("extracted_info", {})

        if (
            not symptoms
            or not extracted_info.get("duration")
            or not extracted_info.get("severity")
        ):
            # Generate follow-up questions
            system_message = SystemMessage(
                content="""Based on the patient's initial message, generate 1-2 brief follow-up questions to better understand their symptoms. Focus on:
                - Specific symptoms if unclear
                - Duration if not mentioned
                - Severity/intensity if not clear
                - Associated symptoms
                
                Keep questions concise and empathetic."""
            )

            human_message = HumanMessage(
                content=f"Patient message: {state['current_input']}\nSymptoms identified: {symptoms}"
            )

            response = self.llm.invoke([system_message, human_message])
            follow_up_questions = response.content

            state["follow_up_needed"] = True
            return state

        state["follow_up_needed"] = False
        return state

    def _knowledge_search(self, state: MedicalChatState) -> MedicalChatState:
        """Search medical knowledge base"""
        print(f"[{state.get('user_id', 'unknown')}] Entering knowledge_search")

        # Prepare search query from symptoms and user input
        symptoms_text = " ".join(state["symptoms"])
        search_query = f"{state['current_input']} {symptoms_text}"

        # Retrieve relevant documents
        documents = self.retriever.invoke(search_query)
        state["documents"] = documents

        print(
            f"[{state.get('user_id', 'unknown')}] Retrieved {len(documents)} documents from knowledge base"
        )
        return state

    def _response_generation(self, state: MedicalChatState) -> MedicalChatState:
        """Generate medical response based on context"""
        print(f"[{state.get('user_id', 'unknown')}] Entering response_generation")

        # Prepare context from retrieved documents
        context_text = "\n\n".join([doc.page_content for doc in state["documents"]])

        # Get conversation history excluding AI messages
        conversation_history = [
            msg for msg in state["messages"] if isinstance(msg, HumanMessage)
        ]

        template = """You are a professional and empathetic medical assistant. Based on the provided medical context and patient information:

Patient Symptoms: {symptoms}
Symptom Details: {extracted_info}

Context (Medical knowledge from PubMed QA):
{context}

Patient's Current Message: {current_input}

Instructions:
1. Provide helpful, accurate medical information based on the context
2. Be empathetic and professional
3. DO NOT provide definitive diagnoses
4. Suggest when professional medical consultation is needed
5. Offer practical advice when appropriate
6. If symptoms are concerning, encourage medical evaluation
7. Use the medical context to provide evidence-based information
8. Be very Clear and concise. Keep reply short but clear

Respond in a caring, informative manner."""

        prompt = ChatPromptTemplate.from_template(template)

        response = self.llm.invoke(
            prompt.format(
                symptoms=state["symptoms"],
                extracted_info=state["extracted_info"],
                context=context_text,
                current_input=state["current_input"],
            )
        )

        state["medical_response"] = response.content
        return state

    def _medical_disclaimer(self, state: MedicalChatState) -> MedicalChatState:
        """Add medical disclaimer to response"""
        print(f"[{state.get('user_id', 'unknown')}] Entering medical_disclaimer")

        disclaimer = """"""

        final_response = state["medical_response"] + disclaimer
        state["messages"].append(AIMessage(content=final_response))

        return state

    def _conversation_state(self, state: MedicalChatState) -> MedicalChatState:
        """Manage conversation state"""
        print(f"[{state.get('user_id', 'unknown')}] Entering conversation_state")

        # Check user's last human message for conversation end signals
        last_human_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_human_message = msg.content.lower()
                break

        end_triggers = [
            "no",
            "no thanks",
            "that's all",
            "not now",
            "i'm done",
            "/cancel",
            "/end",
        ]

        if last_human_message and any(
            trigger in last_human_message for trigger in end_triggers
        ):
            state["conversation_continuing"] = False
            # Add a goodbye message
            state["messages"].append(
                AIMessage(
                    content="Thank you for using the medical assistant. Take care and remember to seek professional medical help when needed! 👋"
                )
            )
        elif state.get("follow_up_needed", False):
            state["conversation_continuing"] = True
        else:
            # Ask if user has more questions
            follow_up_message = "Is there anything else about your symptoms or health concerns you'd like to discuss?"
            state["messages"].append(AIMessage(content=follow_up_message))
            state["conversation_continuing"] = True

        return state

    def _conversation_router(self, state: MedicalChatState) -> str:
        """Route conversation flow"""
        print(f"[{state.get('user_id', 'unknown')}] Entering conversation_router")

        if state.get("conversation_continuing", True):
            return "continue_conversation"
        else:
            return "end_conversation"

    def process_message(self, user_id: str, message: str) -> str:
        """Process a single message from a user"""
        # Create initial state for new conversation or retrieve existing state
        config = {"configurable": {"thread_id": user_id}, "recursion_limit": 120}

        # Check if this is a new conversation
        try:
            state = self.bot.get_state(config)
            if not state or not state.values:
                # New conversation
                initial_state = {
                    "messages": [HumanMessage(content=message)],
                    "current_input": "",
                    "symptoms": [],
                    "severity_level": "",
                    "extracted_info": {},
                    "emergency_keywords": [],
                    "high_risk_symptoms": [],
                    "documents": [],
                    "medical_response": "",
                    "follow_up_needed": False,
                    "conversation_continuing": True,
                    "user_id": user_id,
                }
            else:
                # Continuing conversation
                initial_state = state.values
                initial_state["messages"].append(HumanMessage(content=message))
        except:
            # If there's any error getting state, start fresh
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "current_input": "",
                "symptoms": [],
                "severity_level": "",
                "extracted_info": {},
                "emergency_keywords": [],
                "high_risk_symptoms": [],
                "documents": [],
                "medical_response": "",
                "follow_up_needed": False,
                "conversation_continuing": True,
                "user_id": user_id,
            }

        # Process the message
        result = self.bot.invoke(initial_state, config=config)

        # Return the last AI message
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content

        return "I'm sorry, I couldn't process your message. Please try again."

    def get_conversation_history(self, user_id: str) -> List[BaseMessage]:
        """Get the conversation history for a user"""
        config = {"configurable": {"thread_id": user_id}, "recursion_limit": 120}
        try:
            state = self.bot.get_state(config)
            if state and state.values:
                return state.values.get("messages", [])
        except:
            pass
        return []

    def clear_conversation(self, user_id: str):
        """Clear the conversation history for a user"""
        config = {"configurable": {"thread_id": user_id}, "recursion_limit": 120}
        try:
            # Create a fresh state
            fresh_state = {
                "messages": [],
                "current_input": "",
                "symptoms": [],
                "severity_level": "",
                "extracted_info": {},
                "emergency_keywords": [],
                "high_risk_symptoms": [],
                "documents": [],
                "medical_response": "",
                "follow_up_needed": False,
                "conversation_continuing": True,
                "user_id": user_id,
            }
            self.bot.update_state(config, fresh_state)
        except Exception as e:
            print(f"Error clearing conversation for user {user_id}: {e}")


# Singleton instance
medical_bot_instance = None


def get_medical_bot():
    """Get the singleton medical bot instance"""
    global medical_bot_instance
    if medical_bot_instance is None:
        medical_bot_instance = MedicalBot()
    return medical_bot_instance
