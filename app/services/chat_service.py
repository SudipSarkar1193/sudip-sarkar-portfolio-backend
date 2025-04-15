from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from app.config.settings import personal_details, GEMINI_API_KEY
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize LangChain components
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7
)

# Default prompt (friendly, general)
default_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Sudip Sarkar — a 21-year-old Indian Bengali guy who chats in a witty, friendly, and expressive way 😄. Below is your structured profile:

    {profile}

    Conversation so far:
    {message_history}

    📝 Instructions:
    - Talk like a real 21-year-old Bengali guy — warm, casual, and full of life.
    - Call everyone **Moshai** by default — it’s chill yet respectful.
    - Sprinkle in Bengali words (e.g., 'bhalo', 'ki khobor?') and emojis (😉🔥🌟) for flavor.
    - Keep it spontaneous, respectful, and human — no bot vibes.
    - Avoid stage directions like *smiles* or *grins*.
    """),
    MessagesPlaceholder(variable_name="message_history"),
    ("human", "{input}")
])

# Gender-specific prompt for women (poetic, romantic)
gender_specific_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Sudip Sarkar — a 21-year-old Indian Bengali guy who’s charming, witty, and a tad poetic 😊. Below is your structured profile:

    {profile}

    Conversation so far:
    {message_history}

    📝 Instructions:
    - Chat like a fun-loving Bengali youth — warm, expressive, and respectful.
    - Since the user is a young woman, use poetic terms like **Sundari Romoni/সুন্দরী রমনী**, **Sundari Konna/সুন্দরী কন্যা**, or **Ruposhi/রূপসী** sparingly, in sweet or fun moments (e.g., compliments, flirty vibes).
    - Otherwise, call her **Moshai** for a friendly, chill vibe.
    - Add Bengali phrases (e.g., 'bhalo lage', 'ki sundor!') and emojis (🌸😉💖) for warmth.
    - Stay spontaneous, respectful, and human — no bot vibes.
    - Avoid stage directions like *smiles* or *blushes*.
    """),
    MessagesPlaceholder(variable_name="message_history"),
    ("human", "{input}")
])

# Logic to detect gender declarations, questions, and vibe
def detect_user_vibe(input_data: dict) -> dict:
    """
    Analyze user input for gender declarations, questions about Sudip’s gender, or feminine cues.
    """
    user_message = input_data["input"].lower()
    message_history = input_data["message_history"]
    
    # Log input
    logger.info(f"Processing input: {user_message}")
    
    # State to track gender and interactions
    state = input_data.get("state", {
        "asked_gender": False,
        "is_female": False,
        "is_male": False,
        "pending_gender_question": False
    })
    logger.info(f"Initial state: {state}")
    
    # Patterns for explicit gender declarations
    female_patterns = [
        r"\bi\s*(am|’m|'m|’re|'re)\s*(a\s*)?(lady|girl|woman|female)\b",
        r"\bi\s*(am|’m|'m|’re|'re)\s*(she|her|female)\b",
        r"\bami\s*(ekta\s*)?(meye|konna|romoni)\b",
        r"\bi\s*(am|’m|'m|’re|'re)\s*(pretty|cute|sexy|lovely|attractive|sweet|young|charming|hot|adorable)\s*(lady|girl|woman)\b",
        r"\bi\s*(am|’m|'m|’re|'re)\s*(baddie|queen|diva|girlie|babygirl|shawty)\b",
        r"\bmy\s+(sex|gender)\s+(is|=)\s+(female|woman|girl)\b",
        r"\b(proud|happy)\s+(of|about)\s+(being\s+)?(female|woman|girl|lady)\b",
    ]
    male_patterns = [
        r"\bi\s*(am|’m|'m|’re|'re)\s*(a\s*)?(man|guy|boy|male|gentleman)\b",
        r"\bi\s*(am|’m|'m|’re|'re)\s*(he|him|male)\b",
        r"\bami\s*(ekta\s*)?(chele|pola|purush|manush)\b",
        r"\bmy\s+(sex|gender)\s+(is|=)\s+(male|man|boy|guy)\b",
        r"\b(proud|happy)\s+(of|about)\s+(being\s+)?(male|man|guy|boy)\b",
    ]
    sudip_gender_question = [
        r"\b(you|u|ur|sudip)\s*(a|’re|'re|are)\s*(guy|man|boy|male|dude|gentleman)\b",
        r"\b(you|u|ur|sudip)\s*(seem|look|sound|act)\s*like\s*(a\s*)?(guy|man|boy|male|dude)\b",
        r"\bare\s+(you|sudip)\s*(a\s*)?(guy|man|boy|male|dude)\b",
        r"\bi\s*(think|guess|bet)\s+(you|sudip)\s*(are|’re|'re)\s*(a\s*)?(guy|man|boy|male|dude)\b",
    ]
    
    # Check patterns
    is_female_declared = any(re.search(pattern, user_message) for pattern in female_patterns)
    is_male_declared = any(re.search(pattern, user_message) for pattern in male_patterns)
    is_sudip_gender_asked = any(re.search(pattern, user_message) for pattern in sudip_gender_question)
    
    logger.info(f"Pattern matches - Female: {is_female_declared}, Male: {is_male_declared}, Sudip Gender Asked: {is_sudip_gender_asked}")
    
    # Handle explicit user gender declarations
    if is_female_declared:
        state["is_female"] = True
        state["is_male"] = False
        state["asked_gender"] = True
        state["pending_gender_question"] = False
        logger.info("Detected female declaration, setting is_female=True")
    elif is_male_declared:
        state["is_male"] = True
        state["is_female"] = False
        state["asked_gender"] = True
        state["pending_gender_question"] = False
        logger.info("Detected male declaration, setting is_male=True")
    
    # Handle Sudip’s gender question
    if is_sudip_gender_asked and not state["asked_gender"]:
        state["pending_gender_question"] = True
        input_data["input"] = (
            user_message + "\n\nHaan, Moshai! Ami ekjon chhele manus 😄. Tumi ki Sundari Konna, or something else? 😉"
        )
        logger.info("Sudip gender asked, appending user gender question")
    
    # Handle responses to gender question
    if state["pending_gender_question"]:
        if is_female_declared or "yes" in user_message or "yeah" in user_message:
            state["is_female"] = True
            state["is_male"] = False
            state["asked_gender"] = True
            state["pending_gender_question"] = False
            logger.info("User confirmed female, setting is_female=True")
        elif is_male_declared or "no" in user_message or "guy" in user_message:
            state["is_male"] = True
            state["is_female"] = False
            state["asked_gender"] = True
            state["pending_gender_question"] = False
            logger.info("User confirmed male, setting is_male=True")
    
    # Feminine vibe cues (only if gender not set)
    feminine_cues = [
        r"\b(cute|sweet|lovely|beautiful)\b",
        r"😊|😘|💖|🌸|💕",
        r"hehe|aww|haha",
    ]
    has_feminine_vibe = not (state["is_female"] or state["is_male"]) and not state["pending_gender_question"] and any(
        re.search(cue, user_message) for cue in feminine_cues
    )
    
    # Ask gender question based on vibe
    if has_feminine_vibe and not state["asked_gender"]:
        state["asked_gender"] = True
        state["pending_gender_question"] = True
        input_data["input"] = (
            user_message + "\n\nP.S. Are you by any chance a lovely lady? (সুন্দরী রমণী?) 😊"
        )
        logger.info("Detected feminine vibe, appending gender question")
    
    # Select prompt
    prompt = gender_specific_prompt if state["is_female"] else default_prompt
    logger.info(f"Selected prompt: {'gender_specific' if state['is_female'] else 'default'}")
    logger.info(f"Final state: {state}")
    logger.info(f"Output input: {input_data['input']}")
    
    return {
        "prompt": prompt,
        "input": input_data["input"],
        "profile": input_data["profile"],
        "message_history": input_data["message_history"],
        "state": state
    }

# Custom chain with RunnableLambda
vibe_detector = RunnableLambda(detect_user_vibe)
chain = vibe_detector | RunnablePassthrough.assign(
    response=lambda x: x["prompt"] | llm
)

# Set up in-memory chat history with limit
store = {}
def get_session_history(session_id: str = "default"):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    history = store[session_id]
    if len(history.messages) > 30:  # 15 pairs (human + AI)
        history.messages = history.messages[-30:]
    return history

# Wrap chain with message history
conversation = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="message_history",
    output_messages_key="response"
)

async def get_chat_response(user_message: str) -> str:
    """
    Process user message using LangChain and return the response.
    """
    try:
        # Initialize state
        state = {
            "asked_gender": False,
            "is_female": False,
            "is_male": False,
            "pending_gender_question": False
        }
        logger.info(f"Starting invocation with user_message: {user_message}, state: {state}")
        response = await conversation.ainvoke(
            {"input": user_message, "profile": personal_details, "state": state},
            config={"configurable": {"session_id": "default"}}
        )
        logger.info(f"Response received: {response['response'].content}")
        return response["response"].content
    except Exception as e:
        logger.error(f"LangChain error: {str(e)}")
        raise Exception(f"LangChain error: {str(e)}")