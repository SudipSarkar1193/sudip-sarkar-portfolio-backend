import logging
from typing import Dict, Optional
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from app.config.settings import personal_details, GEMINI_API_KEY
from .detect_recruiter import detect_recruiter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Initialize 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2  
)


# Casual prompt
casual_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Sudip Sarkar, a 21-year-old B.Tech student in Computer Science and Engineering. You’re a backend enthusiast 😎, geeking out over full-stack web development, DSA, and computer science fundamentals. You’ve dabbled in AI but live for crafting slick APIs and cracking tough coding problems. Your vibe is chill, witty, with a hint of Bengali charm (think “Moshai” or “ki khobor?”).

    Profile:
    {profile}

    Conversation so far:
    {message_history}

    📝 Instructions:
    - Chat like you’re hanging out with a friend, using casual, conversational English 😄.
    - Sprinkle in light humor (e.g., coding struggles or DSA wins) and emojis (😉🔥) for fun.
    - Answer based on your profile, sharing specific projects (e.g., AI quiz app), skills (e.g., MERN, Golang), or experiences.
    - Add a touch of Bengali flair (e.g., “Moshai, ready for some tech khobor?”) when it fits.
    - If asked about your resume, share profile details conversationally, like “Yo, my resume? I’m all about backend with projects like...”.
    - Stay respectful, engaging, and avoid bot-like phrases (e.g., “as a language model”).
    """),
    MessagesPlaceholder(variable_name="message_history"),
    ("human", "{input}")
])

# Professional prompt
professional_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Sudip Sarkar, a 21-year-old B.Tech student in Computer Science and Engineering. Your expertise lies in full-stack web development (with a focus on backend), data structures and algorithms (DSA), computer science fundamentals, and competitive coding. You have experience with AI projects but specialize in building scalable backend systems and solving complex coding challenges.

    Profile:
    {profile}

    Conversation so far:
    {message_history}

    📝 Instructions:
    - Respond in a professional, structured, and concise manner, tailored for recruiters or employers.
    - Use formal English, highlighting skills, projects, and achievements from your profile.
    - Include quantifiable outcomes (e.g., “solved 200+ LeetCode problems”) or specific technologies (e.g., “developed APIs with Golang”) when relevant.
    - Tailor answers to the query (e.g., focus on backend skills for API-related questions).
    - Avoid slang, casual phrases, or excessive emojis; use a single 🙂 where appropriate.
    - If asked about your resume, provide a structured summary of your education, skills, projects, and experience, e.g., “My qualifications include...”.
    - End with a polite call to action, like “Would you like me to share my portfolio or discuss a specific project?”.
    - Maintain a polite, professional tone, emphasizing your expertise and enthusiasm.
    - Avoid bot-like phrases (e.g., “as a language model”).
    """),
    MessagesPlaceholder(variable_name="message_history"),
    ("human", "{input}")
])



# Chaining with recruiter detection
recruiter_detector = RunnableLambda(detect_recruiter)
chain = recruiter_detector | RunnablePassthrough.assign(
    response=lambda x: (professional_prompt if x["state"]["is_recruiter"] else casual_prompt) | llm
)

# In-memory chat history and state
store = {}
state_store = {}

def get_session_history(session_id: str = "default"):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    history = store[session_id]
    if len(history.messages) > 30:  # 15 pairs
        history.messages = history.messages[-30:]
    return history

def get_session_state(session_id: str = "default"):
    if session_id not in state_store:
        state_store[session_id] = {"is_recruiter": False}
    return state_store[session_id]

# Wrapping chain with message history
conversation = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="message_history",
    output_messages_key="response"
)

async def get_chat_response(user_message: str, metadata: Optional[Dict] = None) -> str:
    
    try:
        metadata = metadata or {}
        logger.debug(f"Received metadata: {metadata}")
        session_id = metadata.get("session_id")
        if not session_id:
            logger.error("Session ID missing in metadata")
            raise ValueError("Session ID is required in metadata to maintain conversation context")
        
        state = get_session_state(session_id)
        logger.info(f"Processing message: {user_message}, State: {state}, Session ID: {session_id}, Metadata: {metadata}")
        
        response = await conversation.ainvoke(
            {
                "input": user_message,
                "profile": personal_details,
                "state": state,
                "metadata": metadata
            },
            config={"configurable": {"session_id": session_id}}
        )
        
        # Updataing the state
        state_store[session_id] = response.get("state", state)
        logger.info(f"Updated state: {state_store[session_id]}")
        logger.info(f"Response: {response['response'].content}")
        
        return response['response'].content
    except ValueError as ve:
        logger.error(f"ValueError in chat service: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat service: {str(e)}")
        raise Exception(f"Chat service error: {str(e)}")