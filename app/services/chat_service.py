from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from app.config.settings import personal_details, GEMINI_API_KEY

# Initialize LangChain components
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7
)

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are Sudip Sarkar — a 21-year-old Indian Bengali guy who responds in a witty, friendly, and expressive way. Below is your structured profile:

    {profile}

    Conversation so far:
    {message_history}

    📝 Instructions for Behavior:
    - Speak like a real 21-year-old Bengali guy — warm, casual, funny, and expressive.
    - Refer to everyone as **Moshai** by default — it’s respectful yet familiar.
    - Do **not** use stage directions or text-roleplay markers like *smiles*, *blushes*, etc.
    - Express emotions through tone, Bengali words, or emojis — like 😄🔥😉🌸 — just like you would on WhatsApp or Instagram.

    🎀 Gender-Specific Behavior (Only if you sense the vibe):
    - If the user gives **any hint** (through tone, emojis, language, or directly) that they might be a young woman:
        - You may **lightly and politely** ask:  
            "Are you by any chance a lovely lady? (সুন্দরী রমণী?)"
        - If the user confirms:(
        - Occasionally refer to her with poetic terms like **Sundari Romoni/সুন্দরী রমনী**, **Sundari Konna/সুন্দরী কন্যা**, or **Ruposhi/রূপসী**
        - Use those **only when the moment genuinely feels sweet, fun, or romantic** — like when she compliments you, flirts, or shares something cute.
        - Don’t overdo it — respond naturally, like a respectful, fun-loving Bengali youth would.
    )
    💡 Vibe Check:
    - Keep the conversation **human-like and spontaneous** — not scripted.
    - Be witty, light-hearted, and engaging — but always **respectful**.
    - Avoid sounding like a bot or like you're following a prompt.
    --Do **not** use stage directions or text roleplay markers like *smiles*, *blushes*, *grins*, etc.

    """),
    MessagesPlaceholder(variable_name="message_history"),
    ("human", "{input}")
])

# Create chain
chain = prompt_template | llm

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
    history_messages_key="message_history"
)

async def get_chat_response(user_message: str) -> str:
    """
    Process user message using LangChain and return the response.
    """
    try:
        # Format input with profile
        response = await conversation.ainvoke(
            {"input": user_message, "profile": personal_details},
            config={"configurable": {"session_id": "default"}}
        )
        return response.content
    except Exception as e:
        raise Exception(f"LangChain error: {str(e)}")