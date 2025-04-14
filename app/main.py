from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models.message import Message
from app.services.chat_service import get_chat_response

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://sudip-sarkar.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(message: Message):
    try:
        response = await get_chat_response(message.user_message)
        return {"response": response}
    except Exception as e:
        return {"error": f"Error processing message: {str(e)}"}, 500

@app.get("/health")
async def health_check():
    return {"status": "healthy"}