from fastapi import FastAPI , HTTPException
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
        response = await get_chat_response(message.user_message, message.metadata)
        return {"response": response}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}