from pydantic import BaseModel
from typing import Optional, Dict


class Message(BaseModel):
    user_message: str
    metadata: Optional[Dict] = None