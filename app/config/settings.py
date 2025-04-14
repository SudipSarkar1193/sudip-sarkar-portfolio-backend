import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Load personal details from sudip.json
json_path = os.path.join(os.path.dirname(__file__), "../../sudip.json")
try:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"sudip.json not found at {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            raise ValueError("sudip.json is empty")
        personal_details = json.dumps(json.loads(content), indent=2)
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in sudip.json: {str(e)}")
except Exception as e:
    raise ValueError(f"Failed to load sudip.json: {str(e)}")