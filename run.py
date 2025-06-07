import uvicorn
import os

if __name__ == "__main__":
    
    environment = os.getenv("ENVIRONMENT", "production")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=(environment == "development")
    )
