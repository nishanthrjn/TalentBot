from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from engine import ask_talent_bot

app = FastAPI()

# 🛡️ CORS Middleware: This allows your Frontend (website) 
# to talk to your Backend (API) safely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your actual website URL
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "TalentBot is online!"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = ask_talent_bot(request.message)
        return {"reply": response}
    except Exception as e:
        return {"reply": f"Sorry, I encountered an error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)