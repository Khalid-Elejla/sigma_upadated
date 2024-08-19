from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List
from .app.run_rag import main #process_and_query_documents
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
orig_prefix = "http://localhost:3000"  # Frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[orig_prefix],  # Or use "*" to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for interactions
interactions = []

# Define the Interaction model
class Interaction(BaseModel):
    query: str
    response: str
    timestamp: datetime

@app.post("/api/query/")
async def submit_query(query: str):
    # response = process_and_query_documents(data_path="data", query_string=query)
    response = main(data_path="app/data", query_string=query)

    interaction = Interaction(query=query, response=response, timestamp=datetime.now())
    interactions.append(interaction)
    return {"response": response}

@app.get("/api/interactions/", response_model=List[Interaction])
async def get_interactions():
    return interactions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
