from pydantic import BaseModel
from datetime import datetime

class Interaction(BaseModel):
    query: str
    response: str
    timestamp: datetime
