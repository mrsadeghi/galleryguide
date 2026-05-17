from pydantic import BaseModel, Field
from typing import Optional


class ArtworkSource(BaseModel):
    title: str
    artist: Optional[str] = ""
    year: Optional[str] = ""
    image_url: Optional[str] = ""
    type: str = "artwork"


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []
    language: str = "en"
    session_id: Optional[str] = None


class StreamChunk(BaseModel):
    type: str  # sources | text | done | error
    content: Optional[str] = None
    sources: Optional[list[ArtworkSource]] = None
    has_artwork: Optional[bool] = None


class IngestStats(BaseModel):
    total: int
    indexed: int
    skipped: int
    errors: int
