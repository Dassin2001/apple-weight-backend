from pydantic import BaseModel
from typing import Optional
from enum import Enum
from typing import List

class ProcessingStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ProcessVideoResponse(BaseModel):
    message: str

class VideoStatus(BaseModel):
    filename: str
    status: ProcessingStatus
    processing_time: Optional[float] = None
    unique_apple_count: Optional[int] = None
    total_weight: Optional[float] = None
    error_message: Optional[str] = None


class ProcessPicturesResponse(BaseModel):
    message: str


class PicturesStatus(BaseModel):
    filenames: List[str]
    task_id: Optional[str]
    status: ProcessingStatus
    processing_time: Optional[float] = None
    unique_apple_count: Optional[int] = None
    total_weight: Optional[float] = None
    error_message: Optional[str] = None