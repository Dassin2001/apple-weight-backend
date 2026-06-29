from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import video, image
from app.utils.error_handlers import add_error_handlers
from pathlib import Path
from app.config import get_settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Apple Detection API",
    description="API for processing videos and images to detect and analyze apples",
    version="1.0.0",
)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def ensure_runtime_directories():
    settings = get_settings()

    directories = [
        settings.VIDEO_UPLOAD_DIR,
        settings.PICTURE_UPLOAD_DIR,
        settings.VIDEO_OUTPUT_DIR,
        settings.PICTURE_OUTPUT_DIR,
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


ensure_runtime_directories()

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Backend is running"
    }


# Include routers
app.include_router(video.router, prefix="/video", tags=["video"])
app.include_router(image.router, prefix="/image", tags=["image"])

# Add error handlers
add_error_handlers(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True)