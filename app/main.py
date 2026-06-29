from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import video, image
from app.utils.error_handlers import add_error_handlers
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

# Include routers
app.include_router(video.router, prefix="/video", tags=["video"])
app.include_router(image.router, prefix="/image", tags=["image"])

# Add error handlers
add_error_handlers(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True)