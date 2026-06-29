from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from typing import List
import os
import shutil
import random
import logging
from datetime import datetime

from app.database import get_db, VideoProcessingHistory
from app.models import ProcessVideoResponse, VideoStatus, ProcessingStatus
from app.config import get_settings
from app.services import get_apple_detector
from app.utils.validators import validate_video_file

# Set up the router for the FastAPI app
router = APIRouter()
logger = logging.getLogger(__name__)

# Define directories for uploading and outputting videos
VIDEO_UPLOAD_DIR = get_settings().VIDEO_UPLOAD_DIR
VIDEO_OUTPUT_DIR = get_settings().VIDEO_OUTPUT_DIR
os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# Background task to process a video
def process_video_task(file_path: str, filename: str, output_path: str, db: Session):
    apple_detector = get_apple_detector('video')  # Initialize the apple detector for video processing
    try:
        start_time = datetime.utcnow()  # Record the start time to calculate processing duration
        
        # Process the video using the apple detector
        unique_apple_count, total_weight = apple_detector.process_video(file_path, output_path)
        processing_time = (datetime.utcnow() - start_time).total_seconds()  # Calculate processing time

        # Subquery to find the latest record for the given video file
        subquery = db.query(func.max(VideoProcessingHistory.upload_time)).filter(VideoProcessingHistory.filename == filename)
        
        # Update the latest record in the database to mark the task as completed
        db.query(VideoProcessingHistory).filter(
            VideoProcessingHistory.filename == filename,
            VideoProcessingHistory.upload_time == subquery
        ).update({
            "processing_time": processing_time,
            "unique_apple_count": unique_apple_count,
            "total_weight": total_weight,
            "output_path": output_path,
            "status": ProcessingStatus.COMPLETED
        })
        db.commit()  # Commit the changes to the database
        logger.info(f"Video processing completed for {filename}")
    except Exception as e:
        # If an error occurs during processing, log the error and mark the task as failed
        logger.error(f"Error processing video {filename}: {str(e)}")

        # Update the latest record in the database to mark the task as failed
        subquery = db.query(func.max(VideoProcessingHistory.upload_time)).filter(VideoProcessingHistory.filename == filename)
        
        db.query(VideoProcessingHistory).filter(
            VideoProcessingHistory.filename == filename,
            VideoProcessingHistory.upload_time == subquery
        ).update({
            "status": ProcessingStatus.FAILED,
            "error_message": str(e)
        })
        db.commit()
    finally:
        # Remove the uploaded video file after processing is complete
        os.remove(file_path)

# Endpoint to handle video upload and start background task for processing
@router.post("/process/", response_model=ProcessVideoResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        validate_video_file(file)  # Validate that the uploaded file is a valid video
        file_path = os.path.join(VIDEO_UPLOAD_DIR, file.filename)
        
        # Save the uploaded video to the upload directory
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Generate an output path for the processed video
        output_path = os.path.join(VIDEO_OUTPUT_DIR, f"output_{random.randint(1000, 9999)}_{file.filename}")
        
        # Add a new record to the database for tracking the video processing task
        db_record = VideoProcessingHistory(
            filename=file.filename,
            status=ProcessingStatus.PROCESSING
        )
        db.add(db_record)
        db.commit()

        # Add the video processing task to background tasks
        background_tasks.add_task(process_video_task, file_path, file.filename, output_path, db)
        logger.info(f"Video processing started for {file.filename}")
        
        # Return a response indicating the task has been started
        return ProcessVideoResponse(message=f"The video {file.filename} is being processed")
    except ValueError as ve:
        # If a validation error occurs, raise an HTTP exception with a 400 status
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # If an unexpected error occurs, log it and raise an HTTP exception with a 500 status
        logger.error(f"Error in process_video: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

# Endpoint to check the status of a video processing task
@router.get("/status/{filename}", response_model=VideoStatus)
async def get_video_status(filename: str, db: Session = Depends(get_db)):
    # Query the database for the status of the video processing task by filename
    video = db.query(VideoProcessingHistory).filter(VideoProcessingHistory.filename == filename).order_by(VideoProcessingHistory.upload_time.desc()).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")  # Raise error if task is not found
    return VideoStatus(
        filename=video.filename,
        status=video.status,
        processing_time=video.processing_time,
        unique_apple_count=video.unique_apple_count,
        total_weight=video.total_weight,
        error_message=video.error_message
    )

# Endpoint to download the processed video result
@router.get("/download/{filename}")
async def download_video(filename: str, db: Session = Depends(get_db)):
    # Query the database for the completed video processing task
    video = db.query(VideoProcessingHistory).filter(VideoProcessingHistory.filename == filename).order_by(VideoProcessingHistory.upload_time.desc()).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")  # Raise error if task is not found
    if video.status != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Video processing not completed")  # Task must be completed
    
    file_path = video.output_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Processed video file not found")  # Error if output file doesn't exist
    
    # Return the processed video as an MP4 file
    return FileResponse(file_path, media_type="video/mp4", filename=f"processed_{filename}")
