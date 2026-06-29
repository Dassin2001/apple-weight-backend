from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List
import os
import shutil
import random
from datetime import datetime
import logging
import zipfile
import cv2

from app.database import get_db, PictureProcessingHistory
from app.models import ProcessPicturesResponse, PicturesStatus, ProcessingStatus
from app.config import get_settings
from app.services import get_apple_detector
from app.utils.validators import validate_image_files


# Set up the router for the FastAPI app
router = APIRouter()
logger = logging.getLogger(__name__)

# Define directories for uploading and outputting pictures
PICTURE_UPLOAD_DIR = get_settings().PICTURE_UPLOAD_DIR
PICTURE_OUTPUT_DIR = get_settings().PICTURE_OUTPUT_DIR

# Ensure directories exist
os.makedirs(PICTURE_UPLOAD_DIR, exist_ok=True)
os.makedirs(PICTURE_OUTPUT_DIR, exist_ok=True)

# Generate a random task ID for each picture processing task
def generate_task_id():
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=10))

# Background task to process pictures
def process_pictures_task(file_paths: List[str], filenames: List[str], output_path: str, task_id: str, db: Session):
    # Get the directory where images are stored
    image_folder = os.path.dirname(file_paths[0])
    # Initialize the apple detector for image processing
    apple_detector = get_apple_detector("image")
    try:
        start_time = datetime.utcnow()  # Start time for calculating processing duration
        
        # If processing a single image
        if len(file_paths) == 1:
            image = cv2.imread(file_paths[0])  # Read image
            processed_image, unique_apple_count, total_weight = apple_detector._process_single_image(image)  # Process image
            os.makedirs(output_path, exist_ok=True)
            cv2.imwrite(os.path.join(output_path, f"processed_{filenames[0]}"), processed_image)  # Save processed image
        else:
            # If processing multiple images
            unique_apple_count, total_weight, _ = apple_detector.process_pictures(image_folder, output_folder=output_path)
        
        # Convert total weight from grams to kilograms
        total_weight = total_weight / 1000
        processing_time = (datetime.utcnow() - start_time).total_seconds()  # Calculate processing time

        # Create a zip file of the output images
        zip_file_path = f"{output_path}.zip"
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            for image_file in os.listdir(output_path):
                zipf.write(os.path.join(output_path, image_file), image_file)
        output_path = zip_file_path

        # Update the processing status in the database to 'COMPLETED'
        db.query(PictureProcessingHistory).filter(PictureProcessingHistory.task_id == task_id).update({
            "processing_time": processing_time,
            "unique_apple_count": unique_apple_count,
            "total_weight": total_weight,
            "output_path": output_path,
            "status": ProcessingStatus.COMPLETED
        })
        db.commit()  # Commit the changes to the database
    except Exception as e:
        # If an error occurs, log the error and update the status in the database to 'FAILED'
        logger.error(f"Error processing pictures {','.join(filenames)}: {str(e)}")
        db.query(PictureProcessingHistory).filter(PictureProcessingHistory.task_id == task_id).update({
            "status": ProcessingStatus.FAILED,
            "error_message": str(e)
        })
        db.commit()
    finally:
        # Clean up and remove the original uploaded files
        for file_path in file_paths:
            os.remove(file_path)

# Endpoint to handle picture upload and start background task for processing
@router.post("/process/", response_model=ProcessPicturesResponse)
async def process_pictures(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    try:
        # Validate that uploaded files are valid image types
        validate_image_files(files)
        file_paths = []
        filenames = []
        
        # Save each uploaded file to the upload directory
        for file in files:
            file_path = os.path.join(PICTURE_UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(file_path)
            filenames.append(file.filename)
        
        # Generate an output path and a unique task ID for this processing job
        output_path = os.path.join(PICTURE_OUTPUT_DIR, f"output_pictures_{random.randint(1000, 9999)}")
        task_id = generate_task_id()
        
        # Add a new entry to the database to track the picture processing task
        db_record = PictureProcessingHistory(
            filenames=",".join(filenames),
            status=ProcessingStatus.PROCESSING,
            task_id=task_id
        )
        db.add(db_record)
        db.commit()

        # Add the processing task to background tasks
        background_tasks.add_task(process_pictures_task, file_paths, filenames, output_path, task_id, db)
        logger.info(f"Picture processing started for {','.join(filenames)}")
        
        # Return a response indicating the task has been started
        return ProcessPicturesResponse(message=f"{len(files)} pictures are being processed with task id {task_id}")
    except ValueError as ve:
        # If a validation error occurs, raise an HTTP exception with a 400 status
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # If an unexpected error occurs, log it and raise an HTTP exception with a 500 status
        logger.error(f"Error in process_pictures: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

# Endpoint to check the status of a picture processing task
@router.get("/status/{task_id}", response_model=PicturesStatus)
async def get_pictures_status(task_id: str, db: Session = Depends(get_db)):
    # Query the database for the status of the task with the provided task ID
    pictures = db.query(PictureProcessingHistory).filter(PictureProcessingHistory.task_id == task_id).first()
    if not pictures:
        raise HTTPException(status_code=404, detail="Task not found")  # Raise error if task is not found
    return PicturesStatus(
        filenames=pictures.filenames.split(","),
        status=pictures.status,
        task_id=pictures.task_id,
        processing_time=pictures.processing_time,
        unique_apple_count=pictures.unique_apple_count,
        total_weight=pictures.total_weight,
        error_message=pictures.error_message
    )

# Endpoint to download the result of a processed picture task
@router.get("/download/{task_id}")
async def download_pictures_result(task_id: str, db: Session = Depends(get_db)):
    # Query the database for the completed task with the provided task ID
    pictures = db.query(PictureProcessingHistory).filter(PictureProcessingHistory.task_id == task_id).first()
    if not pictures:
        raise HTTPException(status_code=404, detail="Task not found")  # Raise error if task is not found
    if pictures.status != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Pictures processing not completed")  # Task must be completed
    
    file_path = pictures.output_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Processed pictures file not found")  # Error if output file doesn't exist
    
    # Return the processed pictures as a zip file if the output is a zip
    if file_path.endswith(".zip"):
        return FileResponse(file_path, media_type="application/zip", filename=f"processed_pictures_{task_id}.zip")
    else:
        # Return the result as a single image if it's not a zip file
        return FileResponse(file_path, media_type="image/jpeg", filename=f"processed_pictures_{task_id}.jpg")
