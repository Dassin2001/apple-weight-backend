from fastapi import UploadFile
from typing import List

# Function to validate a single video file
def validate_video_file(file: UploadFile):
    # List of allowed video file extensions
    allowed_extensions = ['.mp4', '.avi', '.mov']
    # Extract the file extension from the uploaded file
    file_extension = file.filename.lower().split('.')[-1]
    
    # Check if the file extension is in the allowed list
    if f'.{file_extension}' not in allowed_extensions:
        # Raise an error if the file extension is not valid
        raise ValueError(f"Invalid file type. Allowed types are: {', '.join(allowed_extensions)}")

# Function to validate a list of image files
def validate_image_files(files: List[UploadFile]):
    # Ensure that at least one file is uploaded
    if len(files) < 1:
        raise ValueError("At least 1 picture is required")
    
    # List of allowed image file extensions
    allowed_extensions = ['.jpg', '.jpeg', '.png']
    
    # Iterate over each uploaded file to validate its extension
    for file in files:
        # Extract the file extension from each uploaded file
        file_extension = file.filename.lower().split('.')[-1]
        # Check if the file extension is in the allowed list
        if f'.{file_extension}' not in allowed_extensions:
            # Raise an error if the file extension is not valid
            raise ValueError(f"Invalid file type for {file.filename}. Allowed types are: {', '.join(allowed_extensions)}")
