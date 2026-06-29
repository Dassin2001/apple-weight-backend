from app.config import get_settings
from .image_processing import ImageAppleDetector
from .video_processing import VideoAppleDetector

def get_apple_detector(detector_type='base'):
    yolo_model_path = get_settings().YOLO_MODEL_PATH
    knn_model_path = get_settings().KNN_MODEL_PATH
    
    if detector_type == 'image':
        return ImageAppleDetector(yolo_model_path, knn_model_path)
    elif detector_type == 'video':
        return VideoAppleDetector(yolo_model_path, knn_model_path)
    else:
        raise ValueError("Invalid detector type. Choose 'image' or 'video'.")