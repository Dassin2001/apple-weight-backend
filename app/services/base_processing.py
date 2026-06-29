import random
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import joblib
from app.config import get_settings 
import math

# Class to track the weight of apples using their tracking IDs
class WeightTracker:
    def __init__(self):
        # Dictionary to store weights with their corresponding track ID
        self.weight_map = {}
    
    # Update the weight of a specific tracked apple
    def update(self, track_id, weight):
        self.weight_map[track_id] = weight
    
    # Retrieve the weight of a specific tracked apple
    def get(self, track_id):
        return self.weight_map.get(track_id, 0)

# Main class for apple detection and weight estimation
class BaseAppleDetector:
    def __init__(self, yolo_model_path, knn_model_path):
        # Load YOLO model for object detection
        self.model = YOLO(yolo_model_path)
        # Initialize DeepSort for tracking objects (apples) across frames
        self.tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
        # Load KNN model for weight estimation
        self.weight_estimation_model = joblib.load(knn_model_path)
        # Initialize weight tracker
        self.weight_tracker = WeightTracker()

    # Method to normalize the features before sending to the KNN model
    def normalize_features(self, volume, surface_area):
        # Define mean and standard deviation for normalization (based on dataset characteristics)
        mean_volume, std_volume = 500, 100
        mean_surface_area, std_surface_area = 300, 50
        # Normalize the volume and surface area
        normalized_volume = (volume - mean_volume) / std_volume
        normalized_surface_area = (surface_area - mean_surface_area) / std_surface_area
        return normalized_volume, normalized_surface_area

    # Process detections in video frames to extract bounding boxes, class, and confidence
    def _process_detections_video(self, results, conf_threshold=0.7):
        detections = []  # Store bounding box coordinates and classes
        weights = []  # Store weights of detected apples
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Extract confidence and class of the detection
                conf = box.conf.cpu().numpy()[0]
                cls = box.cls.cpu().numpy()[0]

                # Check if the detected object is an apple and confidence is high enough
                if self.model.names[int(cls)] in ['apple', 'orange', 'apple_on_floor'] and conf >= conf_threshold:
                    # Calculate width and height of the bounding box
                    w, h = x2 - x1, y2 - y1
                    # Estimate volume and surface area of the apple using its bounding box
                    radius = (w + h) / 4.0
                    volume = (4 / 3) * math.pi * (radius ** 3)
                    surface_area = 4 * math.pi * (radius ** 2)

                    # Normalize features before sending them to the KNN model
                    norm_volume, norm_surface_area = self.normalize_features(volume, surface_area)
                    feature_vector = np.array([norm_volume, norm_surface_area]).reshape(1, -1)

                    # Predict the weight using the KNN model
                    weight = self.weight_estimation_model.predict(feature_vector)[0]
                    # Adjust weight based on the confidence score
                    adjusted_weight = weight * conf

                    # Apply heuristic adjustment based on the aspect ratio of the bounding box
                    aspect_ratio = w / h
                    if aspect_ratio > 1.5 or aspect_ratio < 0.75:
                        adjusted_weight *= 0.9  # Penalize for unusual aspect ratios
                    # Append the detection and weight
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls)))
                    weights.append(adjusted_weight)
                    
        return detections, weights

    # Process detections (applies to static images)
    def _process_detections(self, results, conf_threshold=0.7):
        detections = []  # Store bounding boxes and weights
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()[0]
                cls = box.cls.cpu().numpy()[0]

                # Filter detections to apples or objects of interest
                if self.model.names[int(cls)] in ['apple', 'orange', 'apple_on_floor'] and conf >= conf_threshold:
                    w, h = x2 - x1, y2 - y1
                    # Estimate volume and surface area
                    radius = (w + h) / 4.0
                    volume = (4 / 3) * math.pi * (radius ** 3)
                    surface_area = 4 * math.pi * (radius ** 2)

                    # Normalize features before KNN weight prediction
                    norm_volume, norm_surface_area = self.normalize_features(volume, surface_area)
                    feature_vector = np.array([norm_volume, norm_surface_area]).reshape(1, -1)

                    # Predict and adjust weight
                    weight = self.weight_estimation_model.predict(feature_vector)[0]
                    adjusted_weight = weight * conf

                    # Adjust weight based on aspect ratio heuristics
                    aspect_ratio = w / h
                    if aspect_ratio > 1.5 or aspect_ratio < 0.75:
                        adjusted_weight *= 0.9

                    # Append bounding box and weight information
                    detections.append(([x1, y1, w, h], conf, adjusted_weight))
        return detections

    # Draw the tracking and weight results on the frame
    def _draw_results(self, frame, tracked_objects, unique_apples, total_weight):
        for obj in tracked_objects:
            # Only process confirmed objects that are updated in the current frame
            if obj.is_confirmed() and obj.time_since_update == 0:
                bbox = obj.to_tlbr()  # Bounding box for the object
                track_id = obj.track_id  # Unique tracking ID
                weight = self.weight_tracker.get(track_id)  # Get weight from tracker
                
                # Draw bounding box and tracking information on the frame
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f'Weight: {weight:.2f} kg', (int(bbox[0]), int(bbox[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the number of unique apples and total weight on the frame
        height, width = frame.shape[:2]
        apple_count_text = f'Unique Apples: {len(unique_apples)}'
        total_weight_text = f'Total Weight: {total_weight:.2f} kg'
        cv2.putText(frame, apple_count_text, (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, total_weight_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return frame  # Return the modified frame

# Factory function to create an instance of the apple detector with appropriate model paths
def get_apple_detector():
    return BaseAppleDetector(
        yolo_model_path=get_settings().YOLO_MODEL_PATH,
        knn_model_path=get_settings().KNN_MODEL_PATH
    )
