import cv2
from .base_processing import BaseAppleDetector

# Class for detecting apples in video frames, inheriting from BaseAppleDetector
class VideoAppleDetector(BaseAppleDetector):
    def __init__(self, yolo_model_path, knn_model_path):
        # Initialize the YOLO and KNN models using the parent class
        super().__init__(yolo_model_path, knn_model_path)

    # Method to process a video for apple detection
    def process_video(self, video_path, output_path=None, skip_frames=1, frame_resize_factor=0.5):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Get frames per second and frame size of the video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * frame_resize_factor)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * frame_resize_factor)

        # If an output path is provided, initialize a video writer for saving processed frames
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # Create video writer object

        unique_apples = set()  # Set to track unique apples based on track IDs
        total_weight = 0.0  # Initialize total weight of detected apples
        frame_count = 0  # Track the number of processed frames

        # Process each frame of the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop if no frame is returned (end of video)

            frame_count += 1
            # Skip frames if specified (process every nth frame)
            if frame_count % skip_frames != 0:
                continue

            # Resize the frame to reduce computational load
            frame_resized = cv2.resize(frame, (width, height))
            
            # Run YOLO model to detect apples in the frame
            results = self.model(frame_resized)
            # Process the detections and get bounding boxes and estimated weights
            detections, weights = self._process_detections_video(results, conf_threshold=0.6)
            
            # Update tracking information for detected apples using DeepSort
            tracked_objects = self.tracker.update_tracks(detections, frame=frame_resized)
            
            # Update weights and track unique apples
            for obj, weight in zip(tracked_objects, weights):
                if obj.is_confirmed() and obj.time_since_update == 0:
                    # Convert weight to kilograms and update tracker
                    weight = weight / 1000
                    self.weight_tracker.update(obj.track_id, weight)
                    
                    # If it's a new apple, add it to the set and update total weight
                    if obj.track_id not in unique_apples:
                        unique_apples.add(obj.track_id)
                        total_weight += weight

            # Draw bounding boxes and tracking results on the frame
            frame_resized = self._draw_results(frame_resized, tracked_objects, unique_apples, total_weight)

            # Write the processed frame to the output video if output path is provided
            if output_path:
                out.write(frame_resized)

        # Release video capture and writer resources
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()  # Close all OpenCV windows

        # Return the number of unique apples and total weight detected in the video
        return len(unique_apples), total_weight
