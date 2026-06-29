from collections import defaultdict
import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from .base_processing import BaseAppleDetector

# Class for detecting apples in images, inherits from BaseAppleDetector
class ImageAppleDetector(BaseAppleDetector):
    def __init__(self, yolo_model_path, knn_model_path):
        # Initialize the YOLO and KNN models using the parent class
        super().__init__(yolo_model_path, knn_model_path)

    # Process a single image to detect apples and estimate their weights
    def _process_single_image(self, image):
        # Run YOLO model to get detections
        results = self.model(image)
        # Process detections with a confidence threshold of 0.2
        detections = self._process_detections(results, conf_threshold=0.2)
        
        unique_apples = set()  # Set to track unique apples
        total_weight = 0.0  # Initialize total weight of apples
        
        # Draw bounding boxes and accumulate weights
        for box, conf, weight in detections:
            x1, y1, w, h = map(int, box)
            x2, y2 = x1 + w, y1 + h
            # Draw bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Track unique apples by their coordinates
            unique_apples.add((x1, y1, x2, y2))
            total_weight += weight  # Accumulate total weight

        # Add text overlays showing apple count and total weight
        cv2.putText(image, f"Unique Apples: {len(unique_apples)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Total Weight: {(total_weight/1000):.2f} kg", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return image, len(unique_apples), total_weight  # Return processed image, apple count, and total weight

    # Generate keypoints using feature detection methods like AKAZE, BRISK, SIFT, or ORB
    def _generate_keypoints(self, image, method='AKAZE', n_keypoints=1500):
        if method == 'AKAZE':
            akaze = cv2.AKAZE_create()
            keypoints, descriptors = akaze.detectAndCompute(image, None)
        elif method == 'BRISK':
            brisk = cv2.BRISK_create()
            keypoints, descriptors = brisk.detectAndCompute(image, None)
        elif method == 'SIFT':
            sift = cv2.SIFT_create(nfeatures=n_keypoints)
            keypoints, descriptors = sift.detectAndCompute(image, None)
        elif method == 'ORB':
            orb = cv2.ORB_create(nfeatures=n_keypoints)
            keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors  # Return detected keypoints and their descriptors

    # Match keypoints between two images using the BFMatcher algorithm
    def _match_keypoints(self, kp1, desc1, kp2, desc2, ratio_thresh=0.75):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(desc1, desc2, k=2)
        good_matches = []
        # Apply ratio test to filter out bad matches
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        return good_matches

    # Use RANSAC to refine the matched keypoints and find a homography matrix
    def _ransac_match(self, kp1, kp2, good_matches, min_matches=10):
        if len(good_matches) < min_matches:
            return None, []

        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography using RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return None, []

        # Filter good matches based on the RANSAC mask
        matchesMask = mask.ravel().tolist()
        return M, [match for match, keep in zip(good_matches, matchesMask) if keep]

    # Calculate the average point between two matched keypoints
    def _average_point(self, kp1, kp2, match):
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        return ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)

    # Cluster matched keypoints using DBSCAN
    def cluster_keypoints(self, matched_points_np, eps=50, min_samples=1):
        """Uses DBSCAN for clustering the matched keypoints."""
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(matched_points_np)
        return clustering.labels_  # Return cluster labels

    # Process a folder of images to detect apples and track them across images
    def process_pictures(self, image_folder, output_folder=None):
        # Get all image files from the specified folder
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        images = []
        all_bounding_boxes = []
        apple_centers = defaultdict(list)  # To track apple centers across images
        global_id = 0  # For tracking potentially unique apples across images
        unique_apples = set()
        total_weight = 0

        # Step 1: Process each image and extract apples
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            # Detect apples in the image
            results = self.model(image)
            apples = self._process_detections(results)

            images.append(image)
            all_bounding_boxes.append(apples)

            # Track apple centers
            for box, _, weight in apples:
                x1, y1, w, h = map(int, box)
                center = (x1 + w // 2, y1 + h // 2)
                apple_centers[image_file].append((center, global_id, weight))
                global_id += 1

        # Step 2: Assign IDs and draw bounding boxes
        for idx, (image, boxes) in enumerate(zip(images, all_bounding_boxes)):
            image_file = image_files[idx]
            for local_id, (box, conf, weight) in enumerate(boxes):
                x1, y1, w, h = map(int, box)
                center = (x1 + w // 2, y1 + h // 2)
                # Get global ID for the apple center
                global_id = next(gid for (c, gid, _) in apple_centers[image_file] if c == center)
                
                # Draw bounding boxes and apple ID
                color = self._get_color(local_id)
                cv2.rectangle(image, (x1, y1), (x1+w, y1+h), color, 2)
                cv2.putText(image, f"{local_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Only count the weight once per unique apple
                if global_id not in unique_apples:
                    unique_apples.add(global_id)
                    total_weight += weight

            # Add summary text to image
            cv2.putText(image, f"Apples in this image: {len(boxes)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Total unique apples so far: {len(unique_apples)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Total Weight: {total_weight/1000:.2f} kg", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save output images if output folder is specified
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            for i, img in enumerate(images):
                output_path = os.path.join(output_folder, f"visualized_image_{i}.jpg")
                cv2.imwrite(output_path, img)

        return len(unique_apples), total_weight, images  # Return number of unique apples, total weight, and processed images
    
    # Generate a color for drawing bounding boxes
    def _get_color(self, label):
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        return colors[label % len(colors)]  # Return a color based on the label
