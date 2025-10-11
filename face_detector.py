#this handles out basic face detection using opencv and Haar cascades

import cv2
import numpy as np
from typing import List, Tuple, Optional

class FaceDetector:
    def __init__(self, cascade_path: Optional[str] = None):

        """
        Initialize face detector with Haar Cascade classifier
        
        Args:
            cascade_path: Path to custom Haar cascade file, uses default if None
        """
        if cascade_path is None:
            # Use OpenCV's built-in face cascade
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        else:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
        if self.face_cascade.empty():
            raise ValueError("Error loading face cascade classifier")
    
    def detect_faces(self, image: np.ndarray, scale_factor: float = 1.1, 
                    min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)) -> List[Tuple[int, int, int, int]]:
        
        """
        Detect faces in an image
        
        Args:
            image: Input image as numpy array (BGR format)
            scale_factor: How much the image size is reduced at each scale
            min_neighbors: How many neighbors each face should retain
            min_size: Minimum possible face size (width, height)
            
        Returns:
            List of tuples containing (x, y, width, height) of detected faces
        """
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        return faces.tolist()
    
    def extract_face_regions(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                           padding: int = 10) -> List[np.ndarray]:
        """
        Extract face regions from image based on detection coordinates
        
        Args:
            image: Original image
            faces: List of face coordinates (x, y, width, height)
            padding: Additional pixels around face region
            
        Returns:
            List of extracted face images
        """
        face_regions = []
        
        for (x, y, w, h) in faces:
            # Add padding while ensuring we don't go outside image bounds
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w + padding)
            y_end = min(image.shape[0], y + h + padding)
            
            face_region = image[y_start:y_end, x_start:x_end]
            face_regions.append(face_region)
        
        return face_regions
    
    def draw_face_rectangles(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                           color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw rectangles around detected faces
        
        Args:
            image: Input image
            faces: List of face coordinates
            color: Rectangle color in BGR format
            thickness: Rectangle line thickness
            
        Returns:
            Image with face rectangles drawn
        """
        result_image = image.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
        
        return result_image

def test_face_detection():
    """Test function to verify face detection works"""
    # Initialize detector
    detector = FaceDetector()
    
    # Try to access webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return
    
    print("Face detection test started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from webcam")
            break
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Draw rectangles around faces
        result = detector.draw_face_rectangles(frame, faces)
        
        # Display result
        cv2.imshow('Face Detection Test', result)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_face_detection() 