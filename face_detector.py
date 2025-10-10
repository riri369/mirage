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