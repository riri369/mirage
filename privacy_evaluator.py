import cv2
import numpy as np
import face_recognition
from typing import List, Tuple, Dict, Optional
import time

class PrivacyEvaluator:
    def __init__(self):
        """Initialize privacy evaluator"""
        self.known_encodings = []
        self.known_names = []
        self.recognition_threshold = 0.6  # Lower = more strict matching
    
    def add_known_face(self, image: np.ndarray, name: str) -> bool:
        """
        Add a known face to the database for recognition testing
        
        Args:
            image: Face image
            name: Person's name/identifier
            
        Returns:
            True if face was successfully encoded and added
        """
        try:
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            
            if len(face_locations) == 0:
                print(f"No face found in image for {name}")
                return False
            
            # Get face encoding (use first face if multiple found)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if len(face_encodings) > 0:
                self.known_encodings.append(face_encodings[0])
                self.known_names.append(name)
                print(f"Added {name} to known faces database")
                return True
            else:
                print(f"Could not encode face for {name}")
                return False
                
        except Exception as e:
            print(f"Error adding known face {name}: {e}")
            return False
    
    def recognize_face(self, image: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Attempt to recognize faces in an image
        
        Args:
            image: Input image to analyze
            
        Returns:
            List of tuples containing (name, confidence, face_location)
        """
        if len(self.known_encodings) == 0:
            return []
        
        try:
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            results = []
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_encodings, face_encoding, tolerance=self.recognition_threshold
                )
                face_distances = face_recognition.face_distance(
                    self.known_encodings, face_encoding
                )
                
                if True in matches:
                    # Find best match
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                        results.append((name, confidence, face_location))
                    else:
                        results.append(("Unknown", 0.0, face_location))
                else:
                    results.append(("Unknown", 0.0, face_location))
            
            return results
            
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return []
        
    def evaluate_privacy_protection(self, original_image: np.ndarray, 
                                  modified_image: np.ndarray) -> Dict[str, any]:
        """
        Evaluate how well the modification protects privacy
        
        Args:
            original_image: Original face image
            modified_image: Privacy-protected face image
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Recognize faces in both images
        original_results = self.recognize_face(original_image)
        modified_results = self.recognize_face(modified_image)
        
        # Calculate metrics
        original_recognized = len([r for r in original_results if r[0] != "Unknown"])
        modified_recognized = len([r for r in modified_results if r[0] != "Unknown"])
        
        privacy_protection_rate = 0.0
        if original_recognized > 0:
            privacy_protection_rate = (original_recognized - modified_recognized) / original_recognized
        
        # Calculate image quality metrics (simple PSNR)
        psnr = self.calculate_psnr(original_image, modified_image)
        
        # Calculate structural similarity
        ssim = self.calculate_ssim_simple(original_image, modified_image)
        
        evaluation = {
            "original_faces_recognized": original_recognized,
            "modified_faces_recognized": modified_recognized,
            "privacy_protection_rate": privacy_protection_rate,
            "privacy_successful": modified_recognized == 0,
            "image_quality_psnr": psnr,
            "structural_similarity": ssim,
            "original_results": original_results,
            "modified_results": modified_results
        }
        
        return evaluation
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio between two images"""
        try:
            mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
            if mse == 0:
                return float('inf')
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            return psnr
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
            return 0.0    