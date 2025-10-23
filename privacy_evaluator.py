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
           
    def calculate_ssim_simple(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Simple structural similarity calculation"""
        try:
            # Convert to grayscale if needed
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            if len(img2.shape) == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Resize to same size if different
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Simple correlation coefficient as SSIM approximation
            img1_flat = img1.flatten().astype(np.float64)
            img2_flat = img2.flatten().astype(np.float64)
            
            correlation = np.corrcoef(img1_flat, img2_flat)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            return 0.0
    
    def print_evaluation_report(self, evaluation: Dict[str, any]):
        """Print a formatted evaluation report"""
        print("\n" + "="*50)
        print("PRIVACY PROTECTION EVALUATION REPORT")
        print("="*50)
        print(f"Original faces recognized: {evaluation['original_faces_recognized']}")
        print(f"Modified faces recognized: {evaluation['modified_faces_recognized']}")
        print(f"Privacy protection rate: {evaluation['privacy_protection_rate']:.2%}")
        print(f"Privacy protection successful: {evaluation['privacy_successful']}")
        print(f"Image quality (PSNR): {evaluation['image_quality_psnr']:.2f} dB")
        print(f"Structural similarity: {evaluation['structural_similarity']:.3f}")
        
        if evaluation['original_results']:
            print("\nOriginal image recognition results:")
            for name, confidence, location in evaluation['original_results']:
                print(f"  - {name} (confidence: {confidence:.3f})")
        
        if evaluation['modified_results']:
            print("\nModified image recognition results:")
            for name, confidence, location in evaluation['modified_results']:
                print(f"  - {name} (confidence: {confidence:.3f})")
        
        print("="*50)

def test_privacy_evaluation():
    """Test function for privacy evaluation"""
    evaluator = PrivacyEvaluator()
    
    # Try to access webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return
    
    print("Privacy evaluation test. Press:")
    print("'a' - Add current face as known person")
    print("'t' - Test recognition on current frame")
    print("'q' - Quit")
    
    person_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Privacy Evaluation Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            person_count += 1
            person_name = f"Person_{person_count}"
            success = evaluator.add_known_face(frame, person_name)
            if success:
                print(f"Added {person_name} to database")
        elif key == ord('t'):
            print("Testing recognition...")
            results = evaluator.recognize_face(frame)
            if results:
                for name, confidence, location in results:
                    print(f"Recognized: {name} (confidence: {confidence:.3f})")
            else:
                print("No faces recognized")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_privacy_evaluation()    