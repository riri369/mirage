import cv2
import numpy as np
import tensorflow as tf
from typing import List, Optional, Tuple
import os

class AdversarialFaceModifier:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize adversarial face modifier
        
        Args:
            model_path: Path to pre-trained GAN model, uses default if None
        """
        self.model = None
        self.input_size = (128, 128)  # Standard face input size
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No pre-trained model provided. Will use simple perturbation method.")
    
    def load_model(self, model_path: str):
        """Load a pre-trained GAN model for face modification"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for model input
        
        Args:
            face_image: Input face image
            
        Returns:
            Preprocessed face image ready for model
        """
        # Resize to model input size
        resized = cv2.resize(face_image, self.input_size)
        
        # Normalize to [-1, 1] range (common for GANs)
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def postprocess_face(self, model_output: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess model output back to image format
        
        Args:
            model_output: Model output tensor
            target_size: Target output size (width, height)
            
        Returns:
            Processed face image
        """
        # Remove batch dimension
        output = model_output[0]
        
        # Denormalize from [-1, 1] to [0, 255]
        denormalized = ((output + 1.0) * 127.5).astype(np.uint8)
        
        # Resize to target size
        resized = cv2.resize(denormalized, target_size)
        
        return resized
    
    def apply_simple_perturbation(self, face_image: np.ndarray, 
                                 noise_factor: float = 0.1) -> np.ndarray:
        """
        Apply simple adversarial perturbation without GAN model
        
        Args:
            face_image: Input face image
            noise_factor: Amount of noise to add (0.0 to 1.0)
            
        Returns:
            Modified face image with perturbation
        """
        # Generate random noise
        noise = np.random.normal(0, noise_factor * 255, face_image.shape)
        
        # Add noise to image
        perturbed = face_image.astype(np.float32) + noise
        
        # Clip to valid pixel range
        perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
        
        return perturbed
    
    def apply_gaussian_blur(self, face_image: np.ndarray, 
                           kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian blur for privacy protection
        
        Args:
            face_image: Input face image
            kernel_size: Size of Gaussian kernel
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Blurred face image
        """
        return cv2.GaussianBlur(face_image, (kernel_size, kernel_size), sigma)
    
    def apply_pixelation(self, face_image: np.ndarray, pixel_size: int = 10) -> np.ndarray:
        """
        Apply pixelation effect for privacy
        
        Args:
            face_image: Input face image
            pixel_size: Size of each pixel block
            
        Returns:
            Pixelated face image
        """
        h, w = face_image.shape[:2]
        
        # Resize down
        small = cv2.resize(face_image, (w // pixel_size, h // pixel_size), 
                          interpolation=cv2.INTER_LINEAR)
        
        # Resize back up with nearest neighbor for blocky effect
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return pixelated
    
    def modify_face(self, face_image: np.ndarray, method: str = "perturbation", 
                   **kwargs) -> np.ndarray:
        """
        Main function to modify face for privacy protection
        
        Args:
            face_image: Input face image
            method: Modification method ("gan", "perturbation", "blur", "pixelate")
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Modified face image
        """
        if method == "gan" and self.model is not None:
            # Use GAN model for modification
            preprocessed = self.preprocess_face(face_image)
            modified = self.model.predict(preprocessed, verbose=0)
            return self.postprocess_face(modified, 
                                       (face_image.shape[1], face_image.shape[0]))
        
        elif method == "perturbation":
            noise_factor = kwargs.get("noise_factor", 0.1)
            return self.apply_simple_perturbation(face_image, noise_factor)
        
        elif method == "blur":
            kernel_size = kwargs.get("kernel_size", 5)
            sigma = kwargs.get("sigma", 1.0)
            return self.apply_gaussian_blur(face_image, kernel_size, sigma)
        
        elif method == "pixelate":
            pixel_size = kwargs.get("pixel_size", 10)
            return self.apply_pixelation(face_image, pixel_size)
        
        else:
            print(f"Unknown method: {method}. Using perturbation as fallback.")
            return self.apply_simple_perturbation(face_image)

def test_adversarial_modification():
    """Test function for adversarial face modification"""
    modifier = AdversarialFaceModifier()
    
    # Try to access webcam for testing
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return
    
    print("Adversarial modification test started. Press keys:")
    print("'1' - Perturbation, '2' - Blur, '3' - Pixelate, 'q' - Quit")
    
    current_method = "perturbation"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # For demo, apply modification to entire frame
        if current_method == "perturbation":
            modified = modifier.modify_face(frame, "perturbation", noise_factor=0.05)
        elif current_method == "blur":
            modified = modifier.modify_face(frame, "blur", kernel_size=7, sigma=2.0)
        elif current_method == "pixelate":
            modified = modifier.modify_face(frame, "pixelate", pixel_size=8)
        else:
            modified = frame
        
        # Display original and modified side by side
        combined = np.hstack([frame, modified])
        cv2.imshow('Original (left) vs Modified (right)', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_method = "perturbation"
            print("Switched to perturbation method")
        elif key == ord('2'):
            current_method = "blur"
            print("Switched to blur method")
        elif key == ord('3'):
            current_method = "pixelate"
            print("Switched to pixelate method")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_adversarial_modification()
    
    