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
    
    