"""
Data Preprocessing Module for Object Detection
Handles CIFAR10 dataset loading, preprocessing, and augmentation
with security measures and input validation
"""

import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image
import logging
from typing import Tuple, Dict, Any
import hashlib
import magic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityValidator:
    """Security validation class for input data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_size = config['security']['max_input_size'] * 1024 * 1024  # Convert to bytes
        self.allowed_types = config['security']['allowed_file_types']
    
    def validate_file(self, file_path: str) -> bool:
        """Validate file security before processing"""
        try:
            # Check file size
            if os.path.getsize(file_path) > self.max_size:
                logger.warning(f"File {file_path} exceeds maximum size limit")
                return False
            
            # Check file type using magic numbers
            file_type = magic.from_file(file_path, mime=True)
            if not any(ext in file_type for ext in self.allowed_types):
                logger.warning(f"File {file_path} has unsupported type: {file_type}")
                return False
            
            # Calculate file hash for integrity
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                logger.info(f"File {file_path} hash: {file_hash[:8]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Security validation failed for {file_path}: {str(e)}")
            return False

class DataPreprocessor:
    """Data preprocessing class for CIFAR10 dataset"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize data preprocessor with configuration"""
        self.config = self._load_config(config_path)
        self.security_validator = SecurityValidator(self.config)
        self.input_shape = tuple(self.config['data']['input_shape'])
        self.num_classes = self.config['data']['num_classes']
        
        # CIFAR10 class names
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        logger.info("DataPreprocessor initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def load_cifar10_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                        Tuple[np.ndarray, np.ndarray], 
                                        Tuple[np.ndarray, np.ndarray]]:
        """
        Load CIFAR10 dataset with proper splitting
        
        Returns:
            Tuple of (train_data, validation_data, test_data)
        """
        logger.info("Loading CIFAR10 dataset...")
        
        # Load CIFAR10 data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # Convert labels to categorical
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
        
        # Split training data into train and validation
        split_idx = int(len(x_train) * self.config['data']['train_split'])
        x_val = x_train[split_idx:]
        y_val = y_train[split_idx:]
        x_train = x_train[:split_idx]
        y_train = y_train[:split_idx]
        
        logger.info(f"Dataset loaded - Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
    def preprocess_data(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data with normalization and augmentation
        
        Args:
            data: Tuple of (images, labels)
            
        Returns:
            Tuple of (preprocessed_images, labels)
        """
        images, labels = data
        
        # Normalize pixel values to [0, 1]
        images = images.astype('float32') / 255.0
        
        # Resize images to target input shape if needed
        if images.shape[1:3] != self.input_shape[:2]:
            resized_images = []
            for img in images:
                resized_img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
                resized_images.append(resized_img)
            images = np.array(resized_images)
        
        logger.info(f"Data preprocessed - Shape: {images.shape}")
        return images, labels
    
    def create_data_generators(self) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """
        Create data generators for training and validation with augmentation
        
        Returns:
            Tuple of (train_generator, val_generator)
        """
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=15,          # Random rotation up to 15 degrees
            width_shift_range=0.1,      # Random horizontal shift
            height_shift_range=0.1,     # Random vertical shift
            horizontal_flip=True,        # Random horizontal flip
            zoom_range=0.1,             # Random zoom
            shear_range=0.1,            # Random shear
            fill_mode='nearest',         # Fill strategy for transformed pixels
            rescale=1./255              # Normalize pixel values
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255              # Only normalization
        )
        
        logger.info("Data generators created successfully")
        return train_datagen, val_datagen
    
    def create_tf_datasets(self, train_data: Tuple[np.ndarray, np.ndarray],
                          val_data: Tuple[np.ndarray, np.ndarray],
                          test_data: Tuple[np.ndarray, np.ndarray]) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create TensorFlow datasets for efficient data loading
        
        Args:
            train_data: Training data tuple
            val_data: Validation data tuple
            test_data: Test data tuple
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        batch_size = self.config['data']['batch_size']
        buffer_size = self.config['data']['buffer_size']
        
        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
        
        # Configure datasets for performance
        train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        logger.info("TensorFlow datasets created successfully")
        return train_dataset, val_dataset, test_dataset
    
    def save_labels(self, output_path: str = "models/labels.txt"):
        """Save class labels to file for deployment"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for i, class_name in enumerate(self.class_names):
                f.write(f"{i}: {class_name}\n")
        
        logger.info(f"Labels saved to {output_path}")
    
    def get_class_names(self) -> list:
        """Get list of class names"""
        return self.class_names
    
    def validate_input_image(self, image_path: str) -> bool:
        """Validate input image for security and format"""
        return self.security_validator.validate_file(image_path)

def main():
    """Main function to test data preprocessing"""
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Load data
        train_data, val_data, test_data = preprocessor.load_cifar10_data()
        
        # Preprocess data
        train_data = preprocessor.preprocess_data(train_data)
        val_data = preprocessor.preprocess_data(val_data)
        test_data = preprocessor.preprocess_data(test_data)
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = preprocessor.create_tf_datasets(
            train_data, val_data, test_data
        )
        
        # Save labels
        preprocessor.save_labels()
        
        logger.info("Data preprocessing completed successfully!")
        
        return train_dataset, val_dataset, test_dataset, preprocessor.get_class_names()
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 