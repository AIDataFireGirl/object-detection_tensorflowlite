"""
Model Quantization Module for TensorFlow Lite
Handles model quantization, mixed precision, and optimization
for Raspberry Pi deployment
"""

import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot
from tensorflow.lite.python.interpreter import Interpreter
import logging
from typing import Dict, Any, Tuple, List
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelQuantizer:
    """Model quantization class for TensorFlow Lite conversion"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize model quantizer with configuration"""
        self.config = self._load_config(config_path)
        self.quantization_config = self.config['quantization']
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def create_representative_dataset(self, dataset, num_samples: int = 100) -> List[np.ndarray]:
        """
        Create representative dataset for quantization
        
        Args:
            dataset: TensorFlow dataset
            num_samples: Number of samples to use for calibration
            
        Returns:
            List of representative samples
        """
        logger.info(f"Creating representative dataset with {num_samples} samples...")
        
        representative_samples = []
        sample_count = 0
        
        for batch in dataset:
            images, _ = batch
            for image in images:
                if sample_count >= num_samples:
                    break
                # Convert to float32 and ensure proper shape
                image = tf.cast(image, tf.float32)
                representative_samples.append(image.numpy())
                sample_count += 1
            if sample_count >= num_samples:
                break
        
        logger.info(f"Representative dataset created with {len(representative_samples)} samples")
        return representative_samples
    
    def quantize_model_dynamic_range(self, model_path: str, 
                                   representative_dataset: List[np.ndarray],
                                   output_path: str = "models/quantized_model_dynamic.tflite") -> str:
        """
        Quantize model using dynamic range quantization
        
        Args:
            model_path: Path to the trained model
            representative_dataset: Representative dataset for calibration
            output_path: Output path for quantized model
            
        Returns:
            Path to quantized model
        """
        logger.info("Starting dynamic range quantization...")
        
        # Load the model
        model = load_model(model_path)
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set representative dataset
        def representative_dataset_gen():
            for sample in representative_dataset:
                yield [sample]
        
        converter.representative_dataset = representative_dataset_gen
        
        # Convert model
        quantized_model = converter.convert()
        
        # Save quantized model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(quantized_model)
        
        logger.info(f"Dynamic range quantized model saved to {output_path}")
        return output_path
    
    def quantize_model_float16(self, model_path: str,
                              output_path: str = "models/quantized_model_float16.tflite") -> str:
        """
        Quantize model using float16 precision
        
        Args:
            model_path: Path to the trained model
            output_path: Output path for quantized model
            
        Returns:
            Path to quantized model
        """
        logger.info("Starting float16 quantization...")
        
        # Load the model
        model = load_model(model_path)
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags for float16
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # Convert model
        quantized_model = converter.convert()
        
        # Save quantized model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(quantized_model)
        
        logger.info(f"Float16 quantized model saved to {output_path}")
        return output_path
    
    def quantize_model_int8(self, model_path: str,
                           representative_dataset: List[np.ndarray],
                           output_path: str = "models/quantized_model_int8.tflite") -> str:
        """
        Quantize model using int8 precision
        
        Args:
            model_path: Path to the trained model
            representative_dataset: Representative dataset for calibration
            output_path: Output path for quantized model
            
        Returns:
            Path to quantized model
        """
        logger.info("Starting int8 quantization...")
        
        # Load the model
        model = load_model(model_path)
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags for int8
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16]
        converter.target_spec.supported_types = [tf.int8]
        
        # Set representative dataset
        def representative_dataset_gen():
            for sample in representative_dataset:
                yield [sample]
        
        converter.representative_dataset = representative_dataset_gen
        
        # Convert model
        quantized_model = converter.convert()
        
        # Save quantized model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(quantized_model)
        
        logger.info(f"Int8 quantized model saved to {output_path}")
        return output_path
    
    def apply_pruning(self, model_path: str,
                     output_path: str = "models/pruned_model.h5") -> str:
        """
        Apply pruning to reduce model size
        
        Args:
            model_path: Path to the trained model
            output_path: Output path for pruned model
            
        Returns:
            Path to pruned model
        """
        logger.info("Applying model pruning...")
        
        # Load the model
        model = load_model(model_path)
        
        # Define pruning schedule
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=1000
        )
        
        # Apply pruning
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
            model, pruning_schedule=pruning_schedule
        )
        
        # Compile the pruned model
        model_for_pruning.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save pruned model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model_for_pruning.save(output_path)
        
        logger.info(f"Pruned model saved to {output_path}")
        return output_path
    
    def strip_pruning(self, pruned_model_path: str,
                     output_path: str = "models/stripped_model.h5") -> str:
        """
        Strip pruning wrappers from the model
        
        Args:
            pruned_model_path: Path to the pruned model
            output_path: Output path for stripped model
            
        Returns:
            Path to stripped model
        """
        logger.info("Stripping pruning wrappers...")
        
        # Load the pruned model
        model = load_model(pruned_model_path)
        
        # Strip pruning
        stripped_model = tfmot.sparsity.keras.strip_pruning(model)
        
        # Save stripped model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        stripped_model.save(output_path)
        
        logger.info(f"Stripped model saved to {output_path}")
        return output_path
    
    def convert_to_tflite(self, model_path: str,
                          output_path: str = "models/model.tflite") -> str:
        """
        Convert Keras model to TensorFlow Lite format
        
        Args:
            model_path: Path to the Keras model
            output_path: Output path for TFLite model
            
        Returns:
            Path to TFLite model
        """
        logger.info("Converting model to TensorFlow Lite format...")
        
        # Load the model
        model = load_model(model_path)
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save TFLite model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TensorFlow Lite model saved to {output_path}")
        return output_path
    
    def benchmark_model(self, model_path: str, test_dataset) -> Dict[str, float]:
        """
        Benchmark model performance and size
        
        Args:
            model_path: Path to the model
            test_dataset: Test dataset for inference
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Benchmarking model performance...")
        
        # Load interpreter
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare test data
        test_images = []
        test_labels = []
        
        for batch in test_dataset:
            images, labels = batch
            test_images.extend(images.numpy())
            test_labels.extend(labels.numpy())
            if len(test_images) >= 100:  # Limit for benchmarking
                break
        
        test_images = np.array(test_images[:100])
        test_labels = np.array(test_labels[:100])
        
        # Benchmark inference
        start_time = time.time()
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(len(test_images)):
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], test_images[i:i+1])
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output[0])
            true_class = np.argmax(test_labels[i])
            
            if predicted_class == true_class:
                correct_predictions += 1
            total_predictions += 1
        
        inference_time = time.time() - start_time
        accuracy = correct_predictions / total_predictions
        avg_inference_time = inference_time / total_predictions
        
        # Get model size
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        results = {
            'accuracy': accuracy,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'model_size_mb': model_size,
            'total_inference_time_s': inference_time
        }
        
        logger.info(f"Benchmark results - Accuracy: {accuracy:.4f}, "
                   f"Avg Inference Time: {avg_inference_time*1000:.2f}ms, "
                   f"Model Size: {model_size:.2f}MB")
        
        return results
    
    def compare_models(self, model_paths: List[str], test_dataset) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models performance
        
        Args:
            model_paths: List of model paths to compare
            test_dataset: Test dataset for inference
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing model performances...")
        
        results = {}
        
        for model_path in model_paths:
            model_name = os.path.basename(model_path).split('.')[0]
            logger.info(f"Benchmarking {model_name}...")
            
            try:
                benchmark_results = self.benchmark_model(model_path, test_dataset)
                results[model_name] = benchmark_results
            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def create_optimized_model(self, model_path: str, dataset,
                              output_dir: str = "models/") -> Dict[str, str]:
        """
        Create multiple optimized versions of the model
        
        Args:
            model_path: Path to the original model
            dataset: Dataset for representative sampling
            output_dir: Output directory for optimized models
            
        Returns:
            Dictionary with paths to optimized models
        """
        logger.info("Creating optimized model versions...")
        
        # Create representative dataset
        representative_dataset = self.create_representative_dataset(
            dataset, self.quantization_config['representative_dataset_size']
        )
        
        optimized_models = {}
        
        # Convert to TFLite
        if self.quantization_config['enable_dynamic_range']:
            try:
                dynamic_path = self.quantize_model_dynamic_range(
                    model_path, representative_dataset,
                    os.path.join(output_dir, "model_dynamic.tflite")
                )
                optimized_models['dynamic_range'] = dynamic_path
            except Exception as e:
                logger.error(f"Dynamic range quantization failed: {str(e)}")
        
        # Float16 quantization
        if self.quantization_config['enable_float16']:
            try:
                float16_path = self.quantize_model_float16(
                    model_path,
                    os.path.join(output_dir, "model_float16.tflite")
                )
                optimized_models['float16'] = float16_path
            except Exception as e:
                logger.error(f"Float16 quantization failed: {str(e)}")
        
        # Int8 quantization
        if self.quantization_config['enable_int8']:
            try:
                int8_path = self.quantize_model_int8(
                    model_path, representative_dataset,
                    os.path.join(output_dir, "model_int8.tflite")
                )
                optimized_models['int8'] = int8_path
            except Exception as e:
                logger.error(f"Int8 quantization failed: {str(e)}")
        
        # Pruning
        try:
            pruned_path = self.apply_pruning(
                model_path,
                os.path.join(output_dir, "model_pruned.h5")
            )
            stripped_path = self.strip_pruning(
                pruned_path,
                os.path.join(output_dir, "model_stripped.h5")
            )
            optimized_models['pruned'] = stripped_path
        except Exception as e:
            logger.error(f"Pruning failed: {str(e)}")
        
        logger.info(f"Created {len(optimized_models)} optimized model versions")
        return optimized_models

def main():
    """Main function to test model quantization"""
    try:
        # Import required modules
        from data_preprocessing import DataPreprocessor
        
        # Initialize components
        preprocessor = DataPreprocessor()
        quantizer = ModelQuantizer()
        
        # Load and preprocess data
        train_data, val_data, test_data = preprocessor.load_cifar10_data()
        train_data = preprocessor.preprocess_data(train_data)
        val_data = preprocessor.preprocess_data(val_data)
        test_data = preprocessor.preprocess_data(test_data)
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = preprocessor.create_tf_datasets(
            train_data, val_data, test_data
        )
        
        # Assume we have a trained model
        model_path = "models/object_detection_model.h5"
        
        if os.path.exists(model_path):
            # Create optimized models
            optimized_models = quantizer.create_optimized_model(
                model_path, train_dataset
            )
            
            # Benchmark all models
            all_model_paths = [model_path] + list(optimized_models.values())
            comparison_results = quantizer.compare_models(all_model_paths, test_dataset)
            
            logger.info("Model optimization and benchmarking completed!")
            return optimized_models, comparison_results
        else:
            logger.warning(f"Model file {model_path} not found. Please train a model first.")
            return {}, {}
        
    except Exception as e:
        logger.error(f"Model quantization failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 