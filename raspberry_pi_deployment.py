"""
Raspberry Pi Deployment Module
Handles model deployment to Raspberry Pi with setup instructions
and inference script generation
"""

import os
import yaml
import shutil
import subprocess
import logging
from typing import Dict, Any, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RaspberryPiDeployer:
    """Raspberry Pi deployment class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Raspberry Pi deployer"""
        self.config = self._load_config(config_path)
        self.deployment_config = self.config['deployment']
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def create_deployment_package(self, model_path: str, 
                                output_dir: str = "deployment/") -> str:
        """
        Create deployment package for Raspberry Pi
        
        Args:
            model_path: Path to the TensorFlow Lite model
            output_dir: Output directory for deployment package
            
        Returns:
            Path to deployment package
        """
        logger.info("Creating deployment package for Raspberry Pi...")
        
        # Create deployment directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy model file
        model_dest = os.path.join(output_dir, "model.tflite")
        shutil.copy2(model_path, model_dest)
        
        # Copy labels file
        labels_src = "models/labels.txt"
        if os.path.exists(labels_src):
            labels_dest = os.path.join(output_dir, "labels.txt")
            shutil.copy2(labels_src, labels_dest)
        
        # Create requirements file for Raspberry Pi
        pi_requirements = [
            "tensorflow-lite-runtime==2.15.0",
            "numpy==1.24.3",
            "opencv-python==4.8.1.78",
            "Pillow==10.0.1",
            "requests==2.31.0"
        ]
        
        with open(os.path.join(output_dir, "requirements.txt"), 'w') as f:
            for req in pi_requirements:
                f.write(f"{req}\n")
        
        # Create inference script
        self._create_inference_script(output_dir)
        
        # Create setup script
        self._create_setup_script(output_dir)
        
        # Create README
        self._create_readme(output_dir)
        
        # Create configuration file
        self._create_deployment_config(output_dir)
        
        logger.info(f"Deployment package created in {output_dir}")
        return output_dir
    
    def _create_inference_script(self, output_dir: str):
        """Create inference script for Raspberry Pi"""
        inference_script = '''#!/usr/bin/env python3
"""
Object Detection Inference Script for Raspberry Pi
Runs TensorFlow Lite model inference on images or camera feed
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
import logging
from typing import List, Tuple, Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetector:
    """Object detection class for Raspberry Pi"""
    
    def __init__(self, model_path: str, labels_path: str, config_path: str = "config.json"):
        """Initialize object detector"""
        self.model_path = model_path
        self.labels_path = labels_path
        self.config = self._load_config(config_path)
        
        # Load labels
        self.labels = self._load_labels()
        
        # Initialize interpreter
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        logger.info(f"Model input shape: {self.input_shape}")
        
        # Load configuration
        self.target_size = tuple(self.config.get('target_size', [224, 224]))
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.max_detections = self.config.get('max_detections', 10)
        
        logger.info("Object detector initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.warning(f"Failed to load config: {str(e)}, using defaults")
            return {}
    
    def _load_labels(self) -> List[str]:
        """Load class labels from file"""
        try:
            with open(self.labels_path, 'r') as f:
                labels = [line.strip().split(': ')[1] for line in f.readlines()]
            return labels
        except Exception as e:
            logger.error(f"Failed to load labels: {str(e)}")
            return []
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize image
        resized_image = cv2.resize(image, self.target_size)
        
        # Convert to RGB if needed
        if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        normalized_image = resized_image.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_image = np.expand_dims(normalized_image, axis=0)
        
        return input_image
    
    def run_inference(self, image: np.ndarray) -> Tuple[List[str], List[float]]:
        """Run inference on image"""
        # Preprocess image
        input_image = self.preprocess_image(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        
        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = time.time() - start_time
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Get predictions
        predictions = output[0]
        predicted_classes = []
        confidences = []
        
        # Get top predictions
        top_indices = np.argsort(predictions)[::-1][:self.max_detections]
        
        for idx in top_indices:
            confidence = predictions[idx]
            if confidence >= self.confidence_threshold:
                class_name = self.labels[idx] if idx < len(self.labels) else f"Class_{idx}"
                predicted_classes.append(class_name)
                confidences.append(float(confidence))
        
        logger.info(f"Inference completed in {inference_time*1000:.2f}ms")
        return predicted_classes, confidences
    
    def detect_from_image(self, image_path: str) -> Dict[str, Any]:
        """Detect objects from image file"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Run inference
            classes, confidences = self.run_inference(image)
            
            # Prepare results
            results = {
                'image_path': image_path,
                'detections': [],
                'total_detections': len(classes)
            }
            
            for i, (class_name, confidence) in enumerate(zip(classes, confidences)):
                detection = {
                    'id': i + 1,
                    'class': class_name,
                    'confidence': confidence,
                    'percentage': confidence * 100
                }
                results['detections'].append(detection)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return {'error': str(e)}
    
    def detect_from_camera(self, camera_id: int = 0, duration: int = 30):
        """Detect objects from camera feed"""
        logger.info(f"Starting camera detection for {duration} seconds...")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                # Check duration
                if time.time() - start_time > duration:
                    break
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue
                
                # Run inference
                classes, confidences = self.run_inference(frame)
                
                # Draw results on frame
                annotated_frame = self._draw_detections(frame, classes, confidences)
                
                # Display frame
                cv2.imshow('Object Detection', annotated_frame)
                
                # Handle key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                
        except KeyboardInterrupt:
            logger.info("Camera detection stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            fps = frame_count / (time.time() - start_time)
            logger.info(f"Camera detection completed - FPS: {fps:.2f}")
    
    def _draw_detections(self, image: np.ndarray, classes: List[str], 
                        confidences: List[float]) -> np.ndarray:
        """Draw detection results on image"""
        annotated_image = image.copy()
        
        # Draw detections
        for i, (class_name, confidence) in enumerate(zip(classes, confidences)):
            # Create text
            text = f"{class_name}: {confidence:.2f}"
            
            # Calculate position
            y_position = 30 + (i * 30)
            
            # Draw background rectangle
            cv2.rectangle(annotated_image, (10, y_position - 20), 
                         (300, y_position + 10), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(annotated_image, text, (15, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_image
    
    def benchmark_performance(self, test_images: List[str]) -> Dict[str, Any]:
        """Benchmark model performance"""
        logger.info("Starting performance benchmark...")
        
        total_time = 0
        total_predictions = 0
        
        for image_path in test_images:
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Run inference
                start_time = time.time()
                classes, confidences = self.run_inference(image)
                inference_time = time.time() - start_time
                
                total_time += inference_time
                total_predictions += 1
                
                logger.info(f"Processed {image_path} in {inference_time*1000:.2f}ms")
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
        
        if total_predictions > 0:
            avg_inference_time = total_time / total_predictions
            fps = 1.0 / avg_inference_time
            
            results = {
                'total_images': total_predictions,
                'avg_inference_time_ms': avg_inference_time * 1000,
                'fps': fps,
                'total_time_s': total_time
            }
            
            logger.info(f"Benchmark results - Avg time: {avg_inference_time*1000:.2f}ms, FPS: {fps:.2f}")
            return results
        else:
            logger.error("No images processed successfully")
            return {'error': 'No images processed'}

def main():
    """Main function for inference script"""
    parser = argparse.ArgumentParser(description='Object Detection on Raspberry Pi')
    parser.add_argument('--model', default='model.tflite', help='Path to TFLite model')
    parser.add_argument('--labels', default='labels.txt', help='Path to labels file')
    parser.add_argument('--config', default='config.json', help='Path to config file')
    parser.add_argument('--image', help='Path to input image')
    parser.add_argument('--camera', action='store_true', help='Use camera input')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID')
    parser.add_argument('--duration', type=int, default=30, help='Camera duration in seconds')
    parser.add_argument('--benchmark', help='Path to test images directory')
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = ObjectDetector(args.model, args.labels, args.config)
        
        if args.benchmark:
            # Run benchmark
            import glob
            test_images = glob.glob(os.path.join(args.benchmark, "*.jpg"))
            test_images.extend(glob.glob(os.path.join(args.benchmark, "*.png")))
            
            if test_images:
                results = detector.benchmark_performance(test_images)
                print(json.dumps(results, indent=2))
            else:
                logger.error("No test images found")
        
        elif args.camera:
            # Camera detection
            detector.detect_from_camera(args.camera_id, args.duration)
        
        elif args.image:
            # Image detection
            results = detector.detect_from_image(args.image)
            print(json.dumps(results, indent=2))
        
        else:
            logger.error("Please specify --image, --camera, or --benchmark")
    
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        with open(os.path.join(output_dir, "inference.py"), 'w') as f:
            f.write(inference_script)
        
        # Make script executable
        os.chmod(os.path.join(output_dir, "inference.py"), 0o755)
    
    def _create_setup_script(self, output_dir: str):
        """Create setup script for Raspberry Pi"""
        setup_script = '''#!/bin/bash
# Setup script for Raspberry Pi deployment

echo "Setting up Object Detection on Raspberry Pi..."

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install required packages
echo "Installing required packages..."
sudo apt-get install -y python3-pip python3-venv
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y libhdf5-dev libhdf5-serial-dev
sudo apt-get install -y libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt-get install -y libjasper-dev libqtcore4 libqt4-test
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libgtk-3-dev
sudo apt-get install -y libatlas-base-dev gfortran

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install TensorFlow Lite runtime
echo "Installing TensorFlow Lite runtime..."
pip install tensorflow-lite-runtime

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p test_images

# Set permissions
echo "Setting permissions..."
chmod +x inference.py

echo "Setup completed successfully!"
echo "To run inference:"
echo "  source venv/bin/activate"
echo "  python3 inference.py --image test_images/sample.jpg"
echo "  python3 inference.py --camera"
'''
        
        with open(os.path.join(output_dir, "setup.sh"), 'w') as f:
            f.write(setup_script)
        
        # Make script executable
        os.chmod(os.path.join(output_dir, "setup.sh"), 0o755)
    
    def _create_readme(self, output_dir: str):
        """Create README for deployment"""
        readme_content = '''# Object Detection on Raspberry Pi

This package contains the necessary files to run object detection on a Raspberry Pi using TensorFlow Lite.

## Files

- `model.tflite`: TensorFlow Lite model for object detection
- `labels.txt`: Class labels for the model
- `inference.py`: Main inference script
- `setup.sh`: Setup script for Raspberry Pi
- `requirements.txt`: Python dependencies
- `config.json`: Configuration file

## Setup

1. Copy this package to your Raspberry Pi
2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

## Usage

### Image Detection
```bash
source venv/bin/activate
python3 inference.py --image path/to/image.jpg
```

### Camera Detection
```bash
source venv/bin/activate
python3 inference.py --camera
```

### Performance Benchmark
```bash
source venv/bin/activate
python3 inference.py --benchmark test_images/
```

## Configuration

Edit `config.json` to modify:
- Target image size
- Confidence threshold
- Maximum detections

## Troubleshooting

1. **Camera not working**: Check camera permissions and connections
2. **Slow performance**: Consider using a smaller model or optimizing settings
3. **Memory issues**: Close other applications to free up RAM

## Performance Tips

1. Use a USB 3.0 camera for better performance
2. Overclock the Raspberry Pi (with proper cooling)
3. Use an SSD for faster storage access
4. Close unnecessary background processes

## Security Notes

- Validate input images before processing
- Limit file sizes to prevent memory issues
- Use secure file transfer methods
- Monitor system resources during inference
'''
        
        with open(os.path.join(output_dir, "README.md"), 'w') as f:
            f.write(readme_content)
    
    def _create_deployment_config(self, output_dir: str):
        """Create deployment configuration file"""
        config = {
            'target_size': self.deployment_config['target_size'],
            'confidence_threshold': self.deployment_config['confidence_threshold'],
            'max_detections': self.deployment_config['max_detections'],
            'model_path': 'model.tflite',
            'labels_path': 'labels.txt',
            'log_level': 'INFO',
            'enable_gpu': False,
            'num_threads': 4
        }
        
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
    
    def create_transfer_script(self, pi_ip: str, pi_user: str, 
                             local_path: str, remote_path: str = "~/object_detection/") -> str:
        """
        Create script to transfer files to Raspberry Pi
        
        Args:
            pi_ip: Raspberry Pi IP address
            pi_user: Raspberry Pi username
            local_path: Local deployment package path
            remote_path: Remote path on Raspberry Pi
            
        Returns:
            Path to transfer script
        """
        transfer_script = f'''#!/bin/bash
# Transfer script for Raspberry Pi deployment

PI_IP="{pi_ip}"
PI_USER="{pi_user}"
LOCAL_PATH="{local_path}"
REMOTE_PATH="{remote_path}"

echo "Transferring files to Raspberry Pi..."

# Create remote directory
ssh $PI_USER@$PI_IP "mkdir -p $REMOTE_PATH"

# Transfer files
scp -r $LOCAL_PATH/* $PI_USER@$PI_IP:$REMOTE_PATH/

echo "Files transferred successfully!"
echo "SSH into Raspberry Pi and run setup:"
echo "  ssh $PI_USER@$PI_IP"
echo "  cd $REMOTE_PATH"
echo "  ./setup.sh"
'''
        
        script_path = "transfer_to_pi.sh"
        with open(script_path, 'w') as f:
            f.write(transfer_script)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Transfer script created: {script_path}")
        return script_path
    
    def generate_deployment_instructions(self, model_path: str) -> str:
        """
        Generate complete deployment instructions
        
        Args:
            model_path: Path to the TensorFlow Lite model
            
        Returns:
            Path to instructions file
        """
        instructions = f'''# Raspberry Pi Deployment Instructions

## Prerequisites

1. Raspberry Pi 3 or 4 (recommended)
2. MicroSD card with Raspberry Pi OS
3. Camera module (optional)
4. Network connection

## Step 1: Prepare the Model

The TensorFlow Lite model has been optimized for Raspberry Pi deployment:
- Model: {model_path}
- Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB
- Optimized for: ARM architecture

## Step 2: Transfer Files

### Option A: Using SCP
```bash
# Create deployment package
python3 raspberry_pi_deployment.py

# Transfer to Raspberry Pi
scp -r deployment/ pi@<raspberry_pi_ip>:~/object_detection/
```

### Option B: Using USB/SD Card
1. Copy the deployment folder to a USB drive or SD card
2. Insert into Raspberry Pi
3. Copy files to home directory

## Step 3: Setup on Raspberry Pi

1. SSH into Raspberry Pi:
   ```bash
   ssh pi@<raspberry_pi_ip>
   ```

2. Navigate to deployment directory:
   ```bash
   cd ~/object_detection/
   ```

3. Run setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

## Step 4: Test the Deployment

### Test with sample image:
```bash
source venv/bin/activate
python3 inference.py --image test_images/sample.jpg
```

### Test with camera:
```bash
source venv/bin/activate
python3 inference.py --camera
```

## Step 5: Performance Optimization

1. **Overclock Raspberry Pi** (with proper cooling):
   ```bash
   sudo raspi-config
   # Navigate to Overclock and select appropriate setting
   ```

2. **Enable GPU memory split**:
   ```bash
   sudo raspi-config
   # Advanced Options > Memory Split > 128
   ```

3. **Optimize system settings**:
   ```bash
   # Add to /boot/config.txt
   gpu_mem=128
   arm_freq=1750
   ```

## Troubleshooting

### Common Issues:

1. **Camera not detected**:
   ```bash
   sudo raspi-config
   # Interface Options > Camera > Enable
   ```

2. **Slow performance**:
   - Check CPU temperature: `vcgencmd measure_temp`
   - Close unnecessary processes
   - Use smaller model if needed

3. **Memory errors**:
   - Increase swap space
   - Close other applications
   - Use lighter model

4. **Import errors**:
   ```bash
   source venv/bin/activate
   pip install --upgrade tensorflow-lite-runtime
   ```

## Security Considerations

1. **Input validation**: The inference script validates input images
2. **File size limits**: Prevents memory overflow attacks
3. **Secure transfer**: Use SCP or secure file transfer methods
4. **Network security**: Configure firewall rules if needed

## Monitoring

Monitor system resources during inference:
```bash
# CPU and memory usage
htop

# GPU memory usage
vcgencmd get_mem gpu

# Temperature
vcgencmd measure_temp
```

## Performance Benchmarks

Expected performance on Raspberry Pi 4:
- Inference time: 50-200ms per image
- FPS: 5-20 (depending on model size)
- Memory usage: 100-500MB

## Support

For issues and questions:
1. Check the logs in the deployment directory
2. Verify all dependencies are installed
3. Test with smaller images first
4. Monitor system resources
'''
        
        instructions_path = "deployment_instructions.md"
        with open(instructions_path, 'w') as f:
            f.write(instructions)
        
        logger.info(f"Deployment instructions created: {instructions_path}")
        return instructions_path

def main():
    """Main function to test Raspberry Pi deployment"""
    try:
        # Initialize deployer
        deployer = RaspberryPiDeployer()
        
        # Check if model exists
        model_path = "models/model.tflite"
        if not os.path.exists(model_path):
            # Try to find any TFLite model
            for root, dirs, files in os.walk("models"):
                for file in files:
                    if file.endswith(".tflite"):
                        model_path = os.path.join(root, file)
                        break
                if os.path.exists(model_path):
                    break
        
        if os.path.exists(model_path):
            # Create deployment package
            deployment_dir = deployer.create_deployment_package(model_path)
            
            # Generate instructions
            instructions_path = deployer.generate_deployment_instructions(model_path)
            
            logger.info("Raspberry Pi deployment package created successfully!")
            logger.info(f"Deployment directory: {deployment_dir}")
            logger.info(f"Instructions: {instructions_path}")
            
            return deployment_dir, instructions_path
        else:
            logger.warning("No TensorFlow Lite model found. Please train and convert a model first.")
            return None, None
        
    except Exception as e:
        logger.error(f"Raspberry Pi deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 