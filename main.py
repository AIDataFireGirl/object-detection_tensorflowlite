#!/usr/bin/env python3
"""
Main Execution Script for Object Detection with TensorFlow Lite
Orchestrates the complete pipeline from data preprocessing to Raspberry Pi deployment
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('object_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

def create_project_structure():
    """Create necessary project directories"""
    directories = [
        "models",
        "logs",
        "checkpoints",
        "hyperparameter_tuning",
        "deployment",
        "test_images",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def run_data_preprocessing(config: Dict[str, Any]) -> Tuple:
    """Run data preprocessing pipeline"""
    logger.info("=== Starting Data Preprocessing ===")
    
    try:
        from data_preprocessing import DataPreprocessor
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Load CIFAR10 data
        train_data, val_data, test_data = preprocessor.load_cifar10_data()
        
        # Preprocess data
        train_data = preprocessor.preprocess_data(train_data)
        val_data = preprocessor.preprocess_data(val_data)
        test_data = preprocessor.preprocess_data(test_data)
        
        # Create TensorFlow datasets
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

def run_model_training(config: Dict[str, Any], train_dataset, val_dataset, 
                      enable_tuning: bool = False) -> Tuple:
    """Run model training pipeline"""
    logger.info("=== Starting Model Training ===")
    
    try:
        from model_training import ModelTrainer
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Train model
        model, history = trainer.train_model(train_dataset, val_dataset, enable_tuning)
        
        # Save model
        model_path = "models/object_detection_model.h5"
        trainer.save_model(model, model_path)
        
        # Plot training history
        trainer.plot_training_history(history, "results/training_history.png")
        
        logger.info("Model training completed successfully!")
        return model, history, model_path
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

def run_model_evaluation(model, test_dataset, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run model evaluation"""
    logger.info("=== Starting Model Evaluation ===")
    
    try:
        from model_training import ModelTrainer
        
        trainer = ModelTrainer()
        results = trainer.evaluate_model(model, test_dataset)
        
        # Save evaluation results
        import json
        with open("results/evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Model evaluation completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise

def run_model_quantization(config: Dict[str, Any], train_dataset, test_dataset) -> Dict[str, str]:
    """Run model quantization pipeline"""
    logger.info("=== Starting Model Quantization ===")
    
    try:
        from model_quantization import ModelQuantizer
        
        # Initialize quantizer
        quantizer = ModelQuantizer()
        
        # Check if trained model exists
        model_path = "models/object_detection_model.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        
        # Create optimized models
        optimized_models = quantizer.create_optimized_model(model_path, train_dataset)
        
        # Benchmark models
        all_model_paths = [model_path] + list(optimized_models.values())
        comparison_results = quantizer.compare_models(all_model_paths, test_dataset)
        
        # Save comparison results
        import json
        with open("results/model_comparison.json", 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        logger.info("Model quantization completed successfully!")
        return optimized_models, comparison_results
        
    except Exception as e:
        logger.error(f"Model quantization failed: {str(e)}")
        raise

def run_raspberry_pi_deployment(config: Dict[str, Any], optimized_models: Dict[str, str]):
    """Run Raspberry Pi deployment pipeline"""
    logger.info("=== Starting Raspberry Pi Deployment ===")
    
    try:
        from raspberry_pi_deployment import RaspberryPiDeployer
        
        # Initialize deployer
        deployer = RaspberryPiDeployer()
        
        # Select best model for deployment (prefer smaller, faster models)
        deployment_model = None
        model_priority = ['int8', 'dynamic_range', 'float16', 'pruned']
        
        for model_type in model_priority:
            if model_type in optimized_models:
                deployment_model = optimized_models[model_type]
                break
        
        if not deployment_model:
            # Fallback to original model
            deployment_model = "models/object_detection_model.h5"
        
        # Create deployment package
        deployment_dir = deployer.create_deployment_package(deployment_model)
        
        # Generate instructions
        instructions_path = deployer.generate_deployment_instructions(deployment_model)
        
        logger.info("Raspberry Pi deployment completed successfully!")
        return deployment_dir, instructions_path
        
    except Exception as e:
        logger.error(f"Raspberry Pi deployment failed: {str(e)}")
        raise

def create_flowchart():
    """Create project execution flowchart"""
    flowchart = '''
```mermaid
flowchart TD
    A[Start] --> B[Load Configuration]
    B --> C[Create Project Structure]
    C --> D[Data Preprocessing]
    D --> E[CIFAR10 Dataset Loading]
    E --> F[Data Augmentation]
    F --> G[Data Splitting]
    G --> H[Model Training]
    H --> I[Hyperparameter Tuning]
    I --> J[Early Stopping]
    J --> K[Model Checkpointing]
    K --> L[TensorBoard Logging]
    L --> M[Model Evaluation]
    M --> N[Confusion Matrix]
    N --> O[Model Quantization]
    O --> P[Dynamic Range Quantization]
    P --> Q[Float16 Quantization]
    Q --> R[Int8 Quantization]
    R --> S[Model Pruning]
    S --> T[Performance Benchmarking]
    T --> U[Raspberry Pi Deployment]
    U --> V[Create Deployment Package]
    V --> W[Generate Inference Script]
    W --> X[Create Setup Instructions]
    X --> Y[End]

    style A fill:#e1f5fe
    style Y fill:#c8e6c9
    style H fill:#fff3e0
    style O fill:#f3e5f5
    style U fill:#e8f5e8
```
'''
    
    with open("project_flowchart.md", 'w') as f:
        f.write("# Object Detection Project Flowchart\n\n")
        f.write(flowchart)
    
    logger.info("Project flowchart created: project_flowchart.md")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Object Detection Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--skip-quantization', action='store_true', help='Skip model quantization')
    parser.add_argument('--skip-deployment', action='store_true', help='Skip Raspberry Pi deployment')
    parser.add_argument('--enable-tuning', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--flowchart', action='store_true', help='Create project flowchart')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Create project structure
        create_project_structure()
        
        # Create flowchart if requested
        if args.flowchart:
            create_flowchart()
        
        # Step 1: Data Preprocessing
        if not args.skip_training:
            train_dataset, val_dataset, test_dataset, class_names = run_data_preprocessing(config)
        else:
            logger.info("Skipping data preprocessing (training skipped)")
            train_dataset = val_dataset = test_dataset = class_names = None
        
        # Step 2: Model Training
        if not args.skip_training:
            model, history, model_path = run_model_training(config, train_dataset, val_dataset, args.enable_tuning)
            
            # Step 3: Model Evaluation
            evaluation_results = run_model_evaluation(model, test_dataset, config)
        else:
            logger.info("Skipping model training and evaluation")
            model = history = model_path = evaluation_results = None
        
        # Step 4: Model Quantization
        if not args.skip_quantization and not args.skip_training:
            optimized_models, comparison_results = run_model_quantization(config, train_dataset, test_dataset)
        else:
            logger.info("Skipping model quantization")
            optimized_models = comparison_results = {}
        
        # Step 5: Raspberry Pi Deployment
        if not args.skip_deployment:
            deployment_dir, instructions_path = run_raspberry_pi_deployment(config, optimized_models)
        else:
            logger.info("Skipping Raspberry Pi deployment")
            deployment_dir = instructions_path = None
        
        # Summary
        logger.info("=== Pipeline Execution Summary ===")
        logger.info("✅ Project structure created")
        
        if not args.skip_training:
            logger.info("✅ Data preprocessing completed")
            logger.info("✅ Model training completed")
            logger.info("✅ Model evaluation completed")
        
        if not args.skip_quantization and not args.skip_training:
            logger.info(f"✅ Model quantization completed ({len(optimized_models)} optimized models)")
        
        if not args.skip_deployment:
            logger.info("✅ Raspberry Pi deployment package created")
        
        logger.info("=== Pipeline completed successfully! ===")
        
        # Print next steps
        print("\n" + "="*50)
        print("NEXT STEPS:")
        print("="*50)
        
        if deployment_dir:
            print(f"1. Transfer deployment package to Raspberry Pi:")
            print(f"   scp -r {deployment_dir} pi@<raspberry_pi_ip>:~/object_detection/")
            print(f"2. Follow instructions in: {instructions_path}")
        
        print("3. View training logs: tensorboard --logdir logs/")
        print("4. Check results in: results/")
        print("5. View project flowchart: project_flowchart.md")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 