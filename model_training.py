"""
Model Training Module for Object Detection
Handles model creation, training with callbacks, hyperparameter tuning,
and TensorBoard integration
"""

import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
import keras_tuner
from datetime import datetime
import logging
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelBuilder:
    """Model builder class for creating CNN models"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model builder with configuration"""
        self.config = config
        self.input_shape = tuple(config['data']['input_shape'])
        self.num_classes = config['data']['num_classes']
        self.base_model_name = config['model']['architecture']
        
    def build_model(self, hp=None) -> Model:
        """
        Build CNN model with optional hyperparameter tuning
        
        Args:
            hp: Keras Tuner hyperparameter object
            
        Returns:
            Compiled Keras model
        """
        # Use hyperparameters if provided (for tuning)
        if hp:
            learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
            dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
            num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=3)
            dense_units = hp.Int('dense_units', min_value=64, max_value=512, step=64)
        else:
            learning_rate = self.config['model']['learning_rate']
            dropout_rate = self.config['model']['dropout_rate']
            num_dense_layers = 2
            dense_units = 128
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Data augmentation layer
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Resize to MobileNetV2 expected input size
        x = layers.Resizing(224, 224)(x)
        
        # Preprocessing for MobileNetV2
        x = layers.Rescaling(1./127.5, offset=-1)(x)
        
        # Base model (MobileNetV2)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_tensor=x
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        # Add dense layers with dropout
        for i in range(num_dense_layers):
            x = layers.Dense(dense_units // (2 ** i), activation='relu')(x)
            x = layers.Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        logger.info(f"Model built successfully - Parameters: {model.count_params():,}")
        return model
    
    def fine_tune_model(self, model: Model, fine_tune_at: int = 100) -> Model:
        """
        Fine-tune the model by unfreezing some layers
        
        Args:
            model: Pre-trained model
            fine_tune_at: Number of layers to unfreeze from the end
            
        Returns:
            Fine-tuned model
        """
        # Get the base model
        base_model = model.layers[4]  # MobileNetV2 layer
        
        # Unfreeze the top layers
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['model']['learning_rate'] / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        logger.info(f"Model fine-tuned - Unfrozen layers: {fine_tune_at}")
        return model

class CallbackManager:
    """Manages training callbacks"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize callback manager"""
        self.config = config
        self.log_dir = config['tensorboard']['log_dir']
        self.checkpoint_dir = config['checkpoints']['directory']
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def get_callbacks(self, model_name: str = "object_detection_model") -> list:
        """
        Get list of training callbacks
        
        Args:
            model_name: Name for the model checkpoint
            
        Returns:
            List of callbacks
        """
        callbacks_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor=self.config['checkpoints']['monitor'],
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)
        
        # Model checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}.h5")
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=self.config['checkpoints']['monitor'],
            mode=self.config['checkpoints']['mode'],
            save_best_only=self.config['checkpoints']['save_best_only'],
            save_weights_only=self.config['checkpoints']['save_weights_only'],
            verbose=1
        )
        callbacks_list.append(model_checkpoint)
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config['training']['reduce_lr_factor'],
            patience=self.config['training']['reduce_lr_patience'],
            min_lr=self.config['training']['min_lr'],
            verbose=1
        )
        callbacks_list.append(reduce_lr)
        
        # TensorBoard
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_log_dir = os.path.join(self.log_dir, current_time)
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=tensorboard_log_dir,
            update_freq=self.config['tensorboard']['update_freq'],
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks_list.append(tensorboard_callback)
        
        # Custom callback for logging
        class LoggingCallback(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logger.info(f"Epoch {epoch + 1} - Loss: {logs['loss']:.4f}, "
                          f"Accuracy: {logs['accuracy']:.4f}, "
                          f"Val Loss: {logs['val_loss']:.4f}, "
                          f"Val Accuracy: {logs['val_accuracy']:.4f}")
        
        callbacks_list.append(LoggingCallback())
        
        logger.info(f"Callbacks created - TensorBoard logs: {tensorboard_log_dir}")
        return callbacks_list

class HyperparameterTuner:
    """Hyperparameter tuning using Keras Tuner"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hyperparameter tuner"""
        self.config = config
        self.tuning_config = config['hyperparameter_tuning']
        
    def create_tuner(self, model_builder, train_dataset, val_dataset) -> keras_tuner.Hyperband:
        """
        Create hyperparameter tuner
        
        Args:
            model_builder: Model builder function
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Keras Tuner object
        """
        # Define objective
        objective = keras_tuner.Objective(
            name=self.tuning_config['objective'],
            direction='max'
        )
        
        # Create tuner
        tuner = keras_tuner.Hyperband(
            model_builder,
            objective=objective,
            max_epochs=50,
            factor=3,
            directory=self.tuning_config['directory'],
            project_name=self.tuning_config['project_name'],
            executions_per_trial=self.tuning_config['executions_per_trial']
        )
        
        logger.info("Hyperparameter tuner created successfully")
        return tuner
    
    def tune_hyperparameters(self, tuner: keras_tuner.Hyperband, 
                           train_dataset, val_dataset) -> Model:
        """
        Perform hyperparameter tuning
        
        Args:
            tuner: Keras Tuner object
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Best model found
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Search for best hyperparameters
        tuner.search(
            train_dataset,
            validation_data=val_dataset,
            epochs=50,
            callbacks=[
                callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )
        
        # Get best model
        best_model = tuner.get_best_models(1)[0]
        
        logger.info("Hyperparameter tuning completed!")
        return best_model

class ModelTrainer:
    """Main model training class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize model trainer"""
        self.config = self._load_config(config_path)
        self.model_builder = ModelBuilder(self.config)
        self.callback_manager = CallbackManager(self.config)
        self.hyperparameter_tuner = HyperparameterTuner(self.config)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def train_model(self, train_dataset, val_dataset, 
                   enable_tuning: bool = False) -> Tuple[Model, Dict[str, Any]]:
        """
        Train the model with optional hyperparameter tuning
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            enable_tuning: Whether to enable hyperparameter tuning
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        if enable_tuning:
            logger.info("Training with hyperparameter tuning...")
            
            # Create tuner
            tuner = self.hyperparameter_tuner.create_tuner(
                self.model_builder.build_model,
                train_dataset,
                val_dataset
            )
            
            # Perform tuning
            best_model = self.hyperparameter_tuner.tune_hyperparameters(
                tuner, train_dataset, val_dataset
            )
            
            # Train the best model
            callbacks_list = self.callback_manager.get_callbacks()
            history = best_model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config['training']['epochs'],
                callbacks=callbacks_list,
                verbose=1
            )
            
        else:
            logger.info("Training with default hyperparameters...")
            
            # Build model
            model = self.model_builder.build_model()
            
            # Get callbacks
            callbacks_list = self.callback_manager.get_callbacks()
            
            # Train model
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config['training']['epochs'],
                callbacks=callbacks_list,
                verbose=1
            )
            
            best_model = model
        
        logger.info("Model training completed!")
        return best_model, history.history
    
    def evaluate_model(self, model: Model, test_dataset) -> Dict[str, float]:
        """
        Evaluate model on test dataset
        
        Args:
            model: Trained model
            test_dataset: Test dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model...")
        
        # Evaluate model
        test_loss, test_accuracy, test_top3_accuracy = model.evaluate(
            test_dataset, verbose=1
        )
        
        # Make predictions
        predictions = model.predict(test_dataset)
        
        # Calculate additional metrics
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np
        
        # Get true labels
        y_true = []
        y_pred = []
        
        for batch in test_dataset:
            images, labels = batch
            batch_pred = model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(batch_pred, axis=1))
        
        # Classification report
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Compile results
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top3_accuracy': test_top3_accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        logger.info(f"Model evaluation completed - Accuracy: {test_accuracy:.4f}")
        return results
    
    def save_model(self, model: Model, model_path: str = "models/object_detection_model.h5"):
        """Save trained model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def plot_training_history(self, history: Dict[str, Any], save_path: str = "training_history.png"):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Training history plot saved to {save_path}")

def main():
    """Main function to test model training"""
    try:
        # Import data preprocessing
        from data_preprocessing import DataPreprocessor
        
        # Initialize components
        preprocessor = DataPreprocessor()
        trainer = ModelTrainer()
        
        # Load and preprocess data
        train_data, val_data, test_data = preprocessor.load_cifar10_data()
        train_data = preprocessor.preprocess_data(train_data)
        val_data = preprocessor.preprocess_data(val_data)
        test_data = preprocessor.preprocess_data(test_data)
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = preprocessor.create_tf_datasets(
            train_data, val_data, test_data
        )
        
        # Train model
        model, history = trainer.train_model(train_dataset, val_dataset, enable_tuning=False)
        
        # Evaluate model
        results = trainer.evaluate_model(model, test_dataset)
        
        # Save model
        trainer.save_model(model)
        
        # Plot training history
        trainer.plot_training_history(history)
        
        logger.info("Model training and evaluation completed successfully!")
        
        return model, results
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 