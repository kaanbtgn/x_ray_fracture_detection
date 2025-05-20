#!/usr/bin/env python3
"""
X-ray Fracture Detection with DenseNet121
Optimized for both Apple Silicon and CUDA GPUs
"""

import os
import platform
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    CSVLogger
)
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
import click
import json

# ---------------------------- CONFIGURATION ----------------------------
IMG_SIZE = (224, 224)
CHANNELS = 3
BATCH_SIZES = {
    'apple_silicon': 32,  # Optimized for M1 Pro
    'cuda': 16,          # Optimized for GTX 1650
    'cpu': 8
}
AUTOTUNE = tf.data.AUTOTUNE

# ---------------------------- LOGGING SETUP ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------- HARDWARE SETUP ----------------------------
def setup_hardware():
    """Configure TensorFlow for optimal performance on available hardware."""
    try:
        # First check for Apple Silicon
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            # Configure for Metal
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                logger.info(f"✅ Apple Silicon detected - Using Metal GPU: {physical_devices[0].name}")
                return 'apple_silicon', BATCH_SIZES['apple_silicon']
            else:
                logger.warning("⚠️ Apple Silicon detected but no Metal GPU found")
                return 'cpu', BATCH_SIZES['cpu']
        
        # Then check for CUDA GPU
        gpus = tf.config.list_physical_devices('GPU')
            if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ CUDA GPU detected: {gpus[0].name}")
            return 'cuda', BATCH_SIZES['cuda']
        
        logger.warning("⚠️ No GPU detected, using CPU")
        return 'cpu', BATCH_SIZES['cpu']
        except Exception as e:
        logger.error(f"❌ Hardware setup failed: {e}")
        return 'cpu', BATCH_SIZES['cpu']

# ---------------------------- DATA PIPELINE ----------------------------
class FractureDataLoader:
    def __init__(self, data_dir: str, batch_size: int):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self._verify_dataset()
        # Set a fixed seed for reproducibility
        self.seed = 42
    
    def _verify_dataset(self):
        """Verify dataset structure and integrity."""
        required_splits = ['train', 'val', 'test']
        required_folders = ['images', 'not_fractured']
        
        for split in required_splits:
            split_path = self.data_dir / split
            if not split_path.exists():
                raise FileNotFoundError(f"Missing split directory: {split_path}")
            
            for folder in required_folders:
                folder_path = split_path / folder
                if not folder_path.exists():
                    raise FileNotFoundError(f"Missing folder: {folder_path}")
                
                # Verify images exist
                images = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))
                if not images:
                    raise ValueError(f"No images found in {folder_path}")
                logger.info(f"Found {len(images)} images in {folder_path}")

    def _load_and_preprocess_image(self, image_path, label):
        """Load and preprocess a single image."""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=CHANNELS)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    def _create_dataset(self, split: str, augment: bool = False):
        """Create a dataset for the specified split."""
        split_path = self.data_dir / split
        
        # Load fractured images (from 'images' folder)
        fractured_paths = tf.data.Dataset.list_files(
            str(split_path / 'images/*'), shuffle=True, seed=self.seed
        )
        fractured_ds = fractured_paths.map(
            lambda x: self._load_and_preprocess_image(x, 1.0),
            num_parallel_calls=AUTOTUNE
        )
        
        # Load non-fractured images
        normal_paths = tf.data.Dataset.list_files(
            str(split_path / 'not_fractured/*'), shuffle=True, seed=self.seed
        )
        normal_ds = normal_paths.map(
            lambda x: self._load_and_preprocess_image(x, 0.0),
            num_parallel_calls=AUTOTUNE
        )
        
        # Combine datasets
        dataset = tf.data.Dataset.concatenate(fractured_ds, normal_ds)
        
        if augment and split == 'train':
            dataset = dataset.map(self._augment, num_parallel_calls=AUTOTUNE)
        
        # Cache the dataset to avoid reloading
        dataset = dataset.cache()
        
        # Shuffle with a fixed seed for reproducibility
        dataset = dataset.shuffle(1000, seed=self.seed, reshuffle_each_iteration=False)
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(AUTOTUNE)
        
        return dataset

    def _augment(self, image, label):
        """Apply data augmentation."""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        return image, label

    def get_datasets(self):
        """Get train, validation, and test datasets."""
        return (
            self._create_dataset('train', augment=True),
            self._create_dataset('val'),
            self._create_dataset('test')
        )

# ---------------------------- MODEL ARCHITECTURE ----------------------------
class FractureModel:
    def __init__(self, input_shape=(*IMG_SIZE, CHANNELS)):
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self):
        """Build and compile the model."""
        # Load pre-trained DenseNet121
        base_model = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        
        # Fine-tune the last 25% of layers
        for layer in base_model.layers[:-len(base_model.layers)//4]:
            layer.trainable = False
            
        # Build model with gradient clipping and batch normalization
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model with gradient clipping
        optimizer = optimizers.Adam(
            learning_rate=1e-4,
            clipnorm=1.0  # Add gradient clipping
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model

# ---------------------------- TRAINING ----------------------------
class FractureTrainer:
    def __init__(self, data_dir: str, batch_size: int):
        self.data_loader = FractureDataLoader(data_dir, batch_size)
        self.model = FractureModel().model
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup training callbacks."""
        class MetricsLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                lr = self.model.optimizer.learning_rate.numpy()
                logger.info(
                    f"Epoch {epoch+1} :: "
                    f"Train Loss: {logs['loss']:.4f} | "
                    f"Val Loss: {logs['val_loss']:.4f} | "
                    f"Train Acc: {logs['accuracy']:.4f} | "
                    f"Val Acc: {logs['val_accuracy']:.4f} | "
                    f"Train AUC: {logs['auc']:.4f} | "
                    f"Val AUC: {logs['val_auc']:.4f} | "
                    f"Train Prec: {logs['precision']:.4f} | "
                    f"Val Prec: {logs['val_precision']:.4f} | "
                    f"Train Rec: {logs['recall']:.4f} | "
                    f"Val Rec: {logs['val_recall']:.4f} | "
                    f"LR: {lr:.2e}"
                )

        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)

        self.callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                mode='min',
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_model.keras',
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            CSVLogger('training_history.csv'),
            MetricsLogger()
        ]

    def _calculate_optimal_threshold(self, val_ds):
        """Calculate optimal threshold using validation data."""
        logger.info("Calculating optimal threshold...")
        
        # Get predictions and true labels
        y_pred = []
        y_true = []
        
        for images, labels in val_ds:
            preds = self.model.predict(images, verbose=0)
            y_pred.extend(preds.flatten())
            y_true.extend(labels.numpy())
        
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        
        # Find threshold that maximizes Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate metrics at optimal threshold
        y_pred_binary = (y_pred >= optimal_threshold).astype(int)
        accuracy = np.mean(y_pred_binary == y_true)
        precision = np.sum((y_pred_binary == 1) & (y_true == 1)) / np.sum(y_pred_binary == 1)
        recall = np.sum((y_pred_binary == 1) & (y_true == 1)) / np.sum(y_true == 1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        auc = tf.keras.metrics.AUC()(y_true, y_pred).numpy()
        
        # Save metrics
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "auc": float(auc),
            "f1_score": float(f1_score)
        }
        
        # Save metrics to JSON file
        with open('models/training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        logger.info(f"Metrics at optimal threshold:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  AUC: {auc:.4f}")
        logger.info(f"  F1 Score: {f1_score:.4f}")
        
        # Save threshold
        with open('models/best_threshold.txt', 'w') as f:
            f.write(str(optimal_threshold))
        
        return optimal_threshold
    
    def train(self, epochs: int = 20):
        """Train the model."""
        train_ds, val_ds, test_ds = self.data_loader.get_datasets()
        
        # Train the model with verbose=2 for more detailed progress
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=self.callbacks,
            verbose=2  # Show progress bar with metrics
        )
        
        # Calculate and save optimal threshold
        optimal_threshold = self._calculate_optimal_threshold(val_ds)
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set:")
        test_results = self.model.evaluate(test_ds, verbose=2)
        
        # Log final metrics
        metrics = dict(zip(self.model.metrics_names, test_results))
        logger.info("\nFinal Test Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return history, metrics, optimal_threshold

# ---------------------------- MAIN ----------------------------
@click.command()
@click.option('--data-dir', default='dataset', help='Path to dataset directory')
@click.option('--epochs', default=20, help='Number of training epochs')
def main(data_dir: str, epochs: int):
    """Main training function."""
    try:
        # Setup hardware
        device_type, batch_size = setup_hardware()
        logger.info(f"🚀 Starting training on {device_type} with batch size {batch_size}")
        
        # Initialize and train
        trainer = FractureTrainer(data_dir, batch_size)
        history, metrics, optimal_threshold = trainer.train(epochs)
        
        logger.info(f"✅ Training completed successfully")
        logger.info(f"📊 Optimal threshold: {optimal_threshold:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()