import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from init_model import build_tree_health_model, CLASSES

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODEL_DIR, '..'))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
SAVED_MODEL_PATH = os.path.join(MODEL_DIR, 'tree_health_mobilenetv2.h5')

BATCH_SIZE = 16
IMG_SIZE = (224, 224)
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 15

def ensure_dataset_exists():
    """Checks if training dataset exists; if not, triggers generator."""
    has_images = False
    if os.path.exists(DATASET_DIR):
        for sub in os.listdir(DATASET_DIR):
            sub_path = os.path.join(DATASET_DIR, sub)
            if os.path.isdir(sub_path):
                imgs = [f for f in os.listdir(sub_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                if len(imgs) > 0:
                    has_images = True
                    break
    
    if not has_images:
        print("No training images found in dataset directory. Generating dataset...")
        import sys
        if DATASET_DIR not in sys.path:
            sys.path.append(DATASET_DIR)
        from generate_dataset import create_dataset
        create_dataset()

def compute_class_weights(generator):
    """Calculates balanced class weights to prevent class imbalance skew."""
    class_counts = np.bincount(generator.classes)
    total_samples = len(generator.classes)
    num_classes = len(class_counts)
    class_weights = {}
    for i, count in enumerate(class_counts):
        class_weights[i] = total_samples / (num_classes * max(1, count))
    return class_weights

def train_model():
    """
    Two-stage fine-tuning pipeline for MobileNetV2:
    1. Feature extraction with frozen base model.
    2. Fine-tuning top layers of MobileNetV2 with low learning rate.
    """
    ensure_dataset_exists()
    print(f"Loading dataset from: {DATASET_DIR}")
    
    # Data Augmentation Generator matching MobileNetV2 preprocessing
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )

    try:
        train_generator = train_datagen.flow_from_directory(
            DATASET_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        val_generator = train_datagen.flow_from_directory(
            DATASET_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        print("Dataset Class Index Mapping:", train_generator.class_indices)

        if train_generator.samples == 0:
            print("No training images found. Aborting training.")
            return

        class_weights = compute_class_weights(train_generator)
        print("Computed Class Weights:", class_weights)

        # Stage 1: Train Top Head
        print("\n--- STAGE 1: Training Top Classification Head ---")
        model = build_tree_health_model(num_classes=len(CLASSES))
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
            ModelCheckpoint(SAVED_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
        ]

        history_stage1 = model.fit(
            train_generator,
            epochs=INITIAL_EPOCHS,
            validation_data=val_generator,
            class_weight=class_weights,
            callbacks=callbacks
        )

        # Stage 2: Fine-Tuning MobileNetV2 Base
        print("\n--- STAGE 2: Fine-Tuning MobileNetV2 Base Layers ---")
        # Unfreeze base model top layers
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) or 'mobilenetv2' in layer.name.lower():
                base_model = layer
                break

        if base_model is not None:
            base_model.trainable = True
            # Freeze all layers except the last 30
            fine_tune_at = len(base_model.layers) - 30
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
        else:
            # If model is flat functional
            model.trainable = True
            fine_tune_at = len(model.layers) - 20
            for layer in model.layers[:fine_tune_at]:
                layer.trainable = False

        # Recompile model with lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks_stage2 = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
            ModelCheckpoint(SAVED_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
        ]

        history_stage2 = model.fit(
            train_generator,
            epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
            initial_epoch=len(history_stage1.epoch),
            validation_data=val_generator,
            class_weight=class_weights,
            callbacks=callbacks_stage2
        )

        model.save(SAVED_MODEL_PATH)
        print(f"\n[OK] Fine-tuned MobileNetV2 model successfully saved to {SAVED_MODEL_PATH}")

    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    train_model()

