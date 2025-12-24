#!/usr/bin/env python3
"""
GPU Training Script for Skin Disease Classification
- Model: EfficientNetV2-S
- Dataset: ISIC/HAM10000 (8 classes, ~25k images)
- Target Accuracy: 87-92%
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
)
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
import os
import math
from datetime import datetime

# Configuration
CONFIG = {
    "img_size": (384, 384),
    "batch_size": 16,
    "epochs": 300,
    "initial_lr": 3e-4,
    "min_lr": 1e-7,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "dropout_rate": 0.4,
    "fine_tune_percent": 0.6,
}

CLASS_NAMES = [
    "actinic_keratosis",
    "basal_cell_carcinoma",
    "benign_keratosis",
    "dermatofibroma",
    "melanocytic_nevus",
    "melanoma",
    "squamous_cell_carcinoma",
    "vascular_lesion",
]

# GPU Setup with Mixed Precision
def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("GPU not found. Enable GPU in Colab.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    mixed_precision.set_global_policy("mixed_float16")
    print("GPU detected & mixed precision enabled")

# Image Rotation using Pure TensorFlow
def rotate_image(image, angle):
    angle_rad = angle * math.pi
    img_shape = tf.shape(image)
    h = tf.cast(img_shape[0], tf.float32)
    w = tf.cast(img_shape[1], tf.float32)
    cx, cy = w / 2.0, h / 2.0
    
    cos_a = tf.cos(angle_rad)
    sin_a = tf.sin(angle_rad)
    tx = cx - cx * cos_a + cy * sin_a
    ty = cy - cx * sin_a - cy * cos_a
    
    transform = [cos_a, -sin_a, tx, sin_a, cos_a, ty, 0.0, 0.0]
    
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=img_shape[:2],
        interpolation="BILINEAR",
        fill_mode="REFLECT"
    )
    return tf.squeeze(image, 0)

# Data Augmentation
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    angle = tf.random.uniform([], -0.10, 0.10)
    image = rotate_image(image, angle)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label

# Dataset Pipeline
def create_dataset(directory, training=True):
    AUTOTUNE = tf.data.AUTOTUNE
    
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=CONFIG["img_size"],
        batch_size=None,
        shuffle=False,
    )
    
    # Preprocess for EfficientNetV2
    ds = ds.map(
        lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
        num_parallel_calls=AUTOTUNE,
    )
    
    if training:
        ds = ds.cache()
        ds = ds.shuffle(5000)
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(CONFIG["batch_size"])
        ds = ds.prefetch(AUTOTUNE)
    else:
        ds = ds.cache()
        ds = ds.batch(CONFIG["batch_size"])
        ds = ds.prefetch(AUTOTUNE)
    
    return ds

# Compute Class-Balanced Alpha for Focal Loss
def compute_alpha(train_dir):
    counts = []
    for cls in CLASS_NAMES:
        path = os.path.join(train_dir, cls)
        counts.append(len(os.listdir(path)))
    
    counts = np.array(counts, dtype=np.float32)
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    weights = np.minimum(weights, 5.0)
    weights = weights / weights.sum()
    
    print("\nClass-balanced alpha weights:")
    for c, w in zip(CLASS_NAMES, weights):
        print(f"   {c}: {w:.4f}")
    
    return tf.constant(weights, dtype=tf.float32)

# Model Architecture
def create_model(num_classes):
    inputs = keras.Input(shape=(*CONFIG["img_size"], 3))
    
    # EfficientNetV2-S Backbone (pretrained on ImageNet)
    base_model = keras.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(*CONFIG["img_size"], 3),
    )
    base_model.trainable = False
    
    # Classification Head
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG["dropout_rate"])(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG["dropout_rate"])(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation("softmax", dtype="float32")(x)
    
    return Model(inputs, outputs), base_model

# Cosine Learning Rate Schedule with Warmup
def cosine_lr(epoch):
    if epoch < CONFIG["warmup_epochs"]:
        return CONFIG["initial_lr"] * (epoch + 1) / CONFIG["warmup_epochs"]
    progress = (epoch - CONFIG["warmup_epochs"]) / (CONFIG["epochs"] - CONFIG["warmup_epochs"])
    return CONFIG["min_lr"] + 0.5 * (CONFIG["initial_lr"] - CONFIG["min_lr"]) * (1 + math.cos(math.pi * progress))

# Test-Time Augmentation
def predict_with_tta(model, images):
    p1 = model.predict(images, verbose=0)
    p2 = model.predict(tf.image.flip_left_right(images), verbose=0)
    return (p1 + p2) / 2

# Main Training Function
def train():
    setup_gpu()
    
    data_dir = "skin_data_v2"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    
    # Compute class weights for imbalanced data
    alpha = compute_alpha(train_dir)
    
    # Create datasets
    train_ds = create_dataset(train_dir, training=True)
    val_ds = create_dataset(val_dir, training=False)
    
    # Build model
    model, base_model = create_model(len(CLASS_NAMES))
    
    # Compile with Focal Loss and AdamW optimizer
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=CONFIG["initial_lr"],
            weight_decay=CONFIG["weight_decay"],
        ),
        loss=keras.losses.CategoricalFocalCrossentropy(
            gamma=2.0,
            alpha=alpha,
        ),
        metrics=["accuracy"],
    )
    
    os.makedirs("models", exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint("models/skin_t4_best.keras", monitor="val_loss", save_best_only=True, mode="min", verbose=1),
        EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True),
        LearningRateScheduler(cosine_lr),
        TensorBoard(log_dir=f"logs/skin_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    ]
    
    # Phase 1: Train classification head (backbone frozen)
    print("\nPhase 1: Training classification head")
    model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks)
    
    # Phase 2: Fine-tune backbone (60% unfrozen)
    print("\nPhase 2: Fine-tuning backbone")
    base_model.trainable = True
    freeze_to = int(len(base_model.layers) * (1 - CONFIG["fine_tune_percent"]))
    for layer in base_model.layers[:freeze_to]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=CONFIG["initial_lr"] / 10,
            weight_decay=CONFIG["weight_decay"],
        ),
        loss=keras.losses.CategoricalFocalCrossentropy(gamma=2.0, alpha=alpha),
        metrics=["accuracy"],
    )
    
    model.fit(train_ds, epochs=CONFIG["epochs"] - 20, validation_data=val_ds, callbacks=callbacks)
    
    # Save final model
    model.save("models/skin_t4_final.keras")
    print("\nTraining complete!")
    print("Best model: models/skin_t4_best.keras")
    print("Final model: models/skin_t4_final.keras")

if __name__ == "__main__":
    train()
