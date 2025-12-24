#!/usr/bin/env python3
"""
CPU Training Script for Skin Disease Classification
- Model: EfficientNetV2-B0
- Dataset: ISIC/HAM10000 (8 classes, ~25k images)
- Target Accuracy: 80-85%
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
)
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
import os
import math
from datetime import datetime

print("=" * 60)
print("CPU TRAINING MODE")
print("=" * 60)
print("This will take several hours. Run overnight for best results.\n")

# Configuration
CONFIG = {
    "img_size": (224, 224),
    "batch_size": 8,
    "epochs": 100,
    "initial_lr": 1e-3,
    "min_lr": 1e-6,
    "warmup_epochs": 3,
    "dropout_rate": 0.4,
    "fine_tune_percent": 0.5,
    "patience": 15,
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
        fill_mode="REFLECT",
        fill_value=0.0
    )
    return tf.squeeze(image, 0)

# Data Augmentation
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    angle = tf.random.uniform([], -0.08, 0.08)
    image = rotate_image(image, angle)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.clip_by_value(image, -1.0, 1.0)
    return image, label

# Dataset Pipeline
def create_dataset(directory, training=True):
    AUTOTUNE = tf.data.AUTOTUNE
    existing_classes = [c for c in CLASS_NAMES if os.path.exists(os.path.join(directory, c))]
    
    if len(existing_classes) != len(CLASS_NAMES):
        print(f"Found {len(existing_classes)} classes (expected {len(CLASS_NAMES)})")
    
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=existing_classes if len(existing_classes) == len(CLASS_NAMES) else None,
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
        ds = ds.shuffle(2000)
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(CONFIG["batch_size"])
        ds = ds.prefetch(AUTOTUNE)
    else:
        ds = ds.cache()
        ds = ds.batch(CONFIG["batch_size"])
        ds = ds.prefetch(AUTOTUNE)
    
    return ds, existing_classes

# Compute Class-Balanced Alpha for Focal Loss
def compute_alpha(train_dir, class_names):
    counts = []
    for cls in class_names:
        path = os.path.join(train_dir, cls)
        if os.path.exists(path):
            counts.append(len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]))
        else:
            counts.append(1)
    
    counts = np.array(counts, dtype=np.float32)
    total = counts.sum()
    weights = total / (len(class_names) * np.maximum(counts, 1))
    weights = np.minimum(weights, 5.0)
    weights = weights / weights.sum()
    
    print("\nClass distribution & weights:")
    for c, cnt, w in zip(class_names, counts, weights):
        pct = cnt / total * 100
        print(f"   {c}: {int(cnt)} ({pct:.1f}%) -> alpha={w:.4f}")
    
    return tf.constant(weights, dtype=tf.float32)

# Model Architecture
def create_model(num_classes):
    inputs = keras.Input(shape=(*CONFIG["img_size"], 3))
    
    # EfficientNetV2-B0 Backbone (pretrained on ImageNet)
    base_model = keras.applications.EfficientNetV2B0(
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
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CONFIG["dropout_rate"])(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return Model(inputs, outputs), base_model

# Cosine Learning Rate Schedule with Warmup
def cosine_lr_schedule(epoch):
    if epoch < CONFIG["warmup_epochs"]:
        return CONFIG["initial_lr"] * (epoch + 1) / CONFIG["warmup_epochs"]
    progress = (epoch - CONFIG["warmup_epochs"]) / (CONFIG["epochs"] - CONFIG["warmup_epochs"])
    return CONFIG["min_lr"] + 0.5 * (CONFIG["initial_lr"] - CONFIG["min_lr"]) * (1 + math.cos(math.pi * progress))

# Test-Time Augmentation
def predict_with_tta(model, dataset):
    predictions = []
    labels = []
    for images, batch_labels in dataset:
        p1 = model.predict(images, verbose=0)
        p2 = model.predict(tf.image.flip_left_right(images), verbose=0)
        predictions.append((p1 + p2) / 2)
        labels.append(batch_labels.numpy())
    return np.vstack(predictions), np.vstack(labels)

# Main Training Function
def train():
    data_dir = "skin_data_v2"
    
    if not os.path.exists(os.path.join(data_dir, "train")):
        print(f"{data_dir}/train not found!")
        return
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    
    print(f"Using data from: {data_dir}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_ds, class_names = create_dataset(train_dir, training=True)
    val_ds, _ = create_dataset(val_dir, training=False)
    
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")
    
    # Compute class weights for imbalanced data
    alpha = compute_alpha(train_dir, class_names)
    
    # Build model
    print("\nBuilding model (EfficientNetV2-B0)...")
    model, base_model = create_model(num_classes)
    
    # Compile with Focal Loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["initial_lr"]),
        loss=keras.losses.CategoricalFocalCrossentropy(gamma=2.0, alpha=alpha),
        metrics=["accuracy"],
    )
    
    model.summary()
    os.makedirs("models", exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint("models/cpu_model_best.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_loss", patience=CONFIG["patience"], restore_best_weights=True, verbose=1),
        LearningRateScheduler(cosine_lr_schedule, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=CONFIG["min_lr"], verbose=1),
    ]
    
    # Phase 1: Train classification head (backbone frozen)
    print("\n" + "=" * 60)
    print("PHASE 1: Training classification head (backbone frozen)")
    print("=" * 60)
    
    warmup_epochs = 10
    model.fit(train_ds, epochs=warmup_epochs, validation_data=val_ds, callbacks=callbacks, verbose=1)
    
    # Phase 2: Fine-tune backbone (50% unfrozen)
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning backbone (50% trainable)")
    print("=" * 60)
    
    base_model.trainable = True
    freeze_to = int(len(base_model.layers) * (1 - CONFIG["fine_tune_percent"]))
    for layer in base_model.layers[:freeze_to]:
        layer.trainable = False
    
    trainable = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"Trainable layers: {trainable}/{len(base_model.layers)}")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["initial_lr"] / 10),
        loss=keras.losses.CategoricalFocalCrossentropy(gamma=2.0, alpha=alpha),
        metrics=["accuracy"],
    )
    
    model.fit(train_ds, epochs=CONFIG["epochs"] - warmup_epochs, validation_data=val_ds, callbacks=callbacks, verbose=1)
    
    # Save final model
    model.save("models/cpu_model_final.keras")
    
    # Evaluation
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation Accuracy: {val_acc:.2%}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # TTA evaluation
    print("\nRunning Test-Time Augmentation...")
    tta_preds, tta_labels = predict_with_tta(model, val_ds)
    tta_acc = np.mean(np.argmax(tta_preds, axis=1) == np.argmax(tta_labels, axis=1))
    print(f"TTA Accuracy: {tta_acc:.2%}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best model: models/cpu_model_best.keras")
    print(f"Final model: models/cpu_model_final.keras")

if __name__ == "__main__":
    train()
