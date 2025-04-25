
#!/usr/bin/env python3
"""
DenseNet‑121 X‑ray fracture classifier · TensorFlow 2.19 / Keras 3

Folder structure REQUIRED
└── data_dir/
    ├── train/
    │   ├── fractured/
    │   └── not_fractured/
    ├── val/
    │   ├── fractured/
    │   └── not_fractured/
    └── test/
        ├── fractured/
        └── not_fractured/

• Only the train split is used for learning; val & test are untouched.
• Automatic loss choice:
      minority < 40 % ➜ BinaryFocalCrossentropy
      otherwise      ➜ BinaryCrossentropy
• Regularisation: augmentation + L2 + dropout + early‑stopping.
• Result saved as **xray_dense121_best.keras** (native format → no “Cast” error).
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L, models as M, optimizers as O
from tensorflow.keras.applications import densenet
import click

# —————————————————————————————————— CONFIG
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
IMBALANCE_THRESHOLD = 0.40           # below this → focal loss
VALID_CLASSES = {"fractured", "not_fractured"}

# —————————————————————————————————— LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s • %(levelname)s • %(message)s",
    handlers=[
        logging.FileHandler("training.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# —————————————————————————————————— HELPERS
def maybe_enable_mp() -> None:
    """Enable mixed‑precision if GPU is available."""
    if tf.config.list_physical_devices("GPU"):
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("Mixed precision enabled")
        except Exception as exc:           # pragma: no cover
            logger.warning("Mixed precision not enabled → %s", exc)


def compute_class_weights(ds: tf.data.Dataset):
    """Return class_weight dict & minority proportion."""
    counts = {0: 0, 1: 0}
    for _x, y in ds.unbatch():
        counts[int(y.numpy())] += 1
    total = counts[0] + counts[1]
    weights = {c: total / (2 * n) for c, n in counts.items() if n}
    minority_prop = min(counts.values()) / total
    logger.info("class counts %s → weights %s (minority %.2f)", counts, weights, minority_prop)
    return weights, minority_prop


def choose_loss(minority_prop: float):
    if minority_prop < IMBALANCE_THRESHOLD:
        logger.info("Using BinaryFocalCrossentropy")
        return tf.keras.losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2.0)
    logger.info("Using BinaryCrossentropy")
    return "binary_crossentropy"

# —————————————————————————————————— DATASET
def make_datasets(data_dir: str, batch_size: int = 32):
    data_path = Path(data_dir)

    def _verify_split(split: str) -> Path:
        split_dir = data_path / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"✖ Missing directory: {split_dir}")
        classes = {p.name for p in split_dir.iterdir() if p.is_dir()}
        if classes != VALID_CLASSES:
            raise ValueError(f"✖ {split_dir} must contain {VALID_CLASSES}; found {classes}")
        return split_dir

    train_dir = _verify_split("train")
    val_dir   = _verify_split("val")
    test_dir  = _verify_split("test")

    def _image_ds(path: Path):
        return tf.keras.utils.image_dataset_from_directory(
            path,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            label_mode="binary",
            class_names=["not_fractured", "fractured"],  # ensure 0→normal, 1→fracture
        ).apply(tf.data.experimental.ignore_errors())

    train_ds = _image_ds(train_dir)
    val_ds   = _image_ds(val_dir)
    test_ds  = _image_ds(test_dir)

    test_steps = test_ds.cardinality().numpy()
    if test_steps < 0:                                    # unknown cardinality
        test_steps = sum(1 for _ in test_ds)

    # augmentation only on train
    aug = tf.keras.Sequential(
        [
            L.RandomFlip("horizontal"),
            L.RandomRotation(0.25),
            L.RandomZoom(0.25),
            L.RandomContrast(0.2),
            L.RandomBrightness(factor=0.2),
        ]
    )
    train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y),
                            num_parallel_calls=AUTOTUNE)

    return (
        train_ds.prefetch(AUTOTUNE),
        val_ds.prefetch(AUTOTUNE),
        test_ds.prefetch(AUTOTUNE),
        test_steps,
    )

# —————————————————————————————————— MODEL
def build_model() -> tf.keras.Model:
    base = densenet.DenseNet121(include_top=False,
                                weights="imagenet",
                                input_shape=(*IMG_SIZE, 3))
    # unfreeze last 10 layers
    for layer in base.layers[:-10]:
        layer.trainable = False

    x = L.GlobalAveragePooling2D()(base.output)
    x = L.BatchNormalization()(x)
    x = L.Dense(512, activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = L.Dropout(0.4)(x)
    x = L.Dense(256, activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = L.Dropout(0.3)(x)
    out = L.Dense(1, activation="sigmoid", dtype="float32")(x)
    return M.Model(base.input, out)

# —————————————————————————————————— TRAIN
def run_train(data_dir: str,
              epochs: int = 23,
              batch_size: int = 32,
              mixed_precision: bool = True):
    if mixed_precision:
        maybe_enable_mp()

    train_ds, val_ds, test_ds, test_steps = make_datasets(data_dir, batch_size)
    class_weight, minority = compute_class_weights(train_ds)
    loss_fn = choose_loss(minority)

    model = build_model()
    model.compile(
        optimizer=O.Adam(1e-4),
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )

    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.2, patience=2, min_lr=5e-6
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f"logs/{datetime.datetime.now():%Y%m%d-%H%M%S}"
        ),
    ]

    logger.info("🚀 Training…")
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=epochs,
              callbacks=cbs,
              class_weight=class_weight,
              verbose=1)

    logger.info("🔍 Evaluating on test set…")
    model.evaluate(test_ds, steps=test_steps, verbose=1)

    model.save("xray_dense121_best.keras")     # native Keras format
    logger.warning("✅ Model saved → xray_dense121_best.keras")

# —————————————————————————————————— CLI
@click.command()
@click.option("--data-dir", default="Bone_Fracture_Binary_Classification",
              help="Dataset root directory")
@click.option("--epochs", default=23, help="Epochs")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--mixed-precision/--no-mixed-precision", default=True,
              help="Enable mixed precision if GPU found")
def cli(data_dir: str, epochs: int, batch_size: int, mixed_precision: bool):
    run_train(data_dir, epochs, batch_size, mixed_precision)


if __name__ == "__main__":
    cli()
