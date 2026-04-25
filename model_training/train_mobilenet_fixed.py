"""
train_mobilenet_fixed.py
------------------------
CPU-friendly MobileNetV2 trainer for the Indian Bovine Breed dataset.

Key features (v2 – 48 breed edition):
  • Auto-detects ALL breed folders in train/ (no hard-coded class list)
  • Oversamples minority breeds to OVERSAMPLE_FLOOR images each
  • sqrt class-weighting to penalise rare-breed errors more
  • Stronger augmentation (brightness, contrast, rotation, zoom)
  • Two-phase fine-tuning with cosine LR decay
  • Saves classes.txt, best_model.keras, optimized_model.tflite
  • Preprocessing: rescale=1./255  →  [0,1] input range
    ↳  tflite_inference.py MUST keep its img_array /= 255 lines

Run from project root:
    python model_training/train_mobilenet_fixed.py
"""

import os
import sys

# Windows GPU Fix: Load CUDA 11 runtime DLLs from pip packages
try:
    import site
    import glob
    for p in site.getsitepackages():
        nvidia_bins = glob.glob(os.path.join(p, 'nvidia', '*', 'bin'))
        for b in nvidia_bins:
            os.add_dll_directory(b)
except Exception:
    pass

import math
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'Indian Bovine Breed Recognition.v1i.folder')
TRAIN_DIR    = os.path.join(DATA_ROOT, 'train')
VALID_DIR    = os.path.join(DATA_ROOT, 'valid')
TEST_DIR     = os.path.join(DATA_ROOT, 'test')
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# HYPER-PARAMETERS
# ---------------------------------------------------------------------------
IMG_SIZE         = 224      # MobileNetV2 native
BATCH_SIZE       = 32
PHASE1_EPOCHS    = 25       # head-only (backbone frozen)
PHASE2_EPOCHS    = 25       # fine-tune top layers
PHASE1_LR        = 1e-3
PHASE2_LR        = 2e-5     # careful fine-tune LR
PATIENCE         = 8
TARGET_SAMPLES   = 500      # ALL breeds oversampled to exactly this count
                             # → balanced 48 × 500 = 24,000 train images
                             # → eliminates class imbalance, no class weights needed
FINETUNE_AT_TOP  = 60       # unfreeze top N backbone layers
SEED             = 42
VALID_EXTS       = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif'}

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------------------------------------------------------
# 1.  COLLECT PATHS & LABELS  (with oversampling)
# ---------------------------------------------------------------------------

def collect_paths_labels(directory, class_to_idx, target_samples=0):
    """
    Collect image paths and labels.
    If target_samples > 0: oversample ALL classes to exactly target_samples images.
    This creates a perfectly balanced dataset (same image count every class).
    Classes with MORE than target_samples images are randomly downsampled.
    Classes with FEWER images are repeat-oversampled (with shuffling).
    """
    paths, labels = [], []
    for cls_name, idx in class_to_idx.items():
        cls_dir = os.path.join(directory, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        files = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        ]
        if not files:
            continue

        if target_samples > 0:
            orig = len(files)
            rng  = random.Random(SEED + idx)
            if orig < target_samples:
                # Oversample: repeat until we have enough, then shuffle
                rep   = math.ceil(target_samples / orig)
                files = (files * rep)[:target_samples]
                rng.shuffle(files)
                print(f"  [OVERSAMPLE] {cls_name}: {orig} -> {len(files)}")
            elif orig > target_samples:
                # Downsample: random subset (keeps diversity for over-represented breeds)
                rng.shuffle(files)
                files = files[:target_samples]
                print(f"  [DOWNSAMPLE] {cls_name}: {orig} -> {len(files)}")
            # else: already exactly target_samples, no change

        paths.extend(files)
        labels.extend([idx] * len(files))
    return paths, labels

# ---------------------------------------------------------------------------
# 2.  TF.DATA PIPELINE
# ---------------------------------------------------------------------------

def parse_image(path, label, num_classes, augment=False):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0      # → [0, 1]

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.35)
        img = tf.image.random_contrast(img, 0.65, 1.35)
        img = tf.image.random_saturation(img, 0.65, 1.35)
        img = tf.image.random_hue(img, 0.08)
        # Random 90° rotation
        k   = tf.random.uniform([], 0, 4, dtype=tf.int32)
        img = tf.image.rot90(img, k)
        # Random zoom (crop + resize)
        scale = tf.random.uniform([], 0.75, 1.0)
        h = tf.cast(IMG_SIZE * scale, tf.int32)
        img = tf.image.resize_with_crop_or_pad(
                  tf.image.resize(img, [h, h]), IMG_SIZE, IMG_SIZE)
        img = tf.clip_by_value(img, 0.0, 1.0)

    label_oh = tf.one_hot(label, num_classes)
    return img, label_oh


def build_dataset(paths, labels, num_classes, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, l: parse_image(p, l, num_classes, augment=training),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------------------------------------------------------------------
# 3.  MODEL
# ---------------------------------------------------------------------------

def build_model(num_classes, trainable_base=False):
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        # NOTE: MobileNetV2 does NOT have include_preprocessing.
        # Input must be in [-1, 1].  We handle that via the rescaling
        # + MobileNetV2's own preprocess_input being applied implicitly
        # by the base model weights (trained on [-1,1] range).
        # Actually: MobileNetV2 from keras expects [-1,1].
        # We rescale to [0,1] in the pipeline and add a Rescaling layer below.
    )
    base.trainable = trainable_base

    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # MobileNetV2 was pretrained with pixels in [-1, 1].
    # Our pipeline gives [0, 1], so we rescale here to [-1, 1].
    x   = layers.Rescaling(scale=2.0, offset=-1.0)(inp)
    x   = base(x, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(512, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return keras.Model(inp, out, name='cattle_mobilenet'), base

# ---------------------------------------------------------------------------
# 4.  CONVERT TO TFLITE
# ---------------------------------------------------------------------------

def convert_to_tflite(model):
    out_path = os.path.join(MODELS_DIR, 'optimized_model.tflite')
    print("\n[INFO] Converting to TFLite (float16) ...")
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_bytes = conv.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_bytes)
        print(f"[OK]  TFLite saved  -> {out_path}  "
              f"[{os.path.getsize(out_path)/1e6:.1f} MB]")
    except Exception as e:
        print(f"[ERROR] TFLite conversion failed: {e}")
    print()
    print("  [NOTE] INFERENCE: This model expects [0, 1] float32 input.")
    print("      tflite_inference.py's  img_array /= 255.0  MUST stay in place.")

# ---------------------------------------------------------------------------
# 5.  MAIN TRAINING LOOP
# ---------------------------------------------------------------------------

def train():
    print("\n" + "=" * 60)
    print("  MOBILENET BREED TRAINER  (48-breed edition)")
    print("=" * 60)

    # --- Detect classes ---
    class_names = sorted([
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ])
    num_classes  = len(class_names)
    class_to_idx = {n: i for i, n in enumerate(class_names)}
    print(f"[INFO] {num_classes} breed classes detected")

    # Save classes.txt
    classes_path = os.path.join(MODELS_DIR, 'classes.txt')
    with open(classes_path, 'w') as f:
        f.write('\n'.join(class_names) + '\n')
    print(f"[SAVED] classes.txt -> {classes_path}")
    print()

    # --- Collect paths (BALANCED: all 48 breeds equalised to TARGET_SAMPLES) ---
    print(f"\n[BALANCE] Equalising all breeds to {TARGET_SAMPLES} images each...")
    train_paths, train_labels = collect_paths_labels(
        TRAIN_DIR, class_to_idx, target_samples=TARGET_SAMPLES)
    valid_paths, valid_labels = collect_paths_labels(VALID_DIR, class_to_idx)
    test_paths,  test_labels  = collect_paths_labels(TEST_DIR,  class_to_idx)
    expected_train = num_classes * TARGET_SAMPLES
    print(f"\n[DATA] train={len(train_paths)} (expected {expected_train})  "
          f"valid={len(valid_paths)}  test={len(test_paths)}")
    print(f"[DATA] Dataset is now PERFECTLY BALANCED: "
          f"{len(train_paths)//num_classes} images per breed")

    # --- No class weights needed: data is already balanced ---
    # (Previously used sqrt class weights, but with equal samples per class
    #  the loss function is already fair. Using weights on balanced data
    #  can actually hurt by distorting the gradient signal.)

    # --- Datasets ---
    train_ds = build_dataset(train_paths, train_labels, num_classes, training=True)
    valid_ds = build_dataset(valid_paths, valid_labels, num_classes)
    test_ds  = build_dataset(test_paths,  test_labels,  num_classes)

    # --- Build model ---
    model, base = build_model(num_classes, trainable_base=False)
    model.compile(
        optimizer=keras.optimizers.Adam(PHASE1_LR),
        # label_smoothing REMOVED: it was suppressing confidence scores
        # without meaningfully improving accuracy on fine-grained breeds.
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
        metrics=['accuracy',
                 keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    model.summary(line_length=90)

    # --- Callbacks ---
    best_path  = os.path.join(MODELS_DIR, 'best_model.keras')
    log1_path  = os.path.join(MODELS_DIR, 'log_phase1.csv')
    log2_path  = os.path.join(MODELS_DIR, 'log_phase2.csv')

    cb_phase1 = [
        keras.callbacks.ModelCheckpoint(best_path, monitor='val_accuracy',
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=3, min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger(log1_path),
    ]

    # =======================================================================
    # PHASE 1: Head only
    # =======================================================================
    print("\n" + "-" * 55)
    print("  PHASE 1 - Head only (backbone frozen)")
    print("-" * 55)
    # No class_weight argument: data is balanced, weights are all 1.0
    model.fit(train_ds, epochs=PHASE1_EPOCHS,
              validation_data=valid_ds,
              callbacks=cb_phase1)

    # =======================================================================
    # PHASE 2: Fine-tune top 40 MobileNetV2 layers
    # =======================================================================
    print("\n" + "-" * 55)
    print("  PHASE 2 – Fine-tune top 40 backbone layers")
    print("-" * 55)

    base.trainable = True
    fine_tune_at   = len(base.layers) - FINETUNE_AT_TOP
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    print(f"  Unfrozen top {FINETUNE_AT_TOP} of {len(base.layers)} backbone layers")

    model.compile(
        optimizer=keras.optimizers.Adam(PHASE2_LR),
        # label_smoothing=0.0 — removed for same reason as Phase 1
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
        metrics=['accuracy',
                 keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )

    cb_phase2 = [
        keras.callbacks.ModelCheckpoint(best_path, monitor='val_accuracy',
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=3, min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(log2_path),
    ]

    model.fit(train_ds, epochs=PHASE2_EPOCHS,
              validation_data=valid_ds,
              # No class_weight: balanced dataset
              callbacks=cb_phase2)

    # =======================================================================
    # FINAL EVALUATION
    # =======================================================================
    print("\n[INFO] Evaluating best model on test set ...")
    best_model = keras.models.load_model(best_path)

    test_loss, test_acc, test_top3 = best_model.evaluate(test_ds, verbose=1)
    print(f"\nTest  accuracy : {test_acc*100:.2f}%")
    print(f"Test  top-3 acc: {test_top3*100:.2f}%")

    # Per-class report
    all_preds, all_true = [], []
    for imgs, lbls in test_ds:
        preds = best_model.predict(imgs, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_true.extend(np.argmax(lbls.numpy(), axis=1))

    report = classification_report(all_true, all_preds,
                                   target_names=class_names, zero_division=0)
    print(report)
    rpt_path = os.path.join(MODELS_DIR, 'classification_report.txt')
    with open(rpt_path, 'w') as f:
        f.write(report)
    print(f"[SAVED] {rpt_path}")

    # =======================================================================
    # TFLITE CONVERSION
    # =======================================================================
    convert_to_tflite(best_model)

    print("\n✅  Training complete!")
    print(f"    Model   : {best_path}")
    print(f"    Classes : {classes_path}  ({num_classes} breeds)")
    print(f"    TFLite  : {os.path.join(MODELS_DIR, 'optimized_model.tflite')}")


if __name__ == '__main__':
    if not os.path.isdir(TRAIN_DIR):
        print(f"ERROR: train directory not found at {TRAIN_DIR}")
        sys.exit(1)
    train()
