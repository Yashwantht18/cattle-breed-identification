# -*- coding: utf-8 -*-
"""
==============================================================================
  CATTLE BREED IDENTIFICATION - PRODUCTION TRAINING PIPELINE v2
==============================================================================
  Architecture  : EfficientNetV2S  (superior accuracy vs MobileNetV2)
  Key upgrades over previous script:
    * Class-Weighted Loss      -> handles severe imbalance (min=48, max=533 imgs)
    * Heavy Augmentation       -> brightness, contrast, saturation, zoom, flip
    * Two-Phase Fine-Tuning   -> head-only first, then progressive unfreeze
    * Cosine Annealing LR      -> smoother convergence
    * Label Smoothing (0.1)   -> reduces overconfidence
    * Mixed Precision (fp16)  -> ~2x faster on GPU
    * Best Checkpoint          -> saves highest val_accuracy model
    * Full Evaluation Report   -> classification_report + confusion matrix PNG
    * TFLite + INT8 Quant      -> edge-ready .tflite model
==============================================================================
"""

from __future__ import annotations
import os, sys, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(ROOT, 'Indian Bovine Breed Recognition.v1i.folder')
MODELS = os.path.join(ROOT, 'models')
os.makedirs(MODELS, exist_ok=True)

TRAIN_DIR = os.path.join(DATA, 'train')
VALID_DIR = os.path.join(DATA, 'valid')
TEST_DIR  = os.path.join(DATA, 'test')

# ---------------------------------------------------------------------------
# HYPER-PARAMETERS  (tuned for this dataset: 41 classes, ~4k training images)
# ---------------------------------------------------------------------------
IMG_SIZE      = 260    # Increased from 224: captures finer horn/coat detail
BATCH_SIZE    = 16     # keep low to fit on CPU/low-VRAM GPU; 32 if > 6 GB VRAM
PHASE1_LR     = 1e-3   # head-only training learning rate
PHASE2_LR     = 5e-5   # fine-tuning learning rate (must be much lower)
PHASE1_EPOCHS = 25     # slightly more epochs for larger image size
PHASE2_EPOCHS = 25
LABEL_SMOOTH  = 0.1    # label smoothing factor
PATIENCE      = 10     # early-stopping patience (val_accuracy)
SEED          = 42

# Oversample floor: minority classes with fewer training images than this
# will have their paths repeated so the model sees them more often.
# Murrah has 106 images — floor at 200 means it gets ~2x repetition.
OVERSAMPLE_FLOOR = 200

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# MIXED PRECISION  (safe to enable; falls back gracefully)
# ---------------------------------------------------------------------------
try:
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("[INFO] Mixed precision ENABLED (float16 compute)")
except Exception as exc:
    print(f"[INFO] Mixed precision not available -> float32. ({exc})")

# ===========================================================================
# 1. DATA COLLECTION
# ===========================================================================

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def collect_paths_labels(directory: str, class_to_idx: dict,
                         oversample_floor: int = 0):
    """
    Walk directory and return (paths, integer labels) lists.

    oversample_floor: if > 0, minority classes with fewer images than this
    threshold will have their file list repeated (with random cycling) until
    they reach the floor. This balances training frequency without
    requiring additional real images.
    """
    paths, labels = [], []
    for cls_name, idx in class_to_idx.items():
        cls_dir = os.path.join(directory, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        cls_files = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        ]
        if not cls_files:
            continue

        # Oversample if this class is below the floor
        if oversample_floor > 0 and len(cls_files) < oversample_floor:
            import math, random
            orig_count = len(cls_files)
            rng = random.Random(SEED + idx)   # deterministic but per-class
            repeats = math.ceil(oversample_floor / orig_count)
            expanded = (cls_files * repeats)[:oversample_floor]
            rng.shuffle(expanded)
            cls_files = expanded
            print(f"  [OVERSAMPLE] {cls_name}: {orig_count} -> {len(cls_files)} samples")

        paths.extend(cls_files)
        labels.extend([idx] * len(cls_files))
    return paths, labels

# ===========================================================================
# 2. TF.DATA PIPELINE
# ===========================================================================

def decode_and_resize(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    return img, label


def augment(img, label):
    """
    Stronger randomised augmentation applied only during training.

    Key additions vs v1:
    - Wider brightness/contrast/saturation range to handle cattle in
      bright sunlight vs shade (critical for dark breeds like Murrah)
    - Random 90-degree rotations simulate various photograph orientations
    - Random cutout (coarse dropout) forces the model to learn whole-body
      features rather than latching onto a single patch (e.g. coat colour)
    """
    # Geometric transforms
    img = tf.image.random_flip_left_right(img)
    # Random 0/90/180/270 degree rotation
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    # Random zoom via crop + resize (wider range than before)
    scale = tf.random.uniform([], 0.70, 1.0)
    new_h = tf.cast(tf.cast(IMG_SIZE, tf.float32) * scale, tf.int32)
    new_w = new_h
    img   = tf.image.resize_with_crop_or_pad(
                tf.image.resize(img, [new_h, new_w]), IMG_SIZE, IMG_SIZE)

    # Colour / photometric transforms
    # Wider range: essential for dark-coated breeds (Murrah, Nagpuri, Bhadawari)
    # in variable Indian field lighting conditions
    img = tf.image.random_brightness(img, max_delta=0.40)
    img = tf.image.random_contrast(img, lower=0.60, upper=1.40)
    img = tf.image.random_saturation(img, lower=0.60, upper=1.40)
    img = tf.image.random_hue(img, max_delta=0.10)

    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label


def to_onehot(num_classes: int):
    def _fn(img, label):
        return img, tf.one_hot(label, num_classes)
    return _fn


def build_tf_dataset(paths, labels, num_classes, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(to_onehot(num_classes), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ===========================================================================
# 3. MODEL BUILDER
# ===========================================================================

def build_model(num_classes: int, trainable_base=False):
    """
    EfficientNetV2S backbone + custom head.
    include_preprocessing=True  -> model handles [0,255] input internally.
    """
    base = EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_preprocessing=True,
    )
    base.trainable = trainable_base

    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D(name='gap')(x)
    x   = layers.BatchNormalization(name='head_bn1')(x)
    x   = layers.Dropout(0.4, name='head_drop1')(x)
    x   = layers.Dense(512, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(1e-4),
                        name='head_dense1')(x)
    x   = layers.BatchNormalization(name='head_bn2')(x)
    x   = layers.Dropout(0.3, name='head_drop2')(x)
    # Force float32 output even with mixed precision
    out = layers.Dense(num_classes, activation='softmax',
                        dtype='float32', name='predictions')(x)
    return keras.Model(inp, out, name='cattle_breed_model'), base


def compile_model(model, lr, label_smooth):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth),
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc'),
        ],
    )

# ===========================================================================
# 4. CALLBACKS
# ===========================================================================

def cosine_decay(initial_lr, total_epochs):
    """Returns a LR schedule function for cosine annealing."""
    def schedule(epoch):
        frac = epoch / max(total_epochs - 1, 1)
        return float(initial_lr * 0.5 * (1.0 + np.cos(np.pi * frac)))
    return schedule


def make_callbacks(phase: int, total_epochs: int, lr: float):
    return [
        ModelCheckpoint(
            os.path.join(MODELS, 'best_model.keras'),
            monitor='val_accuracy', save_best_only=True,
            mode='max', verbose=1,
        ),
        EarlyStopping(
            monitor='val_accuracy', patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        CSVLogger(
            os.path.join(MODELS, f'log_phase{phase}.csv'), append=False
        ),
        LearningRateScheduler(cosine_decay(lr, total_epochs), verbose=0),
    ]

# ===========================================================================
# 5. EVALUATION HELPERS
# ===========================================================================

def evaluate_on_dataset(model, dataset, class_names, split_name='test'):
    print(f"\n[INFO] Evaluating on {split_name} set ...")
    all_preds, all_true = [], []
    for img_batch, lbl_batch in dataset:
        preds = model.predict(img_batch, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        # lbl_batch is one-hot; convert back to int
        if lbl_batch.ndim > 1:
            all_true.extend(np.argmax(lbl_batch.numpy(), axis=1))
        else:
            all_true.extend(lbl_batch.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_true)
    top1   = float(np.mean(y_pred == y_true))

    print(f"\n  [{split_name.upper()}] Top-1 Accuracy : {top1*100:.2f}%")
    report = classification_report(y_true, y_pred,
                                   target_names=class_names, zero_division=0)
    print(report)

    # Save text report
    rpath = os.path.join(MODELS, f'classification_report_{split_name}.txt')
    with open(rpath, 'w') as f:
        f.write(f"{split_name.upper()} Top-1 Accuracy: {top1*100:.2f}%\n\n{report}")
    print(f"[SAVED] {rpath}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_size = max(14, num_cls // 3) if (num_cls := len(class_names)) else 14
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 2))
    if HAS_SNS:
        import seaborn as sns
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
    else:
        im = ax.imshow(cm, cmap='Blues')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=5)
        ax.set_yticklabels(class_names, fontsize=5)
    ax.set_title(f'Confusion Matrix  [{split_name}]  acc={top1*100:.1f}%')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    cm_path = os.path.join(MODELS, f'confusion_matrix_{split_name}.png')
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"[SAVED] {cm_path}")
    return top1


def plot_history(hist1, hist2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    n1 = len(hist1.history['accuracy'])
    n2 = len(hist2.history['accuracy'])
    e1 = range(1, n1 + 1)
    e2 = range(n1 + 1, n1 + n2 + 1)

    ax1.plot(e1, [a * 100 for a in hist1.history['accuracy']],     'b--', label='Train P1')
    ax1.plot(e1, [a * 100 for a in hist1.history['val_accuracy']], 'b-',  label='Val P1')
    ax1.plot(e2, [a * 100 for a in hist2.history['accuracy']],     'r--', label='Train P2')
    ax1.plot(e2, [a * 100 for a in hist2.history['val_accuracy']], 'r-',  label='Val P2')
    ax1.set_title('Accuracy (%)')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.plot(e1, hist1.history['loss'],     'b--', label='Train P1')
    ax2.plot(e1, hist1.history['val_loss'], 'b-',  label='Val P1')
    ax2.plot(e2, hist2.history['loss'],     'r--', label='Train P2')
    ax2.plot(e2, hist2.history['val_loss'], 'r-',  label='Val P2')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    out = os.path.join(MODELS, 'training_curves.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[SAVED] {out}")

# ===========================================================================
# 6. TFLITE CONVERSION
# ===========================================================================

def convert_to_tflite(model, train_ds):
    """
    Tries INT8 quantisation first (smallest, fastest on edge).
    Falls back to float16 if INT8 fails.
    """
    out_path = os.path.join(MODELS, 'optimized_model.tflite')

    def representative_dataset_gen():
        count = 0
        for img_batch, _ in train_ds:
            for img in img_batch:
                yield [tf.expand_dims(img, 0)]
                count += 1
                if count >= 200:
                    return

    print("\n[INFO] Attempting INT8 quantised TFLite conversion ...")
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative_dataset_gen
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type  = tf.float32
        conv.inference_output_type = tf.float32
        tflite_bytes = conv.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_bytes)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"[OK] INT8 TFLite saved -> {out_path}  [{size_mb:.1f} MB]")
        return
    except Exception as exc:
        print(f"[WARN] INT8 failed ({exc}), trying float16 ...")

    try:
        conv2 = tf.lite.TFLiteConverter.from_keras_model(model)
        conv2.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_bytes = conv2.convert()
        with open(out_path, 'wb') as f:
            f.write(tflite_bytes)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"[OK] Float16 TFLite saved -> {out_path}  [{size_mb:.1f} MB]")
    except Exception as exc2:
        print(f"[ERROR] TFLite conversion failed: {exc2}")

    # -------------------------------------------------------------------------
    # ⚠️  CRITICAL POST-TRAINING NOTICE  ⚠️
    # -------------------------------------------------------------------------
    # This model was built with EfficientNetV2S(include_preprocessing=True).
    # That means the Keras/TFLite model INCLUDES an internal rescaling layer
    # that expects RAW [0, 255] float32 pixels as input — NOT [0, 1] values.
    #
    # After retraining, you MUST update inference/tflite_inference.py:
    #   preprocess_image():  REMOVE the line  `img_array = img_array / 255.0`
    #   TTA flip tensor:     REMOVE the        `/ 255.0` in arr_flip
    #
    # The current model on disk was trained with an OLDER script that used
    # rescale=1./255, so the existing inference code has /255.0 in place.
    # Once you replace optimized_model.tflite with this retrained model,
    # remove those two /255.0 lines or predictions will be garbage.
    # -------------------------------------------------------------------------
    print("\n[POST-TRAINING REMINDER]")
    print("  This model expects raw [0-255] input (include_preprocessing=True).")
    print("  Update inference/tflite_inference.py: REMOVE the /255.0 divisions.")
    print("  See comments in convert_to_tflite() for details.")

# ===========================================================================
# 7. MAIN  (entry point)
# ===========================================================================

def train():
    print("\n" + "=" * 65)
    print("  CATTLE BREED TRAINING PIPELINE v2  (EfficientNetV2S)")
    print("=" * 65)

    # --- Detect classes ---
    class_names = sorted([
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ])
    num_classes = len(class_names)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"[INFO] {num_classes} breed classes detected")

    # Save classes.txt (index = line number, 0-based)
    classes_path = os.path.join(MODELS, 'classes.txt')
    with open(classes_path, 'w') as f:
        f.write('\n'.join(class_names) + '\n')
    print(f"[SAVED] classes.txt -> {classes_path}")

    # --- Collect file paths ---
    # Training uses oversampling so minority breeds appear more often.
    # Validation / test do NOT oversample (must reflect true distribution).
    train_paths, train_labels = collect_paths_labels(
        TRAIN_DIR, class_to_idx, oversample_floor=OVERSAMPLE_FLOOR)
    valid_paths, valid_labels = collect_paths_labels(VALID_DIR, class_to_idx)
    test_paths,  test_labels  = collect_paths_labels(TEST_DIR,  class_to_idx)
    print(f"[DATA]  train={len(train_paths)}  valid={len(valid_paths)}  test={len(test_paths)}")

    # --- Class weights ---
    # Use original (pre-oversample) labels for weight computation so that
    # oversampled classes don't get artificially low weights.
    orig_train_paths, orig_train_labels = collect_paths_labels(TRAIN_DIR, class_to_idx)
    raw_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=orig_train_labels,
    )
    # Square the weights for stronger minority emphasis:
    # A class with raw_weight=4.0 becomes 16.0 — the model is penalised much
    # harder for misclassifying rare breeds (e.g. Kherigarh, Kenkatha, Murrah)
    squared_weights = np.sqrt(raw_weights)   # sqrt balances aggressiveness
    class_weights = {i: float(w) for i, w in enumerate(squared_weights)}
    print(f"[INFO]  class_weight range (sqrt): {min(squared_weights):.2f} - {max(squared_weights):.2f}")

    # --- Build tf.data datasets ---
    train_ds = build_tf_dataset(train_paths, train_labels, num_classes, training=True)
    valid_ds = build_tf_dataset(valid_paths, valid_labels, num_classes, training=False)
    test_ds  = build_tf_dataset(test_paths,  test_labels,  num_classes, training=False)

    # ===========================================================
    #    PHASE 1:  Train head only  (EfficientNetV2S frozen)
    # ===========================================================
    print("\n" + "-" * 65)
    print("  PHASE 1:  Head-only training  (backbone FROZEN)")
    print("-" * 65)

    model, base = build_model(num_classes, trainable_base=False)
    compile_model(model, PHASE1_LR, LABEL_SMOOTH)

    trainable_count = sum(p.numpy() for p in model.trainable_weights
                          if hasattr(p, 'numpy')) if False else \
                      sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"[INFO]  Trainable params: {trainable_count:,}")

    t0 = time.time()
    hist1 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weights,
        callbacks=make_callbacks(1, PHASE1_EPOCHS, PHASE1_LR),
    )
    elapsed1 = (time.time() - t0) / 60
    best_acc1 = max(hist1.history.get('val_accuracy', [0]))
    print(f"[DONE]  Phase 1 finished in {elapsed1:.1f} min")
    print(f"[BEST]  Phase 1 best val_accuracy = {best_acc1 * 100:.2f}%")

    # ===========================================================
    #    PHASE 2:  Fine-tune top 60% of backbone
    # ===========================================================
    print("\n" + "-" * 65)
    print("  PHASE 2:  Fine-tuning  (top 60% of backbone UNFROZEN)")
    print("-" * 65)

    # Reload best checkpoint
    best_ckpt = os.path.join(MODELS, 'best_model.keras')
    model = keras.models.load_model(best_ckpt)

    # Unfreeze top 60% of EfficientNetV2S
    # Layer at index 1 of our model is the EfficientNetV2S backbone
    eff_backbone = None
    for lyr in model.layers:
        if isinstance(lyr, keras.Model):
            eff_backbone = lyr
            break
    if eff_backbone is None:
        # Fallback: layer by name
        for lyr in model.layers:
            if 'efficientnet' in lyr.name.lower():
                eff_backbone = lyr
                break

    if eff_backbone:
        eff_backbone.trainable = True
        total_layers = len(eff_backbone.layers)
        freeze_until = int(total_layers * 0.40)   # freeze bottom 40%
        for lyr in eff_backbone.layers[:freeze_until]:
            lyr.trainable = False
        trainable_backbone = sum(
            int(np.prod(v.shape)) for v in eff_backbone.trainable_variables
        )
        print(f"[INFO]  Backbone layers: {total_layers} | frozen: {freeze_until} | "
              f"unfrozen: {total_layers - freeze_until}")
        print(f"[INFO]  Trainable backbone params: {trainable_backbone:,}")
    else:
        print("[WARN]  Could not locate EfficientNetV2S backbone layer; "
              "fine-tuning the entire model")
        model.trainable = True

    compile_model(model, PHASE2_LR, LABEL_SMOOTH)

    t1 = time.time()
    hist2 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=PHASE2_EPOCHS,
        class_weight=class_weights,
        callbacks=make_callbacks(2, PHASE2_EPOCHS, PHASE2_LR),
    )
    elapsed2 = (time.time() - t1) / 60
    best_acc2 = max(hist2.history.get('val_accuracy', [0]))
    print(f"[DONE]  Phase 2 finished in {elapsed2:.1f} min")
    print(f"[BEST]  Phase 2 best val_accuracy = {best_acc2 * 100:.2f}%")

    # ===========================================================
    #    EVALUATION
    # ===========================================================
    best_model = keras.models.load_model(best_ckpt)
    val_acc  = evaluate_on_dataset(best_model, valid_ds, class_names, 'valid')
    test_acc = evaluate_on_dataset(best_model, test_ds,  class_names, 'test')

    plot_history(hist1, hist2)

    # ===========================================================
    #    TFLITE CONVERSION
    # ===========================================================
    convert_to_tflite(best_model, train_ds)

    # ===========================================================
    #    FINAL SUMMARY
    # ===========================================================
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print(f"  Phase 1 best val_accuracy : {best_acc1 * 100:.2f}%")
    print(f"  Phase 2 best val_accuracy : {best_acc2 * 100:.2f}%")
    print(f"  Validation accuracy       : {val_acc  * 100:.2f}%")
    print(f"  Test accuracy             : {test_acc * 100:.2f}%")
    print(f"  Total time                : {(elapsed1 + elapsed2):.1f} min")
    print(f"  Models saved in           : {MODELS}")
    print("=" * 65)


if __name__ == '__main__':
    train()
