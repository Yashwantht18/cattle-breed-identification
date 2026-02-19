"""
==============================================================================
  GATEKEEPER MODULE - Bovine Image Validator
==============================================================================
  Uses a pre-trained MobileNetV2 (ImageNet) to determine if an uploaded image
  contains a bovine animal (cattle or buffalo) before running breed inference.

  Strategy (Lenient-first to avoid false negatives):
  - ACCEPT if ANY animal-related ImageNet class scores > ANIMAL_THRESHOLD
  - ACCEPT if top ImageNet class is a broad "animal" category
  - REJECT ONLY if image is clearly non-animal AND no animal signal found
  - Threshold is low (5%) to catch Indian breeds with unusual poses/backgrounds
==============================================================================
"""

import numpy as np
from PIL import Image
import os

# ---------------------------------------------------------------------------
# ImageNet class index mappings relevant to bovines and animals
# Source: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
# ---------------------------------------------------------------------------

# Primary bovine classes (highest priority accepts)
BOVINE_CLASSES = {
    340: "ox",
    346: "water_buffalo",
    347: "bison",
    349: "bighorn_sheep",
    351: "hartebeest",
    352: "impala",
    353: "gazelle",
}

# Broad animal classes (secondary accepts — catches mis-classified cattle)
ANIMAL_CLASSES = {
    # Other farm / large animals
    334: "boar",
    335: "sow",
    339: "ram",
    348: "ibex",
    354: "Arabian_camel",
    355: "llama",
    356: "weasel",

    # Wild bovines / deer family
    350: "lesser_panda",
    357: "mink",

    # Generic animal-like detections
    # Dogs (used as proxy for "animal-shaped object in natural setting")
    151: "Chihuahua",
    207: "golden_retriever",
    208: "Labrador_retriever",

    # Horses / donkey
    603: "horse_cart",
    897: "wild_ass",

    # More large animals
    385: "Indian_elephant",
    386: "African_elephant",
    388: "giant_panda",
}

# Classes that strongly indicate NOT an animal image
NON_ANIMAL_CLASSES = {
    # People
    878: "person",
    # Vehicles
    407: "ambulance",
    436: "beach_wagon",
    468: "car_wheel",
    511: "convertible",
    # Buildings / indoor
    663: "mosque",
    # Food
    924: "guacamole",
    959: "pizza",
}

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
BOVINE_THRESHOLD = 0.05    # 5%  — accept if any bovine class hits this
ANIMAL_THRESHOLD = 0.08    # 8%  — accept if any broad animal class hits this
REJECT_THRESHOLD = 0.60    # 60% — reject only if a non-animal class dominates


class ImageNetGatekeeper:
    """
    Lightweight gatekeeper using MobileNetV2 + ImageNet weights.
    No custom training required.
    """

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load MobileNetV2 with ImageNet weights (downloads ~14MB on first use)."""
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

            self.preprocess_fn = preprocess_input
            self.model = MobileNetV2(
                weights='imagenet',
                include_top=True,
                input_shape=(224, 224, 3)
            )
            self.model.trainable = False
            print("[Gatekeeper] MobileNetV2 loaded successfully.")
        except Exception as e:
            print(f"[Gatekeeper] WARNING: Could not load MobileNetV2: {e}")
            print("[Gatekeeper] Gatekeeper will PASS ALL images (fail-open mode).")
            self.model = None

    def _preprocess(self, image_path: str) -> np.ndarray:
        """Load and preprocess image for MobileNetV2."""
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        arr = self.preprocess_fn(arr)
        return arr

    def check(self, image_path: str) -> dict:
        """
        Check if an image is likely a bovine.

        Returns:
            {
              "is_bovine": bool,
              "reason": str,
              "top_class": str,
              "top_confidence": float,
              "bovine_signal": float,   # max prob across bovine classes
              "animal_signal": float,   # max prob across broad animal classes
            }
        """
        # Fail-open: if model didn't load, let everything through
        if self.model is None:
            return {
                "is_bovine": True,
                "reason": "gatekeeper_unavailable",
                "top_class": "unknown",
                "top_confidence": 0.0,
                "bovine_signal": 0.0,
                "animal_signal": 0.0,
            }

        try:
            import tensorflow as tf
            input_arr = self._preprocess(image_path)
            preds = self.model.predict(input_arr, verbose=0)[0]  # shape: (1000,)

            # --- Decode top prediction ---
            top_idx = int(np.argmax(preds))
            top_conf = float(preds[top_idx])

            # Decode class name
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
                np.expand_dims(preds, axis=0), top=5
            )[0]
            top_class_name = decoded[0][1] if decoded else f"class_{top_idx}"

            # --- Signal extraction ---
            bovine_signal = max(
                float(preds[idx]) for idx in BOVINE_CLASSES.keys()
                if idx < len(preds)
            )
            animal_signal = max(
                float(preds[idx]) for idx in ANIMAL_CLASSES.keys()
                if idx < len(preds)
            )
            non_animal_signal = max(
                float(preds[idx]) for idx in NON_ANIMAL_CLASSES.keys()
                if idx < len(preds)
            )

            print(f"[Gatekeeper] top={top_class_name}({top_conf:.2%}) | "
                  f"bovine={bovine_signal:.2%} | "
                  f"animal={animal_signal:.2%} | "
                  f"non-animal={non_animal_signal:.2%}")

            # --- Decision logic (lenient to avoid false negatives) ---

            # 1. Strong bovine signal → always accept
            if bovine_signal >= BOVINE_THRESHOLD:
                return self._result(True, "bovine_detected",
                                    top_class_name, top_conf,
                                    bovine_signal, animal_signal)

            # 2. Decent animal signal → accept (Indian breeds may not match
            #    ImageNet bovine classes perfectly)
            if animal_signal >= ANIMAL_THRESHOLD:
                return self._result(True, "animal_detected",
                                    top_class_name, top_conf,
                                    bovine_signal, animal_signal)

            # 3. Some combined animal presence → accept with low confidence
            if (bovine_signal + animal_signal) >= 0.10:
                return self._result(True, "weak_animal_signal",
                                    top_class_name, top_conf,
                                    bovine_signal, animal_signal)

            # 4. Strongly non-animal → reject
            if non_animal_signal >= REJECT_THRESHOLD:
                return self._result(False, "non_animal_dominant",
                                    top_class_name, top_conf,
                                    bovine_signal, animal_signal)

            # 5. Ambiguous → accept (fail-safe, let breed model decide)
            return self._result(True, "ambiguous_pass",
                                top_class_name, top_conf,
                                bovine_signal, animal_signal)

        except Exception as e:
            print(f"[Gatekeeper] Error during check: {e} — passing image through.")
            return self._result(True, "gatekeeper_error",
                                "unknown", 0.0, 0.0, 0.0)

    @staticmethod
    def _result(is_bovine, reason, top_class, top_conf,
                bovine_signal, animal_signal):
        return {
            "is_bovine": is_bovine,
            "reason": reason,
            "top_class": top_class,
            "top_confidence": round(top_conf * 100, 2),
            "bovine_signal": round(bovine_signal * 100, 2),
            "animal_signal": round(animal_signal * 100, 2),
        }
