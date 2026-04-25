"""
==============================================================================
  GATEKEEPER MODULE - Bovine Image Validator  (v4 — decode_predictions fix)
==============================================================================
  Uses a pre-trained MobileNetV2 (ImageNet) to determine if an uploaded image
  contains a bovine animal (cattle or buffalo) before running breed inference.

  KEY FIX (v4):
  - The v2/v3 approach of manually indexing preds[340] etc. was BROKEN.
    MobileNetV2's output array does NOT use the raw ImageNet class numbers
    as indices directly — the mapping is offset and model-specific.
  - We now use tf.keras decode_predictions() to get the synset IDs / class
    names, and check those against known bovine synset names. This is the
    correct, documented Keras API approach.

  Decision policy:
  - ACCEPT if top-5 predictions contain any bovine/large-ungulate class name
    at ≥ BOVINE_THRESHOLD confidence
  - ACCEPT if top-5 predictions contain any large-quadruped proxy class name
    at ≥ ANIMAL_THRESHOLD confidence
  - ACCEPT if bovine+animal combined scores ≥ COMBINED_THRESHOLD
  - REJECT otherwise (default = safe)
==============================================================================
"""

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Bovine / large-ungulate CLASS NAMES (as returned by decode_predictions)
# These are the ImageNet synset labels that MobileNetV2 uses for cattle.
# ---------------------------------------------------------------------------
BOVINE_NAMES = {
    "ox",
    "water_buffalo",
    "bison",
    "bighorn",
    "bighorn_sheep",
    "ram",
    "ibex",
    "hartebeest",
    "impala",
    "gazelle",
    "zebu",           # not always present but included defensively
}

# Large quadrupeds that Indian cattle are commonly mis-classified as by
# ImageNet models (especially zebu breeds with unfamiliar body shapes).
ANIMAL_PROXY_NAMES = {
    "horse",
    "horse_cart",
    "wild_ass",
    "sorrel",           # another horse-related class
    "Indian_elephant",
    "African_elephant",
    "tusker",
    "warthog",          # large African pig — similar body profile
    "Arabian_camel",
    "llama",
}

# ---------------------------------------------------------------------------
# Thresholds (probability, 0-1 scale)
# ---------------------------------------------------------------------------
BOVINE_THRESHOLD   = 0.04   # 4%  — accept if any true bovine class hits this
ANIMAL_THRESHOLD   = 0.20   # 20% — accept if a known large-animal proxy hits this
COMBINED_THRESHOLD = 0.08   # 8%  — accept if bovine+animal combined ≥ this

# ---------------------------------------------------------------------------

class ImageNetGatekeeper:
    """
    Lightweight gatekeeper using MobileNetV2 + ImageNet weights.
    Default policy: REJECT — only accepts images with detected animal signal.
    """

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load MobileNetV2 with ImageNet weights (~14 MB download on first use)."""
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
            print("[Gatekeeper] MobileNetV2 (ImageNet) loaded successfully.")
        except Exception as e:
            print(f"[Gatekeeper] WARNING: Could not load MobileNetV2: {e}")
            print("[Gatekeeper] Running in FAIL-OPEN mode — all images will pass.")
            self.model = None

    def _preprocess(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess for MobileNetV2 ImageNet gatekeeper.
        Uses mobilenet_v2.preprocess_input() → scales pixels to [-1, 1].

        This is CORRECT for the gatekeeper only.
        The breed classifier (EfficientNetV2S with include_preprocessing=True)
        uses a different, incompatible preprocessing path and is NOT affected.
        """
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        arr = self.preprocess_fn(arr)   # → [-1, 1]
        return arr

    def check(self, image_path: str) -> dict:
        """
        Check if an image is likely a bovine.

        Returns:
            {
              "is_bovine":       bool,
              "reason":          str,
              "top_class":       str,
              "top_confidence":  float,   # percentage (0–100)
              "bovine_signal":   float,   # best bovine class score (%)
              "animal_signal":   float,   # best proxy animal score (%)
            }
        """
        if self.model is None:
            return self._result(True, "gatekeeper_unavailable",
                                "unknown", 0.0, 0.0, 0.0)

        try:
            import tensorflow as tf
            input_arr = self._preprocess(image_path)
            preds = self.model.predict(input_arr, verbose=0)   # shape: (1, 1000)

            # decode_predictions returns top-N as (synset_id, class_name, score)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
                preds, top=10
            )[0]  # list of (synset_id, class_name, prob)

            top_class_name = decoded[0][1] if decoded else "unknown"
            top_conf       = float(decoded[0][2]) if decoded else 0.0

            # Scan top-10 predictions for bovine / proxy-animal hits
            bovine_signal = 0.0
            animal_signal = 0.0

            for _, cls_name, prob in decoded:
                name_lower = cls_name.lower()
                # Check bovine set (case-insensitive partial match)
                if any(b.lower() in name_lower or name_lower in b.lower()
                       for b in BOVINE_NAMES):
                    bovine_signal = max(bovine_signal, float(prob))
                # Check proxy animal set
                if any(a.lower() in name_lower or name_lower in a.lower()
                       for a in ANIMAL_PROXY_NAMES):
                    animal_signal = max(animal_signal, float(prob))

            combined = bovine_signal + animal_signal

            print(
                f"[Gatekeeper] top={top_class_name}({top_conf:.2%}) | "
                f"bovine={bovine_signal:.2%} | "
                f"animal={animal_signal:.2%} | "
                f"combined={combined:.2%}"
            )

            # ── Decision logic ────────────────────────────────────────────
            # Rule 1: A known bovine/ungulate class appears at ≥ 4% → ACCEPT
            if bovine_signal >= BOVINE_THRESHOLD:
                return self._result(True, "bovine_detected",
                                    top_class_name, top_conf,
                                    bovine_signal, animal_signal)

            # Rule 2: A large-quadruped proxy appears at ≥ 20% → ACCEPT
            if animal_signal >= ANIMAL_THRESHOLD:
                return self._result(True, "large_animal_detected",
                                    top_class_name, top_conf,
                                    bovine_signal, animal_signal)

            # Rule 3: Combined bovine+proxy ≥ 8% → ACCEPT
            if combined >= COMBINED_THRESHOLD:
                return self._result(True, "combined_animal_signal",
                                    top_class_name, top_conf,
                                    bovine_signal, animal_signal)

            # Rule 4: No meaningful animal signal found → REJECT
            return self._result(False, "no_animal_signal",
                                top_class_name, top_conf,
                                bovine_signal, animal_signal)

        except Exception as e:
            print(f"[Gatekeeper] Error during check ({e}) — passing image through.")
            return self._result(True, "gatekeeper_error",
                                "unknown", 0.0, 0.0, 0.0)

    @staticmethod
    def _result(is_bovine, reason, top_class, top_conf,
                bovine_signal, animal_signal):
        return {
            "is_bovine":       is_bovine,
            "reason":          reason,
            "top_class":       top_class,
            "top_confidence":  round(top_conf * 100, 2),
            "bovine_signal":   round(bovine_signal * 100, 2),
            "animal_signal":   round(animal_signal * 100, 2),
        }
