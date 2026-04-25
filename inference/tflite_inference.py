
import numpy as np
from PIL import Image
import os
import json
import time

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Fallback: use tflite_runtime if available (lighter install)
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        tflite = None

class TFLiteBovineClassifier:
    def __init__(self, model_path=None, classes_path=None, knowledge_path=None):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Default paths if not provided
        if not model_path:
            model_path = os.path.join(self.project_root, 'models', 'optimized_model.tflite')
        
        if not classes_path:
            classes_path = os.path.join(self.project_root, 'models', 'classes.txt')
            
        if not knowledge_path:
            knowledge_path = os.path.join(self.project_root, 'knowledge_base', 'breed_data.json')
            
        self.model_path     = model_path
        self.classes_path   = classes_path
        self.knowledge_path = knowledge_path
        self.threshold_path = os.path.join(self.project_root, 'models', 'breed_thresholds.json')

        self.interpreter    = None
        self.input_details  = None
        self.output_details = None
        self.classes        = []
        self.knowledge_db   = {}
        self.breed_thresholds = {}   # per-breed adaptive confidence thresholds

        # Load resources
        self.load_classes()
        self.load_knowledge()
        self.load_thresholds()
        self.load_model()
        
    def load_classes(self):
        if os.path.exists(self.classes_path):
            with open(self.classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.classes)} classes from {self.classes_path}")
        else:
            print(f"Warning: Classes file not found at {self.classes_path}")
            
    def load_knowledge(self):
        if os.path.exists(self.knowledge_path):
            try:
                with open(self.knowledge_path, 'r') as f:
                    data = json.load(f)
                    # Convert list to dict for faster lookup
                    self.knowledge_db = {item['breed']: item for item in data}
                print(f"Loaded knowledge base for {len(self.knowledge_db)} breeds")
            except Exception as e:
                print(f"Error loading knowledge base: {e}")
        else:
            print(f"Warning: Knowledge base not found at {self.knowledge_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                if TF_AVAILABLE:
                    self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
                elif tflite:
                    self.interpreter = tflite.Interpreter(model_path=self.model_path)
                else:
                    print("ERROR: Neither tensorflow nor tflite_runtime is installed.")
                    print("Run: pip install tensorflow  OR  pip install tflite-runtime")
                    return
                self.interpreter.allocate_tensors()
                self.input_details  = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"TFLite model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading TFLite model: {e}")
        else:
            print(f"Warning: TFLite model not found at {self.model_path}")

    def load_thresholds(self):
        """Load per-breed adaptive confidence thresholds.
        HIGH (>300 train images) → 40%,  MED (100-300) → 28%,  LOW (<100) → 18%
        """
        if os.path.exists(self.threshold_path):
            try:
                with open(self.threshold_path, 'r') as f:
                    raw = json.load(f)
                self.breed_thresholds = {
                    k: v['low_conf_threshold']
                    for k, v in raw.items()
                    if not k.startswith('_') and isinstance(v, dict)
                }
                print(f"Loaded adaptive thresholds for {len(self.breed_thresholds)} breeds")
            except Exception as e:
                print(f"Warning: Could not load breed thresholds: {e}")
        else:
            print("Warning: breed_thresholds.json not found — using flat 28% threshold")

    def get_breed_threshold(self, breed_name):
        """Return adaptive low-confidence threshold for this breed (default 28%)."""
        return self.breed_thresholds.get(breed_name, 28)

    def preprocess_image(self, image_path):

        input_shape = self.input_details[0]['shape']
        target_size = (input_shape[1], input_shape[2])
        
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32)

        # The new Google Colab TFLite model handles its own internal rescaling.
        # Natively passing float32 array in [0, 255] range prevents activation collapse.
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array





    def predict(self, image_path, confidence_threshold=10.0, use_tta=True):
        if not self.interpreter:
            return {"error": "Model not loaded"}, None

        start_time = time.time()
        
        try:
            # 1. Standard Prediction
            input_tensor = self.preprocess_image(image_path)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            output_std = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            final_predictions = output_std

            # 2. Test Time Augmentation (TTA) - Optional Accuracy Boost
            if use_tta:
                # Flip Horizontal
                img_pil = Image.open(image_path).convert('RGB')
                target_size = (self.input_details[0]['shape'][1], self.input_details[0]['shape'][2])
                
                # Flip — normalise same as preprocess_image() → [0, 1]
                img_flip = img_pil.transpose(Image.FLIP_LEFT_RIGHT).resize(target_size)
                arr_flip = np.expand_dims(np.array(img_flip, dtype=np.float32) / 255.0, axis=0)
                
                self.interpreter.set_tensor(self.input_details[0]['index'], arr_flip)
                self.interpreter.invoke()
                output_flip = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                
                # Average (50% Original + 50% Flip)
                final_predictions = (output_std + output_flip) / 2.0

            # ---------------------------------------------------------------
            # CONFUSION PAIR CALIBRATION
            # Based on v2 classification report, certain breed pairs are
            # systematically confused. We apply gentle per-class multipliers
            # derived from precision/recall imbalance to correct this bias.
            #
            # Key pairs from report:
            #   Nagpuri: precision=0.62, recall=0.24  -> OVER-predicted
            #   Murrah:  precision=0.36, recall=0.17  -> UNDER-predicted
            #   Jaffrabadi: precision=0.30, recall=0.20 -> UNDER-predicted
            #   Red_Sindhi: precision=0.26, recall=0.36 -> OVER-predicted
            #   Sahiwal: precision=0.68, recall=0.49  -> UNDER-predicted
            #
            # Formula: multiplier = sqrt(recall / precision)
            # This gently boosts under-predicted and reduces over-predicted.
            # ---------------------------------------------------------------
            # Previously used to artificially balance imbalanced datasets.
            # The current Colab model was trained on a perfectly balanced
            # 500-image dataset, so calibration multipliers are no longer needed
            # and actually hurt performance by creating artificial bias.
            CALIBRATION = {}
            if self.classes:
                for i, cls in enumerate(self.classes):
                    if cls in CALIBRATION:
                        final_predictions[i] *= CALIBRATION[cls]
                # Re-normalise after calibration
                total = final_predictions.sum()
                if total > 0:
                    final_predictions = final_predictions / total

            # ---------------------------------------------------------------
            # TEMPERATURE SHARPENING
            # The model was trained with label_smoothing=0.1, which spreads
            # probability mass and makes all predictions look "soft".
            # Temperature T < 1 sharpens the distribution:
            #   p_sharp[i] = p[i]^(1/T) / sum(p[j]^(1/T))
            # T=0.5 is a good empirical choice for label_smoothed models.
            # This does NOT change WHICH class is predicted -- only how
            # confidently the score is displayed.
            # ---------------------------------------------------------------
            TEMPERATURE = 0.5
            p = np.power(np.maximum(final_predictions, 1e-10), 1.0 / TEMPERATURE)
            predictions = p / p.sum()

        except Exception as e:
            print(f"Inference error: {e}")
            return {"error": str(e)}
        # Get top-5 (we show top-3 in UI, but compute 5 for robustness)
        top_k_indices = predictions.argsort()[-5:][::-1]
        
        results = []
        for idx in top_k_indices:
            score = float(predictions[idx]) * 100
            if idx < len(self.classes):
                breed_name = self.classes[idx]
                
                # Get knowledge info
                knowledge = self.knowledge_db.get(breed_name, {})
                animal_type = knowledge.get('animal_type', 'Unknown')
                
                results.append({
                    "breed": breed_name,
                    "confidence": score,
                    "animal_type": animal_type,
                    "knowledge": knowledge
                })
        
        inference_time = (time.time() - start_time) * 1000
        
        top_result = results[0]
        conf       = top_result['confidence']
        breed_name = top_result['breed']

        # --- Adaptive per-breed threshold -----------------------------------
        # Rare breeds (low training data) get a LOWER threshold so they aren't
        # unfairly flagged as "low confidence" just because the model has seen
        # fewer examples. High-data breeds use a stricter threshold.
        breed_threshold = self.get_breed_threshold(breed_name)
        breed_tier_info = self.breed_thresholds  # full dict for tier lookup
        raw_entry = {}
        thresh_path = self.threshold_path
        if os.path.exists(thresh_path):
            try:
                with open(thresh_path) as f:
                    raw_entry = json.load(f).get(breed_name, {})
            except Exception:
                pass
        breed_tier  = raw_entry.get('tier', 'MED')
        train_count = raw_entry.get('train_count', '?')

        is_bovine      = conf >= 5.0
        low_confidence = conf < breed_threshold   # adaptive per-breed!

        if conf >= 60:
            confidence_label = "High"
        elif conf >= 35:
            confidence_label = "Moderate"
        elif conf >= 18:
            confidence_label = "Low — possible crossbreed or ambiguous pose"
        else:
            confidence_label = "Very Low — result unreliable"
        
        if not is_bovine:
            print(f"Rejected: conf={conf:.1f}% < 5%")
        elif low_confidence:
            print(f"Low-conf [{breed_tier} breed, threshold={breed_threshold}%]: "
                  f"{breed_name} = {conf:.1f}%")
        else:
            print(f"Predicted [{breed_tier} breed]: {breed_name} = {conf:.1f}% "
                  f"(threshold={breed_threshold}%, train_count={train_count})")

        # ------------------------------------------------------------------
        # CONFUSION PAIR WARNING
        # When top-1 and top-2 are a known visually-similar pair AND the
        # confidence gap is small, warn the user to verify manually.
        # ------------------------------------------------------------------
        CONFUSION_PAIRS = [
            # Buffalo - body colour identical, only horns differ
            ({"Murrah",     "Nagpuri"},    "Check horn shape: Murrah=tightly coiled, Nagpuri=upward sweep"),
            ({"Murrah",     "Jaffrabadi"}, "Check body size: Jaffrabadi is much larger with flat drooping horns"),
            ({"Jaffrabadi", "Nagpuri"},    "Check horn shape and body mass to distinguish"),
            # Cattle - coat colour nearly identical
            ({"Sahiwal",    "Red_Sindhi"}, "Both reddish-brown - Sahiwal is larger with more dewlap"),
            ({"Nili_Ravi",  "Banni"},      "Both black - check white markings on face and legs"),
            ({"Gir",        "Kankrej"},    "Both dun-coloured Zebu - check hump and horn position"),
        ]
        confusion_warning = None
        if len(results) >= 2:
            top1_name = results[0]['breed']
            top2_name = results[1]['breed']
            gap = results[0]['confidence'] - results[1]['confidence']
            for pair_set, message in CONFUSION_PAIRS:
                if {top1_name, top2_name} == pair_set and gap < 30.0:
                    confusion_warning = message
                    break

        response = {
            "predicted_breed":   breed_name,
            "confidence":        conf,
            "animal_type":       top_result['animal_type'],
            "is_bovine":         is_bovine,
            "low_confidence":    low_confidence,
            "confidence_label":  confidence_label,
            "breed_tier":        breed_tier,
            "breed_threshold":   breed_threshold,
            "train_count":       train_count,
            "inference_time_ms": inference_time,
            "top_3":             results[:3],
            "top_5":             results,
            "confusion_warning": confusion_warning,   # NEW — shown in UI
            "knowledge":         top_result['knowledge']
        }

        return response

if __name__ == "__main__":
    pass
