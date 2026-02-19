
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import time

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
            
        self.model_path = model_path
        self.classes_path = classes_path
        self.knowledge_path = knowledge_path
        
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.classes = []
        self.knowledge_db = {}
        
        # Load resources
        self.load_classes()
        self.load_knowledge()
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
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"TFLite model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading TFLite model: {e}")
        else:
            print(f"Warning: TFLite model not found at {self.model_path} (Training might be in progress)")



    def preprocess_image(self, image_path):
        input_shape = self.input_details[0]['shape']
        target_size = (input_shape[1], input_shape[2])
        
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1] as MobileNetV2 expects or used in training (rescale=1./255)
        img_array = img_array / 255.0
        
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
                
                # Flip
                img_flip = img_pil.transpose(Image.FLIP_LEFT_RIGHT).resize(target_size)
                arr_flip = np.expand_dims(np.array(img_flip, dtype=np.float32) / 255.0, axis=0)
                
                self.interpreter.set_tensor(self.input_details[0]['index'], arr_flip)
                self.interpreter.invoke()
                output_flip = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                
                # Average (50% Original + 50% Flip)
                # Simple TTA often helps with pose variance
                final_predictions = (output_std + output_flip) / 2.0
                
            predictions = final_predictions
            
        except Exception as e:
            print(f"Inference error: {e}")
            return {"error": str(e)}, None
        
        # Get top-3
        top_k_indices = predictions.argsort()[-3:][::-1]
        
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
        conf = top_result['confidence']
        
        # is_bovine is only False when the model is extremely uncertain (< 5%).
        # Previously was 25% which caused valid predictions to be hidden.
        # For a 41-class model, random chance = 2.4%, so 5% is already conservative.
        is_bovine = conf >= 5.0
        low_confidence = conf < confidence_threshold   # signals UI to show a warning
        
        if not is_bovine:
            print(f"Rejected prediction (conf={conf:.2f}% < 5%)")
        elif low_confidence:
            print(f"Low confidence prediction: {conf:.2f}% (showing best guess)")
        
        response = {
            "predicted_breed": top_result['breed'],
            "confidence": top_result['confidence'],
            "animal_type": top_result['animal_type'],
            "is_bovine": is_bovine,
            "low_confidence": low_confidence,
            "inference_time_ms": inference_time,
            "top_3": results,
            "knowledge": top_result['knowledge']
        }
        
        return response

if __name__ == "__main__":
    pass
