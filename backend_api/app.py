
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from inference.tflite_inference import TFLiteBovineClassifier
from explainability.tf_gradcam import TFGradCAM, apply_heatmap

import io
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# Use absolute paths or relative to project root
PROJECT_ROOT = root_path
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_TFLITE_PATH = os.path.join(MODELS_DIR, 'optimized_model.tflite')
MODEL_KERAS_PATH = os.path.join(MODELS_DIR, 'best_model.keras')
CLASS_PATH = os.path.join(MODELS_DIR, 'classes.txt')
KNOWLEDGE_PATH = os.path.join(PROJECT_ROOT, 'knowledge_base', 'breed_data.json')

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classifier = None
gradcam = None

def get_classifier():
    global classifier, gradcam
    if classifier is None:
        classifier = TFLiteBovineClassifier(
            model_path=MODEL_TFLITE_PATH, 
            classes_path=CLASS_PATH,
            knowledge_path=KNOWLEDGE_PATH
        )
    
    # Initialize Grad-CAM if Keras model exists
    if gradcam is None and os.path.exists(MODEL_KERAS_PATH):
        try:
            print(f"Initializing Grad-CAM with {MODEL_KERAS_PATH}")
            gradcam = TFGradCAM(MODEL_KERAS_PATH)
        except Exception as e:
            print(f"Failed to load Grad-CAM model: {e}")
            
    return classifier, gradcam

@app.route('/predict', methods=['POST'])
def predict():
    img_path = None
    
    if 'image' in request.files:
        file = request.files['image']
        img_bytes = file.read()
        filename = file.filename or "upload.jpg"
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(img_path, 'wb') as f:
            f.write(img_bytes)
    elif request.is_json and 'image' in request.json:
        image_data = request.json['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        try:
            img_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({"error": f"Invalid base64 data: {str(e)}"}), 400
            
        img_path = os.path.join(UPLOAD_FOLDER, "camera_capture.jpg")
        with open(img_path, 'wb') as f:
            f.write(img_bytes)
    else:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        clf, gc = get_classifier()
        
        # 1. Run inference (TFLite)
        response = clf.predict(img_path)
        
        if "error" in response:
            return jsonify(response), 500

        # 2. Run Grad-CAM (Keras) if available and if it's a bovine
        heatmap_base64 = None
        if gc and response.get('is_bovine', False):
            try:
                # Preprocess image for Keras model (same as TFLite usually)
                input_tensor = clf.preprocess_image(img_path)
                
                # Get index of predicted class
                # We need the index corresponding to the predicted breed name
                # TFLite classifier stores classes list
                breed_name = response['predicted_breed']
                if breed_name in clf.classes:
                    pred_index = clf.classes.index(breed_name)
                    heatmap = gc.compute_heatmap(input_tensor, pred_index=pred_index)
                    
                    original_img = cv2.imread(img_path)
                    if original_img is not None:
                        superimposed = apply_heatmap(heatmap, original_img)
                        _, buffer = cv2.imencode('.jpg', superimposed)
                        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
                        heatmap_base64 = f"data:image/jpeg;base64,{heatmap_base64}"
            except Exception as e:
                print(f"Grad-CAM failed: {e}")
        
        # Add heatmap to response
        response['heatmap'] = heatmap_base64
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up the uploaded file after every request (success or failure)
        if img_path and os.path.exists(img_path):
            os.remove(img_path)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
