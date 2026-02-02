import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from inference.inference import BovineClassifier

import io
import base64
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from explainability.gradcam import GradCAM, apply_heatmap

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'models/cattle_resnet50.pth'
CLASS_PATH = 'models/classes.txt'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classifier = None
gradcam = None

def get_classifier():
    global classifier, gradcam
    if classifier is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_PATH):
            if not os.path.exists('models'): os.makedirs('models')
            if not os.path.exists(CLASS_PATH):
                with open(CLASS_PATH, 'w') as f:
                    f.write('\n'.join(['Gir', 'Sahiwal', 'Red Sindhi', 'Tharparkar', 'Kankrej', 'Murrah', 'Jaffrabadi', 'Surti']))
            
        classifier = BovineClassifier(MODEL_PATH, CLASS_PATH)
        gradcam = GradCAM(classifier.model, classifier.model.layer4)
    return classifier, gradcam

API_KEY = "dummy_key"

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
        results, input_tensor, is_bovine = clf.predict(img_path)
        
        heatmap = gc.generate_heatmap(input_tensor)
        
        original_img = cv2.imread(img_path)
        if original_img is None:
            return jsonify({"error": "Could not read uploaded image"}), 500
            
        superimposed = apply_heatmap(heatmap, original_img)
        
        _, buffer = cv2.imencode('.jpg', superimposed)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "animal_type": results[0]['animal_type'],
            "predicted_breed": results[0]['breed'],
            "confidence": results[0]['confidence'],
            "is_bovine": is_bovine,
            "top_3": results,
            "heatmap": f"data:image/jpeg;base64,{heatmap_base64}"
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

