# Bovine (Cattle & Buffalo) Breed AI

A real-time web application for identifying Indian cattle and buffalo breeds using Deep Learning (ResNet-50) and Explainable AI (Grad-CAM).

## Features
- **Bovine Identification**: Support for both Indian indigenous cattle AND buffalo breeds.
- **Animal Type Recognition**: Automatically classifies the animal as "Cattle" or "Buffalo".
- **Camera Integration**: Capture images directly using the device camera or upload from the gallery.
- **Explainability**: Grad-CAM heatmaps highlighting body regions used for classification.
- **Real-time Inference**: Optimized pipeline ensures fast processing (inference < 3s).
- **Premium UI**: Modern dashboard with mode toggles, confidence indicators, and detailed analysis.

## Dataset
This project uses the **Roboflow "Indian Bovine Breed Recognition"** dataset format, organized for image classification.

## Tech Stack
- **Backend**: Python, Flask, PyTorch, OpenCV, Pillow.
- **Frontend**: React.js, Vite, Framer Motion, react-webcam, Lucide React.
- **Explainability**: Grad-CAM (Gradient-weighted Class Activation Mapping).

## Setup Instructions

### 1. Backend Setup
1. Navigate to the project root.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Flask server:
   ```bash
   python backend_api/app.py
   ```

### 2. Frontend Setup
1. Navigate to `frontend_app`.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

## Model Pipeline
The system utilizes a **ResNet-50** architecture fine-tuned on Indian bovine breeds.
- **Input**: 224x224 RGB image (from File or Camera).
- **Augmentation**: RandomResizedCrop, Flip, Rotation, and Color Jitter are used for robustness.
- **Training**: The script tracks validation accuracy and only saves the **BEST** performing weights to ensure maximum accuracy.

## Training for Higher Accuracy
To retrain the model with your dataset:
1. Ensure images are in `data/train` and `data/val` (organized by breed folder).
2. Run the training script:
   ```bash
   python model_training/train.py
   ```
3. The script will output per-epoch accuracy and save the best model to `models/cattle_resnet50.pth`.
```json
{
  "animal_type": "Buffalo",
  "predicted_breed": "Murrah",
  "confidence": 98.5,
  "top_3": [
    { "breed": "Murrah", "animal_type": "Buffalo", "confidence": 98.5 },
    { "breed": "Gir", "animal_type": "Cattle", "confidence": 1.2 },
    { "breed": "Sahiwal", "animal_type": "Cattle", "confidence": 0.3 }
  ],
  "heatmap": "data:image/jpeg;base64,..."
}
```
