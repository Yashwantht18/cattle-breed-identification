
# Indian Bovine Breed Recognition System - Edge AI Product Specification

## 1. Product Overview
This system is a **production-grade, offline Edge AI solution** designed for identifying Indian cattle and buffalo breeds in rural, low-connectivity environments. Unlike cloud-based solutions, this product runs entirely on local hardware (laptops, robust tablets) without requiring an internet connection for inference.

## 2. Key Differentiators
- **100% Offline Inference**: Uses TensorFlow Lite to run deep learning models locally.
- **Edge-Optimized**: Replaced heavy ResNet50 with MobileNetV2 for low latency and efficiency.
- **Comprehensive Knowledge Layer**: Provides breed-specific feed, health, and climate information instantly.
- **Explainable AI**: Includes Grad-CAM visualization to build trust with users (veterinarians/farmers).
- **Dual-Mode Operation**:
  - **Production Mode**: Fast inference using `.tflite`.
  - **Analysis Mode**: Detailed explainability using `.keras` model (when available).

## 3. Technical Architecture

### 3.1 AI Engine
- **Framework**: TensorFlow / TensorFlow Lite
- **Model**: MobileNetV2 (Transfer Learning from ImageNet)
- **Input**: 224x224 RGB Images
- **Output**: 41 Classes (32 Cattle, 9 Buffalo) + Confidence Score
- **Optimization**: Quantized/Float16 TFLite model for CPU/EdgeTPU acceleration.

### 3.2 Knowledge System
- **Database**: Local JSON (`breed_data.json`)
- **Content**:
  - Animal Type (Cattle vs Buffalo)
  - Description
  - Ideal Climate
  - Feed Recommendations
  - Health Notes
- **Update Mechanism**: Simple JSON file update (no code changes needed).

### 3.3 Application Layer
- **Backend**: Flask (Python). Serves the API and handles model inference.
- **Frontend**: React.js (Vite). Responsive, modern UI with "Cyberpunk/High-Tech" aesthetic.
- **Communication**: Localhost HTTP (REST API).

## 4. Dataset & Training
- **Source**: "Indian Bovine Breed Recognition.v1i" (Curated Dataset)
- **Size**: ~4,000+ images across 41 classes.
- **Strategy**:
  - **Training**: Performed on powerful machine (cloud/GPU workstation) to generate artifacts.
  - **Deployment**: Only artifacts (`optimized_model.tflite`, `classes.txt`) are deployed to edge.
  - **Handling Imbalance**: Data Augmentation (Rotation, Zoom, Shift) and Transfer Learning.

## 5. Deployment Instructions
1.  **Prerequisites**: Python 3.9+, Node.js (for frontend build).
2.  **Install Dependencies**: `pip install -r requirements.txt`
3.  **Run Backend**: `python backend_api/app.py`
4.  **Run Frontend**: `npm run dev` (or serve built static files).
5.  **Access**: Open `http://localhost:5173` (default Vite port) or configured host.

## 6. Future Roadmap
- **Mobile App**: Port frontend to React Native or Flutter for Android deployment.
- **Quantization**: Further reduce model size (int8) for microcontroller deployment.
- **Local Logging**: Save inference history locally for later synchronization.
