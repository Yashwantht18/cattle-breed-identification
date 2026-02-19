# Indian Bovine Breed Recognition System (Edge AI)

## 🚀 Production-Grade Edge AI for Livestock Management
**Empowering Rural India with Offline, Real-Time Breed Identification**

This system is a **deployed, edge-native product** designed for low-connectivity environments. Unlike cloud-based demos, this solution runs fully offline using optimized TensorFlow Lite models, ensuring farmers and veterinarians can identify breeds and access critical husbandry information even in "Airplane Mode".

---

## 🌟 Key Features

### 1. ⚡ **100% Offline Edge Inference**
- **Core Engine**: TensorFlow Lite (MobileNetV2) running locally.
- **Latency**: < 200ms on standard CPUs.
- **Zero Internet Required**: Once installed, the system functions without any network dependency.

### 2. 🧬 **Precision Breed Identification**
- **Coverage**: 41 Indigenous Indian Cattle & Buffalo breeds.
- **Accuracy**: Fine-tuned on 4,000+ curated images.
- **Confidence Gates**: Automatically rejects non-bovine or low-confidence images to prevent misinformation.

### 3. 🧠 **Actionable Knowledge Layer**
- **Region-Mapping**: Identifies native regions for each breed.
- **Husbandry Insights**: Provides breed-specific feed recommendations and health/disease awareness.
- **Suitability**: Helps assess if a breed is suitable for the local climate.

### 4. 🔍 **Explainable AI (Trust & Transparency)**
- **Grad-CAM Integration**: Visualizes *why* the model made a prediction by highlighting key physical features (horns, hump, color patterns).
- **Transparency**: Builds trust with users by showing the model's "focus".

### 5. 📱 **Robust Frontend Application**
- **Responsive UI**: React-based interface optimized for tablets and laptops.
- **Dual Mode**: Upload from gallery or capture directly via camera.
- **Top-3 Predictions**: Displays alternative matches when confidence is split.

---

## 🛠️ Technology Stack

| Component | Technology | Role |
|-----------|------------|------|
| **Model Architecture** | MobileNetV2 (Transfer Learning) | Lightweight, high-accuracy CNN |
| **Inference Engine** | TensorFlow Lite | Optimized on-device execution |
| **Backend** | Python (Flask) | Serves the TFLite runtime & API |
| **Frontend** | React + Vite | Interactive, responsive UI |
| **Explainability** | TF-GradCAM | Heatmap generation |
| **Data Format** | JSON (Local) | Zero-latency knowledge retrieval |

---

## 📂 Project Structure

```
/cattle-breed-identification
├── models/                  # Deployment Artifacts
│   ├── optimized_model.tflite  # The Edge Model
│   ├── classes.txt             # Label Map
│   └── best_model.keras        # (Optional) For Grad-CAM
├── knowledge_base/
│   └── breed_data.json      # Offline Knowledge Database
├── inference/
│   └── tflite_inference.py  # Production Inference Script
├── backend_api/             # Flask Backend
├── frontend_app/            # React Frontend
└── model_training/          # Training Pipelines
```

---

## 🚀 Getting Started (Deployment)

### Prerequisites
- Python 3.9+
- Node.js & npm

### 1. Start the Edge Server (Backend)
The backend loads the TFLite model and serves the API.
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python backend_api/app.py
```
*Server will start on http://localhost:5000*

### 2. Launch the Interface (Frontend)
```bash
cd frontend_app
npm install
npm run dev
```
*Access the app at http://localhost:5173*

---

## 🛡️ Reliability & Safety
- **Airplane Mode**: Test by turning off Wi-Fi. The system will continue to identify breeds and provide data.
- **Disclaimer**: This tool is for decision support. Veterinary advice should always take precedence.

---

## 📜 Dataset & Training
- Trained on **Indian Bovine Breed Recognition.v1i** (~4k images).
- **Class Imbalance**: Handled via augmentation (rotation, zoom, flips) and focal loss considerations.
- **Validation**: Rigorous 80/20 split ensure generalization.

---

**Developed for the Future of Rural Livestock Care.**
