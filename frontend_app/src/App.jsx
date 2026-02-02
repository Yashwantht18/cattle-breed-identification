import React, { useState, useRef, useCallback } from 'react';
import axios from 'axios';
import { Upload, Camera, CheckCircle2, AlertCircle, Loader2, RefreshCw, Smartphone } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Webcam from 'react-webcam';

const videoConstraints = {
  width: 1280,
  height: 720,
  facingMode: "environment"
};

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('upload');

  const webcamRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResults(null);
      setError(null);
    }
  };

  const capture = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        setPreview(imageSrc);
        setFile(imageSrc);
        setResults(null);
        setError(null);
      }
    }
  }, [webcamRef]);

  const handleSubmit = async () => {
    if (!file) {
      setError("Please provide an image first");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      let response;
      if (typeof file === 'string') {
        response = await axios.post('/api/predict', { image: file });
      } else {
        const formData = new FormData();
        formData.append('image', file);
        response = await axios.post('/api/predict', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
      }
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || "Error connecting to backend");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="container">
      <header>
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          Breed AI
        </motion.h1>
        <motion.p
          className="subtitle"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          Identify Indian Cattle & Buffalo Breeds with AI Explainability
        </motion.p>
      </header>

      <main className="main-grid">
        <motion.div
          className="card"
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <div className="card-header" style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2>Input Source</h2>
            <div className="mode-toggle">
              <button
                className={`toggle-btn ${mode === 'upload' ? 'active' : ''}`}
                onClick={() => { setMode('upload'); reset(); }}
              >
                <Upload size={16} /> Upload
              </button>
              <button
                className={`toggle-btn ${mode === 'camera' ? 'active' : ''}`}
                onClick={() => { setMode('camera'); reset(); }}
              >
                <Camera size={16} /> Camera
              </button>
            </div>
          </div>

          {!preview ? (
            mode === 'upload' ? (
              <label className="upload-zone">
                <Upload size={48} className="text-primary" />
                <span>Drag & drop or click to upload bovine image</span>
                <input type="file" hidden onChange={handleFileChange} accept="image/*" />
              </label>
            ) : (
              <div className="camera-zone">
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  videoConstraints={videoConstraints}
                  className="webcam-view"
                />
                <button className="btn-capture" onClick={capture}>
                  <Camera size={24} /> Capture Photo
                </button>
              </div>
            )
          ) : (
            <div style={{ textAlign: 'center' }}>
              <img src={preview} alt="Preview" className="preview-img" />
              <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginTop: '1rem' }}>
                <button
                  className="btn-secondary"
                  onClick={reset}
                >
                  <RefreshCw size={18} /> Retake / Change
                </button>
                <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
                  {loading ? <Loader2 className="spin" size={18} /> : <CheckCircle2 size={18} />}
                  {loading ? 'Analyzing...' : 'Identify Breed'}
                </button>
              </div>
            </div>
          )}

          {error && (
            <motion.div
              className="error-msg"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <AlertCircle size={20} /> {error}
            </motion.div>
          )}

          {loading && (
            <div style={{ textAlign: 'center', marginTop: '2rem' }}>
              <div className="loader-container">
                <span className="loader"></span>
              </div>
              <p className="text-muted processing">Neural Engine Inference in Progress...</p>
            </div>
          )}
        </motion.div>

        <motion.div
          className="card"
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <div className="card-header" style={{ marginBottom: '1.5rem' }}>
            <h2>Diagnostic Results</h2>
          </div>

          {!results && !loading && (
            <div className="empty-state">
              <Smartphone size={64} className="empty-icon" />
              <p>Provide an image from your camera or gallery to begin identification.</p>
            </div>
          )}

          <AnimatePresence>
            {results && (
              <motion.div
                className="results-panel"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
              >
                {!results.is_bovine ? (
                  <div className="not-bovine-state" style={{ textAlign: 'center', padding: '2rem' }}>
                    <AlertCircle size={64} color="#f59e0b" style={{ marginBottom: '1rem' }} />
                    <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Not Detected</h2>
                    <p style={{ color: 'var(--text-muted)' }}>This image does not appear to be an Indian Cattle or Buffalo breed. Please ensure the animal is clearly visible.</p>
                    <button className="btn-secondary" onClick={reset} style={{ marginTop: '1.5rem', marginInline: 'auto' }}>
                      <RefreshCw size={18} /> Try Another Image
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="top-prediction">
                      <div className="animal-type-badge">
                        {results.animal_type}
                      </div>
                      <div className="breed-name">
                        {results.predicted_breed}
                      </div>
                      <div className="confidence-main">
                        {results.confidence.toFixed(1)}% Confidence
                      </div>
                    </div>

                    <div className="analysis-section">
                      <h3 className="section-title">Probability Distribution</h3>
                      {results.top_3.map((res, i) => (
                        <div key={i} className="prediction-row">
                          <div className="prediction-info">
                            <span className="res-breed">{res.breed} <small>({res.animal_type})</small></span>
                            <span className="res-conf">{res.confidence.toFixed(1)}%</span>
                          </div>
                          <div className="confidence-track">
                            <motion.div
                              className="confidence-fill"
                              initial={{ width: 0 }}
                              animate={{ width: `${res.confidence}%` }}
                              transition={{ duration: 1, delay: 0.2 }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="heatmap-section">
                      <h3 className="section-title">Explainable AI (Grad-CAM)</h3>
                      <p className="section-desc">
                        High-intensity regions indicate areas the model focused on for identification:
                      </p>
                      <div className="heatmap-container">
                        <img src={results.heatmap} alt="Grad-CAM" className="heatmap-img" />
                      </div>
                    </div>
                  </>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </main>
    </div>
  );
}

export default App;

