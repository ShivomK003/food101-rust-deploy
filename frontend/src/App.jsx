import { useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://localhost:8080";

function formatClassName(name) {
  return name
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setResult(null);

    if (selected) {
      setPreview(URL.createObjectURL(selected));
    }
  };

  const handlePredict = async () => {
    if (!file) return alert("Choose an image first");

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const res = await axios.post(`${API_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(res.data);
    } catch (error) {
      console.error("Prediction error:", error);
      console.error("Response:", error.response?.data);
      console.error("Status:", error.response?.status);

      alert(error.response?.data?.error || "Prediction failed. Check if the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const topPrediction = result?.predictions?.[0];

  return (
    <main className="page">
      <section className="hero">
        <span className="badge">Rust + ONNX Runtime</span>
        <h1 className="hero-title">Food101 Classifier</h1>
        <p className="hero-description">Fine-tuned MobileNetV3 food recognition deployed as a Rust web API.</p>
      </section>

      <section className="app-grid">
        <div className="panel upload-panel">
          <h2>Upload Image</h2>

          <label className="file-picker">
            Choose image
            <input type="file" accept="image/*" onChange={handleFileChange} />
          </label>

          {file && <p className="filename">{file.name}</p>}

          {preview && <img src={preview} alt="preview" className="preview" />}

          <button onClick={handlePredict} disabled={loading}>
            {loading ? "Analysing..." : "Predict Food"}
          </button>
        </div>

        <div className="panel results-panel">
          {!result && (
            <div className="empty-state">
              <h2>Prediction results</h2>
              <p>Upload a food image and run prediction to see the Top-5 classes.</p>
            </div>
          )}

          {result && topPrediction && (
            <>
              <div className="top-result">
                <p className="label">Top Prediction</p>
                <h2>{formatClassName(topPrediction.class_name)}</h2>
                <p>{(topPrediction.confidence * 100).toFixed(2)}% confidence</p>
              </div>

              <div className="prediction-list">
                {result.predictions.map((prediction, index) => {
                  const percent = prediction.confidence * 100;

                  return (
                    <div
                      className="prediction-row"
                      key={prediction.class_index}
                      style={{ animationDelay: `${index * 80}ms` }}>
                      <div className="prediction-header">
                        <span>
                          {index + 1}. {formatClassName(prediction.class_name)}
                        </span>
                        <strong>{percent.toFixed(2)}%</strong>
                      </div>

                      <div className="bar-background">
                        <div className="bar-fill" style={{ width: `${Math.max(percent, 1)}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      </section>
    </main>
  );
}

export default App;
