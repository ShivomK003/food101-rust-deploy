import { useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://localhost:8080";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);

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

    const formData = new FormData();
    formData.append("image", file);

    const res = await axios.post(`${API_URL}/predict`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });

    setResult(res.data);
  };

  return (
    <main className="container">
      <h1>Food101 Classifier</h1>
      <p>MobileNetV3 fine-tuned food image classifier deployed with Rust.</p>

      <input type="file" accept="image/*" onChange={handleFileChange} />

      {preview && <img src={preview} alt="preview" className="preview" />}

      <button onClick={handlePredict}>Predict Food</button>

      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </main>
  );
}

export default App;
