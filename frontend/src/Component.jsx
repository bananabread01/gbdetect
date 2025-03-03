import React, { useState, useRef, useEffect } from "react";

function Component() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [attentionHeatmapSrc, setAttentionHeatmapSrc] = useState(null);
  const [gradcamHeatmapSrc, setGradcamHeatmapSrc] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);

    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select a file first.");
      return;
    }
    setError(null);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Server error");
      }

      const data = await response.json();
      setPrediction(data);

      if (data.attention_heatmap) {
        setAttentionHeatmapSrc(`data:image/png;base64,${data.attention_heatmap}`);
      }
      if (data.gradcam_heatmap) {
        setGradcamHeatmapSrc(`data:image/png;base64,${data.gradcam_heatmap}`);
      }
    } catch (err) {
      console.error("Error uploading file:", err);
      setError("There was an error processing your request.");
    }
  };

  const ImagePreviewWithCanvas = ({ title, originalSrc, heatmapSrc }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
      if (canvasRef.current && originalSrc && heatmapSrc) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        const originalImg = new Image();
        const heatmapImg = new Image();

        originalImg.onload = () => {
          heatmapImg.onload = () => {
            canvas.width = originalImg.width;
            canvas.height = originalImg.height;
            ctx.drawImage(originalImg, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 0.7;
            ctx.drawImage(heatmapImg, 0, 0, canvas.width, canvas.height);
          };
          heatmapImg.src = heatmapSrc;
        };
        originalImg.src = originalSrc;
      }
    }, [originalSrc, heatmapSrc]);

    return (
      <div>
        <h4>{title}</h4>
        <canvas ref={canvasRef} style={{ borderRadius: "2px", boxShadow: "0 4px 8px rgba(0,0,0,0.1)", maxWidth: "350px" }} />
      </div>
    );
  };

  return (
    <div style={{ maxWidth: "900px", margin: "2rem auto", textAlign: "center" }}>
      <h2>Gallbladder Cancer Detection</h2>
      <div className="file-upload">
        <div className="upload-container">
          <div className="upload-icon">
            <img src="upload.png" alt="" />
          </div>
          <input type="file" accept="image/*" onChange={handleFileChange}></input>
        </div>
      </div>
      <br />
      <button onClick={handleUpload} style={{ marginTop: "1rem", padding: "0.5rem 1rem" }}>
        Upload and Predict
      </button>

      {error && <div style={{ marginTop: "1rem", color: "red" }}>{error}</div>}

      {prediction && (
        <div style={{ marginTop: "2rem" }}>
          <h3>
            Prediction:{" "}
            {prediction.predicted_class === 0
              ? "Normal"
              : prediction.predicted_class === 1
              ? "Benign"
              : "Malignant"}
          </h3>

          <h3>Confidence: {(prediction.confidence * 100).toFixed(2)}%</h3>

          {imagePreview && (attentionHeatmapSrc || gradcamHeatmapSrc) && (
            <div style={{ display: "flex", justifyContent: "center", gap: "20px", marginTop: "1.5rem" }}>
              <div>
                <h4>Original Image</h4>
                <img
                  src={imagePreview}
                  alt="Uploaded Preview"
                  style={{
                    maxWidth: "350px",
                    height: "auto",
                    borderRadius: "2px",
                    boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
                  }}
                />
              </div>

              {attentionHeatmapSrc && (
                <ImagePreviewWithCanvas title="Attention Heatmap" originalSrc={imagePreview} heatmapSrc={attentionHeatmapSrc} />
              )}

              {gradcamHeatmapSrc && (
                <ImagePreviewWithCanvas title="Grad-CAM Heatmap" originalSrc={imagePreview} heatmapSrc={gradcamHeatmapSrc} />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Component;