import React, { useState } from "react";

const Header = () => (
  <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "1rem", backgroundColor: "#004080", color: "white" }}>
    <h1>Gallbladder Cancer Detection</h1>
    <nav>
      <a href="/home" style={{ color: "white", marginRight: "1rem" }}>Home</a>
      <a href="/about" style={{ color: "white", marginRight: "1rem" }}>About</a>
      <a href="/contact" style={{ color: "white" }}>Contact</a>
    </nav>
  </header>
);

const ImageUploader = ({ onFileSelect }) => {
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => onFileSelect(file, reader.result);
      reader.readAsDataURL(file);
    }
  };

  return (
    <div style={{ border: "2px dashed #ccc", padding: "20px", textAlign: "center", borderRadius: "10px", margin: "1rem auto", maxWidth: "400px" }}>
      <p>Drag & drop or click to upload an image</p>
      <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: "none" }} id="fileInput" />
      <label htmlFor="fileInput" style={{ cursor: "pointer", color: "#004080" }}>Choose File</label>
    </div>
  );
};

const PredictionDisplay = ({ prediction, imagePreview, attentionHeatmapSrc, gradcamHeatmapSrc }) => {
  return (
    <div style={{ marginTop: "2rem" }}>
      <h3>
        Prediction: {prediction.predicted_class === 0 ? "Normal" : prediction.predicted_class === 1 ? "Benign" : "Malignant"}
      </h3>
      <h3>Confidence: {(prediction.confidence * 100).toFixed(2)}%</h3>
      {imagePreview && (attentionHeatmapSrc || gradcamHeatmapSrc) && (
        <div style={{ display: "flex", justifyContent: "center", gap: "20px", marginTop: "1.5rem" }}>
          <ImagePreview title="Original Image" src={imagePreview} />
          {attentionHeatmapSrc && (<ImagePreviewWithCanvas title="Attention Heatmap" originalSrc={imagePreview} heatmapSrc={attentionHeatmapSrc} />)}
          {gradcamHeatmapSrc && (<ImagePreviewWithCanvas title="Grad-CAM Heatmap" originalSrc={imagePreview} heatmapSrc={gradcamHeatmapSrc} />)}
          {/* {attentionHeatmapSrc && <ImagePreview title="Attention Heatmap" src={attentionHeatmapSrc} />}
          {gradcamHeatmapSrc && <ImagePreview title="Grad-CAM Heatmap" src={gradcamHeatmapSrc} />} */}
        </div>
      )}
    </div>
  );
};

const ImagePreview = ({ title, src }) => (
  <div>
    <h4>{title}</h4>
    <img
      src={src}
      alt={title}
      style={{
        width: "350px",
        maxWidth: "350px",
        height: "auto",
        borderRadius: "2px",
        boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
        //opacity: title.includes("Heatmap") ? 0.6 : 1, // Reducing intensity for heatmaps
      }}
    />
  </div>
);

const ImagePreviewWithCanvas = ({ title, originalSrc, heatmapSrc }) => {
  const canvasRef = React.useRef(null);

  React.useEffect(() => {
    if (canvasRef.current && originalSrc && heatmapSrc) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      const originalImg = new Image();
      const heatmapImg = new Image();

      originalImg.src = originalSrc;
      heatmapImg.src = heatmapSrc;

      originalImg.onload = () => {
        canvas.width = originalImg.width;
        canvas.height = originalImg.height;
        ctx.drawImage(originalImg, 0, 0, canvas.width, canvas.height);

        heatmapImg.onload = () => {
          ctx.globalAlpha = 0.7; // Adjust intensity here (0 = invisible, 1 = full intensity)
          ctx.drawImage(heatmapImg, 0, 0, canvas.width, canvas.height);
        };
      };
    }
  }, [originalSrc, heatmapSrc]);

  return (
    <div>
      <h4>{title}</h4>
      <canvas ref={canvasRef} style={{ borderRadius: "2px", boxShadow: "0 4px 8px rgba(0,0,0,0.1)", maxWidth: "350px" }} />
    </div>
  );
};


function GallbladderCancerDetection() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [attentionHeatmapSrc, setAttentionHeatmapSrc] = useState(null);
  const [gradcamHeatmapSrc, setGradcamHeatmapSrc] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelect = (file, preview) => {
    setSelectedFile(file);
    setImagePreview(preview);
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
      console.log(data);
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

  return (
    <div style={{ maxWidth: "900px", margin: "2rem auto", textAlign: "center" }}>
      <Header />
      <ImageUploader onFileSelect={handleFileSelect} />
      <div style={{ marginTop: "1rem" }}>
        <button onClick={handleUpload} style={{ padding: "0.7rem 1.5rem", backgroundColor: "#004080", color: "white", border: "none", borderRadius: "5px", cursor: "pointer" }}>
          Upload and Predict
        </button>
      </div>
      {error && <div style={{ marginTop: "1rem", color: "red" }}>{error}</div>}
      {prediction && (
        <PredictionDisplay
          prediction={prediction}
          imagePreview={imagePreview}
          attentionHeatmapSrc={attentionHeatmapSrc}
          gradcamHeatmapSrc={gradcamHeatmapSrc}
        />
      )}
    </div>
  );
}

export default GallbladderCancerDetection;
