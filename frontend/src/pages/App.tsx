// @ts-nocheck
import { useRef, useEffect, useState } from "react";
import { Paperclip } from "lucide-react";
import { Button } from "@/components/ui/button";
import { FileUploader, FileUploaderContent, FileUploaderItem, FileInput } from "@/components/ui/fileinput";

const ImagePreview = ({ title, src }) => (
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg shadow-md">
        <h4 className="text-lg font-medium mb-2">{title}</h4>
        <img className="rounded-md shadow-md w-full" src={src} alt={title} />
    </div>
);

const PredictionDisplay = ({ prediction, imagePreview, attentionHeatmapSrc, gradcamHeatmapSrc }) => {
    return (
        <div className="bg-gray-50 dark:bg-gray-900 p-6 rounded-lg shadow-lg space-y-6">
            <h3 className="text-xl font-semibold text-center text-gray-900 dark:text-white">
                Prediction: {prediction.predicted_class === 0 ? "Normal" : prediction.predicted_class === 1 ? "Benign" : "Malignant"}
            </h3>
            <h3 className="text-lg text-center text-gray-700 dark:text-gray-300">
                Confidence: {(prediction.confidence * 100).toFixed(2)}%
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {imagePreview && <ImagePreview title="Original Image" src={imagePreview} />}
                {attentionHeatmapSrc && <ImagePreview title="Attention Heatmap" src={attentionHeatmapSrc} />}
                {gradcamHeatmapSrc && <ImagePreview title="Grad-CAM Heatmap" src={gradcamHeatmapSrc} />}
            </div>
        </div>
    );
};

const App = () => {
    const [files, setFiles] = useState<File[] | null>(null);
    const [imagePreview, setImagePreview] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [attentionHeatmapSrc, setAttentionHeatmapSrc] = useState(null);
    const [gradcamHeatmapSrc, setGradcamHeatmapSrc] = useState(null);
    const [error, setError] = useState(null);

    const handleFileSelect = (file) => {
        setFiles(file);
        const reader = new FileReader();
        reader.readAsDataURL(file[0]);
        reader.onloadend = () => {
            setImagePreview(reader.result);
        };
    };

    const handleUpload = async () => {
        if (!files) {
            setError("Please select a file first.");
            return;
        }

        setError(null);
        const formData = new FormData();
        formData.append("file", files[0]);

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Server error");
            }

            const data = await response.json();
            console.log(data);
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

    return (
        <div className="min-h-screen bg-gray-100 dark:bg-gray-900">

            {/* Main Content Layout */}
            <main className="container mx-auto px-4 py-8 grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Left Side - File Upload */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg space-y-6">
                    <h1 className="text-3xl font-bold text-center text-gray-900 dark:text-white">
                        Gallbladder Cancer Detection
                    </h1>

                    <div className="flex flex-col justify-center items-center">
                    <FileUploader
                        value={files}
                        onValueChange={handleFileSelect}
                        dropzoneOptions={{ accept: { "image/*": [] }, maxFiles: 1, maxSize: 1024 * 1024 * 4 }} // Added dropzoneOptions
                        className="relative bg-background rounded-lg p-2"
                    >
                            <FileInput className="p-6 outline-dashed outline-1 outline-black dark:outline-gray-400">
                                <div className="flex items-center justify-center flex-col pt-3 pb-4 w-full">
                                    <Paperclip className="h-8 w-8 text-gray-500 dark:text-gray-400" />
                                    <p className="text-sm text-gray-500 dark:text-gray-400">
                                        Click to upload or drag and drop
                                    </p>
                                    <p className="text-xs text-gray-500 dark:text-gray-400">SVG, PNG, JPG, or JPEG</p>
                                </div>
                            </FileInput>
                            <FileUploaderContent>
                                {files &&
                                    files.length > 0 &&
                                    files.map((file, i) => (
                                        <FileUploaderItem key={i} index={i}>
                                            <Paperclip className="h-4 w-4 stroke-current" />
                                            <span>{file.name}</span>
                                        </FileUploaderItem>
                                    ))}
                            </FileUploaderContent>
                        </FileUploader>

                        <Button disabled={files?.length < 1} onClick={handleUpload} className="mt-4">
                            Analyze
                        </Button>
                    </div>

                    {error && <p className="text-red-500">{error}</p>}
                </div>

                {/* Right Side - Prediction Results */}
                <div>
                    {prediction ? (
                        <PredictionDisplay
                            prediction={prediction}
                            imagePreview={imagePreview}
                            attentionHeatmapSrc={attentionHeatmapSrc}
                            gradcamHeatmapSrc={gradcamHeatmapSrc}
                        />
                    ) : (
                        <div className="flex flex-col justify-center items-center h-full">
                            <p className="text-gray-500 dark:text-gray-400">Upload an image to see results.</p>
                        </div>
                    )}
                </div>
            </main>

            {/* Footer (Optional) */}
            <footer className="w-full p-4 text-center text-gray-600 dark:text-gray-300">
                © 2025 GBDetect - All rights reserved.
            </footer>
        </div>
    );
};

export default App;
