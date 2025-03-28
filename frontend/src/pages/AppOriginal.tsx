// @ts-nocheck
import { FileUploader, FileUploaderContent, FileUploaderItem, FileInput } from "@/components/ui/fileinput";
import { useRef, useEffect, useState } from 'react';
import { Paperclip } from "lucide-react";
import { Button } from "@/components/ui/button";

const ImagePreview = ({ title, src }) => (
    <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg shadow-md">
        <h4 className="text-lg font-medium mb-2">{title}</h4>
        <img className="rounded-md shadow-md max-w-xs"
            src={src}
            alt={title}
        />
    </div>
);

const PredictionDisplay = ({ prediction, imagePreview, attentionHeatmapSrc, gradcamHeatmapSrc }: any) => {
    return (
        <div className="mt-8 border border-2 p-8 rounded-3xl">
            <h3 className="text-xl font-semibold text-center">
                Prediction: {
                    prediction.predicted_class === 0
                        ? "Normal"
                        : prediction.predicted_class === 1
                            ? "Benign"
                            : "Malignant"
                }
            </h3>
            <h3 className="text-lg text-center">Confidence: {(prediction.confidence * 100).toFixed(2)}%</h3>
            <div className="flex justify-center gap-5 mt-6">
                {imagePreview && (
                    <ImagePreview
                        title="Original Image"
                        src={imagePreview}
                    />
                )}
                {attentionHeatmapSrc && (
                    <ImagePreviewWithCanvas
                        title="Attention Heatmap"
                        originalSrc={imagePreview}
                        heatmapSrc={attentionHeatmapSrc}
                    />)
                }
                {gradcamHeatmapSrc && (
                    <ImagePreviewWithCanvas
                        title="Grad-CAM Heatmap"
                        originalSrc={imagePreview}
                        heatmapSrc={gradcamHeatmapSrc} />
                )}
            </div>

        </div>
    );
};

const FileSvgDraw = () => {
    return (
        <>
            <svg
                className="w-8 h-8 mb-3 text-gray-500 dark:text-gray-400"
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 20 16"
            >
                <path
                    stroke="currentColor"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
                />
            </svg>
            <p className="mb-1 text-sm text-gray-500 dark:text-gray-400">
                <span className="font-semibold">Click to upload</span>
                &nbsp; or drag and drop
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">
                SVG, PNG, JPG or JPEG
            </p>
        </>
    );
};



const ImagePreviewWithCanvas = ({ title, originalSrc, heatmapSrc }: any) => {
    const canvasRef = useRef(null);

    useEffect(() => {
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
        <div className="flex flex-col items-center">
            <h4 className="text-lg font-medium mb-2">{title}</h4>
            <canvas ref={canvasRef} className="rounded-md shadow-md max-w-xs" />
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

    const handleFileSelect = (file, preview) => {
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

    const dropZoneConfig = {
        maxFiles: 1,
        maxSize: 1024 * 1024 * 4,
        multiple: true,
    };

    return (
        <section className='flex flex-col items-center justify-center pt-8 h-full space-y-8'>
            <div className="border border-2 p-8 rounded-3xl">
                <h1 className='p-2 rounded-2xl text-6xl font-semibold tracking-tighter'>
                    Gallbladder Cancer Detection
                </h1>

                <div className="flex flex-col justify-center items-center">
                    <FileUploader
                        value={files}
                        onValueChange={handleFileSelect}
                        dropzoneOptions={dropZoneConfig}
                        className="relative bg-background rounded-lg p-2"
                    >
                        <FileInput className="p-6 outline-dashed outline-1 outline-black">
                            <div className="flex items-center justify-center flex-col pt-3 pb-4 w-full ">
                                <FileSvgDraw />
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

                    <Button disabled={files?.length < 1} onClick={handleUpload} className="mt-2">
                        Analyze
                    </Button>
                </div>
            </div>

            {error && <p className="text-red-500">{error}</p>}

            {prediction && (
                <PredictionDisplay
                    prediction={prediction}
                    imagePreview={imagePreview}
                    attentionHeatmapSrc={attentionHeatmapSrc}
                    gradcamHeatmapSrc={gradcamHeatmapSrc}
                />
            )}
        </section>
    )
}

export default App