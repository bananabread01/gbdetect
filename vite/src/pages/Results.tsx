import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";

const Results = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [imageName, setImageName] = useState<string>("Uploaded Scan");
    const [prediction, setPrediction] = useState<{ predicted_class: number; confidence: number } | null>(null);
    const [attentionHeatmapSrc, setAttentionHeatmapSrc] = useState<string | null>(null);
    const [gradcamHeatmapSrc, setGradcamHeatmapSrc] = useState<string | null>(null);
    const [gradcamPlusHeatmapSrc, setGradcamPlusHeatmapSrc] = useState<string | null>(null);
    const [selectedHeatmap, setSelectedHeatmap] = useState<string | null>(null);
    const [zoomScale, setZoomScale] = useState<number>(1); // Zoom level (default 1x)
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const [startPosition, setStartPosition] = useState({ x: 0, y: 0 });


    useEffect(() => {
        if (!location.state?.image || !location.state?.prediction) {
            navigate("/"); // Redirect if no image or prediction data
            return;
        }

        const reader = new FileReader();
        reader.readAsDataURL(location.state.image);
        reader.onloadend = () => {
            if (typeof reader.result === "string") {
                setImagePreview(reader.result);
            }
        };

        setPrediction(location.state.prediction);

        if (location.state.image?.name) {
            setImageName(location.state.image.name); // Get actual uploaded image name
        }

        if (location.state.prediction?.attention_heatmap) {
            const attentionSrc = `data:image/png;base64,${location.state.prediction.attention_heatmap}`;
            setAttentionHeatmapSrc(attentionSrc);
        }

        if (location.state.prediction?.gradcam_heatmap) {
            setGradcamHeatmapSrc(`data:image/png;base64,${location.state.prediction.gradcam_heatmap}`);
        }

        if (location.state.prediction?.gradcam2plus_heatmap) {
            setGradcamPlusHeatmapSrc(`data:image/png;base64,${location.state.prediction.gradcam2plus_heatmap}`);
        }
    }, [location, navigate]);

    const getProtocolText = () => {
        if (!prediction) return null;
    
        switch (prediction.predicted_class) {
            case 0:
                return (
                    <>
                        <p className="text-sm text-green-700 dark:text-green-300 font-medium">Suggested Actions:</p>
                        <ul className="list-disc list-inside text-sm text-gray-800 dark:text-gray-200">
                            <li>Routine reporting. No suspicious findings detected.</li>
                            <li>Optionally note: "AI-assisted review: Normal scan."</li>
                        </ul>
                    </>
                );
            case 1:
                return (
                    <>
                        <p className="text-sm text-yellow-700 dark:text-yellow-300 font-medium">Suggested Actions:</p>
                        <ul className="list-disc list-inside text-sm text-gray-800 dark:text-gray-200">
                            <li>Confirm lesion type (e.g., polyp, stone).</li>
                            <li>Consider follow-up in 6–12 months depending on size and symptoms.</li>
                        </ul>
                    </>
                );
            case 2:
                return (
                    <>
                        <p className="text-sm text-red-700 dark:text-red-300 font-medium">Suggested Actions:</p>
                        <ul className="list-disc list-inside text-sm text-gray-800 dark:text-gray-200">
                            <li>Conduct second read by a senior radiologist.</li>
                            <li>Recommend CT or MRI for further evaluation.</li>
                            <li>Alert referring physician for escalation.</li>
                        </ul>
                    </>
                );
            default:
                return null;
        }
    };    

    // Zoom Controls
    const handleZoomIn = () => {
        if (zoomScale < 3) setZoomScale(zoomScale + 0.2); // Limit max zoom to 3x
    };

    const handleZoomOut = () => {
        if (zoomScale > 1) setZoomScale(zoomScale - 0.2); // Prevent shrinking below original size
    };

    // Dragging (Panning) Functionality
    const handleMouseDown = (e: React.MouseEvent) => {
        if (zoomScale > 1) {
            setIsDragging(true);
            setStartPosition({ x: e.clientX - position.x, y: e.clientY - position.y });
        }
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (isDragging) {
            setPosition({
                x: e.clientX - startPosition.x,
                y: e.clientY - startPosition.y,
            });
        }
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };


    return (
        <section className="min-h-screen flex flex-col bg-gray-100 dark:bg-gray-900 pt-14">
            {/* Main Content (Sidebar + Images) */}
            <div className="flex flex-1">
                {/* Sidebar (Left Column) */}
                <div className="w-1/5 bg-white dark:bg-gray-800 p-6 shadow-lg flex flex-col justify-between border-r-1 border-gray-400">
                    {/* Prediction Results */}
                    <div className="text-center space-y-2">
                        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Prediction Results</h1>
                        {prediction ? (
                            <>
                                <p className="text-lg">
                                    Class:{" "}
                                    <span className="font-bold">
                                        {prediction.predicted_class === 0
                                            ? "Normal"
                                            : prediction.predicted_class === 1
                                            ? "Benign"
                                            : "Malignant"}
                                    </span>
                                </p>
                                <p className="text-lg">
                                    Confidence: <span className="font-bold">{(prediction.confidence * 100).toFixed(2)}%</span>
                                </p>
                            </>
                        ) : (
                            <p className="text-gray-500">Processing...</p>
                        )}

                        {/* Protocol Info Box */}
                        {prediction && (
                            <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-700 rounded-lg border border-gray-300 dark:border-gray-600">
                                <h2 className="text-md font-semibold text-gray-900 dark:text-white mb-1">Clinical Protocol</h2>
                                {getProtocolText()}
                            </div>
                        )}
                    </div>

                    {/* Heatmap Toggle Buttons */}
                    <div className="flex flex-col space-y-2 mt-4 p-3 bg-gray-100 dark:bg-gray-700 rounded-lg border border-gray-300 dark:border-gray-600">
                        <h2 className="text-lg font-bold text-gray-900 dark:text-white text-center">Select Heatmap</h2>
                        <Button 
                            onClick={() => setSelectedHeatmap(attentionHeatmapSrc)} 
                            disabled={!attentionHeatmapSrc}
                            className={`w-full ${selectedHeatmap === attentionHeatmapSrc ? "bg-slate-500 text-white hover:bg-slate-600" : ""}`}
                        >
                            Attention Heatmap
                        </Button>
                        <Button 
                            onClick={() => setSelectedHeatmap(gradcamHeatmapSrc)} 
                            disabled={!gradcamHeatmapSrc}
                            className={`w-full ${selectedHeatmap === gradcamHeatmapSrc ? "bg-slate-500 text-white hover:bg-slate-600" : ""}`}
                        >
                            Grad-CAM Heatmap
                        </Button>
                        <Button 
                            onClick={() => setSelectedHeatmap(gradcamPlusHeatmapSrc)} 
                            disabled={!gradcamPlusHeatmapSrc}
                            className={`w-full ${selectedHeatmap === gradcamPlusHeatmapSrc ? "bg-slate-500 text-white hover:bg-slate-600" : ""}`}
                        >
                            Grad-CAM++ Heatmap
                        </Button>

                        {/* Zoom Controls */}
                        <div className="flex flex-col space-y-2 mt-4">
                            <div className="flex justify-center space-x-2">
                                <h2 className="text-lg font-bold text-gray-900 dark:text-white text-center">Zoom</h2>
                                <Button onClick={handleZoomOut} className="w-1/4 py-1 text-sm">–</Button>
                                <Button onClick={handleZoomIn} className="w-1/4 py-1 text-sm">+</Button>
                            </div>
                        </div>
                    </div>

                    {/* New Session Button */}
                    <Button onClick={() => navigate("/upload")} className="w-full mt-4">
                        New Session
                    </Button>
                </div>

                {/* Middle & Right Column - Images */}
                <div className="flex-1 flex flex-col">
                    <div className="flex-1 flex flex-row justify-center">
                        {/* Original Ultrasound Image */}
                        <div className="flex-1 flex items-center bg-black border-r border-gray-400">
                            {imagePreview && (
                                <img 
                                    src={imagePreview} 
                                    alt="Uploaded Scan" 
                                    className="w-full h-auto object-contain"
                                    // add max-h-full 
                                />
                            )}
                        </div>

                        {/* Selected Heatmap */}
                        <div 
                            className="flex-1 flex bg-black overflow-hidden relative items-center cursor-grab active:cursor-grabbing"
                            onMouseDown={handleMouseDown}
                            onMouseMove={handleMouseMove}
                            onMouseUp={handleMouseUp}
                            onMouseLeave={handleMouseUp}
                        >
                            {selectedHeatmap ? (
                                <img 
                                    src={selectedHeatmap} 
                                    alt="Heatmap" 
                                    style={{ 
                                        transform: `scale(${zoomScale}) translate(${position.x}px, ${position.y}px)`, 
                                        transformOrigin: "center"
                                        //aspectRatio: imageWidth && imageHeight ? `${imageWidth}/${imageHeight}` : "auto",
                                        // width: "100%",
                                        // height: "100%",
                                        // objectFit: "contain"  
                                    }} 
                                    className="w-full h-auto object-cover transition-transform duration-300"
                                /> //max-w-full max-h-full object-contain transition-transform duration-300 
                                   //thisss w-full h-full object-cover transition-transform duration-300
                            ) : (
                                <p className="text-lg text-gray-500 text-center absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                                    Select a heatmap to display
                                </p>
                            )}
                        </div>

                    </div>

                    {/* Footer Box */}
                    <div className="w-full border-t-2 border-gray-400 bg-gray-200 dark:bg-gray-700 text-center py-3 flex justify-between">
                        {/* Original Image */}
                        <div className="w-1/4 text-center">
                            <h2 className="text-lg text-left ml-4 font-bold text-gray-900 dark:text-white">Original Image</h2>
                            <p className="text-md text-left ml-4 text-gray-600 dark:text-gray-300">{imageName}</p>
                        </div>

                        {/* Date Section */}
                        <div className="w-1/4 text-center border-r-2 border-gray-400">
                            <h2 className="text-lg text-right mr-4 font-bold text-gray-900 dark:text-white">Date</h2>
                            <p className="text-md text-right mr-4 text-gray-600 dark:text-gray-300">{new Date().toLocaleDateString()}</p>
                        </div>

                        {/* Heatmap Section */}
                        <div className="w-1/4 text-center border-gray-400">
                            <h2 className="text-lg text-left ml-4 font-bold text-gray-900 dark:text-white">Heatmap</h2>
                            <p className="text-md text-left ml-4 text-gray-600 dark:text-gray-300">
                                {selectedHeatmap 
                                    ? selectedHeatmap === attentionHeatmapSrc 
                                        ? "Attention Heatmap" 
                                        : selectedHeatmap === gradcamHeatmapSrc 
                                            ? "Grad-CAM Heatmap" 
                                            : selectedHeatmap === gradcamPlusHeatmapSrc 
                                                ? "Grad-CAM++ Heatmap" 
                                                : "Heatmap"
                                    : "No Heatmap Selected"
                                }
                            </p>
                        </div>

                        {/* Image Size Section */}
                        <div className="w-1/4 text-center">
                            <h2 className="text-lg text-right mr-4 font-bold text-gray-900 dark:text-white">Size</h2>
                            <p className="text-md text-right mr-4 text-gray-600 dark:text-gray-300">
                                {imagePreview ? "W: " + 600 + "px L: " + 400 + "px" : "Not Available"}
                            </p>
                        </div>
                    </div>

                </div>
            </div>
        </section>
    );
};

export default Results;
