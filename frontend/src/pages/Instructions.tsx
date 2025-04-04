import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

const Instructions = () => {
    return (
        <section className="flex flex-col items-center justify-center bg-gray-100 dark:bg-gray-900 pt-40 px-6">
            <h1 className="text-7xl font-bold tracking-tighter">Instructions</h1>

            <ol className="text-lg space-y-6 mt-16 mb-12 text-left max-w-3xl">
                <li>
                    Begin by navigating to the <strong>Predict</strong> page.
                </li>

                <li>
                    <strong>1. Upload an Ultrasound Image:</strong> Click the <em>"Analyze"</em> button and select an abdominal ultrasound image of the gallbladder from your device.
                </li>

                <li>
                    <strong>2. AI-Powered Analysis:</strong> The model will classify the image as:
                    <ul className="list-disc list-inside ml-4">
                        <li><span className="font-semibold">Normal</span></li>
                        <li><span className="text-yellow-600 font-semibold">Benign Lesion</span> (e.g., gallstones, polyps)</li>
                        <li><span className="text-red-600 font-semibold">Malignant Suspicion</span></li>
                    </ul>
                    It will also return a <strong>confidence score</strong> (in %) to indicate the certainty of the prediction made.
                </li>

                <li>
                    <strong>3. Interpretability Support:</strong> Visual explanation tools help you understand the model's prediction:
                    <ul className="list-disc list-inside ml-4">
                        <li><strong>Attention Heatmap</strong>: Focus regions the model used to decide.</li>
                        <li><strong>Grad-CAM</strong>: Gradient-based explanation.</li>
                        <li><strong>Grad-CAM++</strong>: Highlights finer details and smaller regions.</li>
                    </ul>
                    <li>
                    The heatmaps uses color gradients to highlight different levels of importance. 
                    Red areas, or "hot" area, indicate where attention is highest or regions of interest the model focused on, 
                    while blue zones or "cool" regions are irrelevant to the model's decision.
                    </li>
                    Select each heatmap from the sidebar.
                </li>

                <li>
                    <strong>4. Clinical Protocol Box:</strong> Based on the model's prediction, a protocol will appear in the sidebar suggesting possible next steps.
                    These suggestions are based on common clinical workflows and serve as decision support:
                    <ul className="list-disc list-inside ml-4">
                        <li><strong>Normal:</strong> Routine reporting.</li>
                        <li><strong>Benign:</strong> Confirm lesion type, consider follow-up.</li>
                        <li><strong>Malignant:</strong> Second read, recommend CT/MRI, escalate as needed.</li>
                    </ul>
                    Always apply your own clinical judgment.
                </li>

                <li>
                    <strong>5. Zoom and Pan:</strong> Use zoom buttons to inspect heatmaps in detail. When zoomed in, click and drag to pan across the image.
                </li>

                <li>
                    <strong>6. New Session:</strong> Use the <em>"New Session"</em> button to reset and upload a new scan.
                </li>

                <li className="text-gray-700 dark:text-gray-300">
                    ⚠️ <strong>Disclaimer:</strong> This application is a diagnostic aid. It does not replace radiological expertise. All predictions should be interpreted in the full clinical context.
                </li>
            </ol>

            <div className="space-x-4 my-4">
                <Button asChild>
                    <Link to="/upload">Predict</Link>
                </Button>
            </div>
        </section>
    );
};

export default Instructions;
