import { Link } from "react-router-dom";
import { Button } from '@/components/ui/button'

const Instructions = () => {
    return (
        <section className='flex flex-col items-center justify-center pt-40'>
            <h1 className='text-7xl font-bold tracking-tighter'>Instructions</h1>

            <ol className="text-lg space-y-4 mt-16 mb-12 text-left">
            <li>
                    Get your predictions by navigating to the <strong>Predict</strong> page. To use the application:
                </li>
                <li>
                    <strong>1. Upload an Ultrasound Image:</strong> Click on the "Analyze" button and select an ultrasound image of the gallbladder from your device.
                </li>
                <li>
                    <strong>2. Analyze the Image:</strong> Once uploaded, the AI model will process the image and provide an analysis. This may take a few seconds.
                </li>
                <li>
                    <strong>3. View Results:</strong> The system will display whether the image indicates a <span className="font-semibold">normal</span>, <span className="text-yellow-600 font-semibold">benign</span>, or <span className="text-red-600 font-semibold">malignant</span> case.
                </li>
                <li>
                    <strong>4. View the Explainability Feature:</strong> View highlighted regions of interest in the ultrasound image from the heatmaps displayed.
                </li>
                <li>
                    <strong>5. Interpret the Results:</strong> Consult a radiologist to confirm the AI-generated results before making any clinical decisions.
                </li>
            </ol>

            <div className='space-x-4 my-4'>
                <Button asChild>
                    <Link to="/app">Predict</Link>
                </Button>
            </div>

        </section>
    )
}

export default Instructions