import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import homeImage from "@/assets/home-page.jpg";


const Home = () => {
    return (
        <section className="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-gray-900">
            {/* Left Side*/}
            <div className="w-1/2 flex flex-col items-start px-12">
                <h1 className="text-6xl font-bold tracking-tight text-gray-900 dark:text-white leading-tight">
                    Gallbladder Cancer Detection
                </h1>
                <p className="text-lg text-gray-600 dark:text-gray-300 mt-4">
                    AI-powered Ultrasound Scan analysis for early Gallbladder Cancer detection.
                </p>
                <div className="mt-6 flex space-x-4">
                    <Button asChild>
                        <Link to="/upload">Predict</Link>
                    </Button>
                    <Button variant="outline">
                        <Link to="/instructions">Instructions</Link>
                    </Button>
                </div>
            </div>

            {/* Right Side*/}
            <div className="w-1/2 flex justify-center">
                <img 
                    src={homeImage}
                    alt="Ultrasound Illustration" 
                    className=" rounded-lg shadow-lg mr-20"
                />
            </div>
        </section>
    );
};

export default Home;