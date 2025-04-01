import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import homeImage from "@/assets/home-page.jpg";


const Home = () => {
    return (
        <section className="min-h-screen flex flex-col md:flex-row items-center justify-center bg-gray-100 dark:bg-gray-900">
            {/* Left Side*/}
            <div className="w-full md:w-1/2 flex flex-col items-start px-6 md:px-12 mb-8 md:mb-0">
                <h1 className="text-4xl md:text-6xl font-bold tracking-tight text-gray-900 dark:text-white leading-tight text-center md:text-left">
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
            <div className="w-full md:w-1/2 flex justify-center px-6">
                <img 
                    src={homeImage}
                    alt="Ultrasound Illustration" 
                    className=" rounded-lg shadow-lg max-w-full h-auto mx-auto  md:ml-[-40px]"
                />
            </div>
        </section>
    );
};

export default Home;