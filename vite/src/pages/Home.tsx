import { Link } from "react-router-dom";
import { Button } from '@/components/ui/button'

const Home = () => {
    return (
        <section className='flex flex-col items-center justify-center pt-40'>
            <h1 className='text-7xl font-bold tracking-tighter'>GBC Detection</h1>
            <div className='space-x-4 my-4'>
                <Button asChild>
                    <Link to="/app">Predict</Link>
                </Button>
                <Button>
                <Link to="/instructions">Instructions</Link>
                </Button>
            </div>
        </section>
    )
}

export default Home