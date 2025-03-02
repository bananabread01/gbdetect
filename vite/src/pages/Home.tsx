import { Button } from '@/components/ui/button'

const Home = () => {
    return (
        <section className='flex flex-col items-center justify-center pt-40'>
            <h1 className='text-7xl font-bold tracking-tighter'>GBC Detection</h1>
            <div className='space-x-4 my-4'>
                <Button>
                    App
                </Button>
                <Button>
                    Instructions
                </Button>
            </div>
        </section>
    )
}

export default Home