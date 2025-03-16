import MainNav from "./main-nav"

export function SiteHeader() {
    return (
        <header className="fixed top-0 w-full border-b border-gray-300 bg-white dark:bg-gray-800 backdrop-blur supports-[backdrop-filter]:bg-background/30">
            <div className="flex h-14 items-center">
                <MainNav />
            </div>
        </header>
    )
}

//<header className="fixed top-0 w-full border-b border-border/40 bg-white dark:bg-gray-800">
