import MainNav from "./main-nav"

export function SiteHeader() {
    return (
        <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/30">
            <div className="container flex h-14 max-w-screen-2xl items-center">
                <MainNav />
            </div>
        </header>
    )
}