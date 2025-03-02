import { cn } from "@/lib/utils"
import { useLocation } from "react-router-dom"

export default function MainNav() {
    let location = useLocation()

    return (
        <div className="mr-4 hidden md:flex">
            <a href="/" className="ml-6 mr-6 flex items-center space-x-2">
                <span className="hidden font-bold sm:inline-block">
                    {"GBDetect"}
                </span>
            </a>
            <nav className="flex items-center space-x-6 text-sm font-medium">
                <a
                    href="/"
                    className={cn(
                        "transition-colors hover:text-foreground/80",
                        location.pathname === "/home" ? "text-foreground" : "text-foreground/60"
                    )}
                >
                    Home
                </a>
                <a
                    href="/app"
                    className={cn(
                        "transition-colors hover:text-foreground/80",
                        location.pathname === "/app" ? "text-foreground" : "text-foreground/60"
                    )}
                >
                    App
                </a>
                <a
                    href="/instructions"
                    className={cn(
                        "transition-colors hover:text-foreground/80",
                        location.pathname === "/instructions" ? "text-foreground" : "text-foreground/60"
                    )}
                >
                    Instructions
                </a>
            </nav>
        </div>
    )
}
