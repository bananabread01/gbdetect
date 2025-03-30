import { cn } from "@/lib/utils"
import { useLocation } from "react-router-dom"
import { ModeToggle } from "../mode-toggle"

export default function MainNav() {
    let location = useLocation()

    return (
    <div className="w-full flex items-center justify-between p-4">        
        <div className="flex items-center space-x-6">
        <a href="/" className="ml-6 mr-6 flex items-center space-x-2">
            <span className="hidden font-bold sm:inline-block">
                {"GBDetect"}
            </span>
        </a>
        {/* Center Navigation */}
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
            {/* <a
                href="/app"
                className={cn(
                    "transition-colors hover:text-foreground/80",
                    location.pathname === "/app" ? "text-foreground" : "text-foreground/60"
                )}
            >
                App
            </a> */}
            <a
                href="/upload"
                className={cn(
                    "transition-colors hover:text-foreground/80",
                    location.pathname === "/upload" ? "text-foreground" : "text-foreground/60"
                )}
            >
                Predict
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
        <div className="flex items-center mr-2">
            <ModeToggle />
        </div>
    </div>
    )
}
