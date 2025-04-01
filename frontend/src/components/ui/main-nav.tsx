import { cn } from "@/lib/utils"
import { Link, useLocation } from "react-router-dom"
import { ModeToggle } from "../mode-toggle"

export default function MainNav() {
    let location = useLocation()

    return (
    <div className="w-full flex items-center justify-between p-4">        
        <div className="flex items-center space-x-6">
        <Link to="/" className="ml-6 mr-6 flex items-center space-x-2">
            <span className="hidden font-bold sm:inline-block">
                {"GBDetect"}
            </span>
        </Link>

        {/* Center Navigation */}
        <nav className="flex items-center space-x-6 text-sm font-medium">
            <Link
                to="/"
                className={cn(
                    "transition-colors hover:text-foreground/80",
                    location.pathname === "/" ? "text-foreground" : "text-foreground/60"
                )}
            >
                Home
                    
            </Link>

            <Link
                to="/upload"
                className={cn(
                    "transition-colors hover:text-foreground/80",
                    location.pathname === "/upload" ? "text-foreground" : "text-foreground/60"
                )}
            >
                Predict
                    
            </Link>

            <Link
                to="/instructions"
                className={cn(
                    "transition-colors hover:text-foreground/80",
                    location.pathname === "/instructions" ? "text-foreground" : "text-foreground/60"
                )}
            >
                Instructions
                    
            </Link>
        </nav>
        </div>
        <div className="flex items-center mr-2">
            <ModeToggle />
        </div>
    </div>
    )
}
