
import { Outlet } from "react-router-dom"
import { SiteHeader } from "@/components/ui/SiteHeader"
import { ScrollArea } from "@/components/ui/scroll-area"

export default function AppUI() {
    return (
        <main className="flex flex-col h-full overflow-y-hidden">
            <ScrollArea>
                <SiteHeader />

                <div className="min-h-[calc(100vh-106px)]">
                    <Outlet />
                </div>

            </ScrollArea>
        </main>
    )
}