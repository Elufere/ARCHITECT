import { Outlet } from "react-router-dom";

import { ProjectSidebar } from "@/features/projects/ProjectSidebar";

export function AppShell() {
  return (
    <div className="flex h-screen overflow-hidden bg-[#f7f7f5] text-neutral-900">
      <ProjectSidebar />
      <main className="min-w-0 flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
}
