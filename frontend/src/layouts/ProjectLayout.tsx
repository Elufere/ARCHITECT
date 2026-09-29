import { FileText, Lightbulb, MessageSquareText } from "lucide-react";
import { NavLink, Outlet, useParams } from "react-router-dom";

import { useWorkspace } from "@/features/discovery/queries";

const tabs = [
  { path: "discovery", label: "Discovery", icon: MessageSquareText },
  { path: "understanding", label: "Understanding", icon: Lightbulb },
  { path: "prd", label: "PRD", icon: FileText },
];

export function ProjectLayout() {
  const { projectId = "" } = useParams();
  const { data: workspace, isLoading, isError } = useWorkspace(projectId);

  if (isLoading) {
    return <div className="p-8 text-sm text-neutral-500">Loading project…</div>;
  }

  if (isError || !workspace) {
    return <div className="p-8 text-sm text-red-600">Project could not be loaded.</div>;
  }

  return (
    <div className="flex h-full min-w-0 flex-col">
      <header className="border-b border-black/8 bg-[#f7f7f5]/95 px-6 pt-5 backdrop-blur">
        <div className="mb-4 flex items-start justify-between gap-6">
          <div className="min-w-0">
            <p className="eyebrow mb-1">Project</p>
            <h1 className="truncate text-xl font-semibold tracking-tight">
              {workspace.project.name}
            </h1>
          </div>
          <div className="rounded-full border border-black/8 bg-white px-3 py-1.5 text-xs text-neutral-500">
            {workspace.project.status.replaceAll("_", " ")}
          </div>
        </div>

        <nav className="flex gap-1">
          {tabs.map(({ path, label, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              className={({ isActive }) =>
                [
                  "relative flex items-center gap-2 rounded-t-xl px-3 py-3 text-sm transition",
                  isActive
                    ? "bg-white font-medium text-neutral-950"
                    : "text-neutral-500 hover:text-neutral-900",
                ].join(" ")
              }
            >
              <Icon size={15} />
              {label}
            </NavLink>
          ))}
        </nav>
      </header>

      <div className="min-h-0 flex-1">
        <Outlet context={workspace} />
      </div>
    </div>
  );
}
