import { Download, FileText, Lightbulb, MessageSquareText } from "lucide-react";
import { NavLink, Outlet, useParams } from "react-router-dom";

import { useWorkspace } from "@/features/discovery/queries";
import { projectDebugLogUrl } from "@/lib/workspaceApi";

const tabs = [
  { path: "discovery", label: "Discovery", icon: MessageSquareText },
  { path: "understanding", label: "Understanding", icon: Lightbulb },
  { path: "prd", label: "PRD", icon: FileText },
];

export function ProjectLayout() {
  const { projectId = "" } = useParams();
  const { data: workspace, isLoading, isError, error, refetch } =
    useWorkspace(projectId);
  const debugLogUrl = projectDebugLogUrl(projectId);

  if (isLoading) {
    return <div className="p-8 text-sm text-neutral-500">Loading project…</div>;
  }

  if (isError || !workspace) {
    return (
      <div className="p-8">
        <p className="text-sm font-medium text-red-700">Project could not be loaded.</p>
        <p className="mt-1 text-sm text-neutral-500">
          {error instanceof Error ? error.message : "The API request failed."}
        </p>
        <button
          type="button"
          onClick={() => void refetch()}
          className="mt-4 rounded-lg bg-neutral-950 px-3 py-2 text-xs font-semibold text-white"
        >
          Try again
        </button>
      </div>
    );
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
          <div className="flex items-center gap-2">
            {debugLogUrl && (
              <a
                href={debugLogUrl}
                className="flex items-center gap-2 rounded-full border border-black/8 bg-white px-3 py-1.5 text-xs font-medium text-neutral-600 transition hover:text-neutral-950"
                title="Download the full terminal-style diagnostic log for this project"
              >
                <Download size={13} />
                Debug log
              </a>
            )}
            <div className="rounded-full border border-black/8 bg-white px-3 py-1.5 text-xs text-neutral-500">
              {workspace.project.status.replaceAll("_", " ")}
            </div>
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
