import {
  Boxes,
  ChevronRight,
  FolderKanban,
  Plus,
  Settings,
  Sparkles,
} from "lucide-react";
import { NavLink, useLocation } from "react-router-dom";

import { useProjects } from "@/features/projects/queries";
import type { ProjectStatus } from "@/types/workspace";

const statusLabel: Record<ProjectStatus, string> = {
  draft: "Draft",
  discovering: "Discovery",
  ready_for_prd: "Ready",
  prd_ready: "PRD",
};

const statusClass: Record<ProjectStatus, string> = {
  draft: "bg-neutral-300",
  discovering: "bg-amber-400",
  ready_for_prd: "bg-violet-500",
  prd_ready: "bg-emerald-500",
};

export function ProjectSidebar() {
  const { data: projects = [] } = useProjects();
  const location = useLocation();

  return (
    <aside className="flex w-[248px] shrink-0 flex-col border-r border-black/8 bg-[#efefec] p-3">
      <NavLink
        to="/"
        className="mb-4 flex items-center gap-2 px-2 py-2 text-sm font-semibold"
      >
        <span className="grid h-8 w-8 place-items-center rounded-xl bg-neutral-950 text-white">
          <Boxes size={16} />
        </span>
        Architect
      </NavLink>

      <NavLink
        to="/projects/new"
        className="mb-4 flex items-center justify-between rounded-xl bg-neutral-950 px-3 py-2.5 text-sm font-medium text-white transition hover:bg-neutral-800"
      >
        <span className="flex items-center gap-2">
          <Plus size={16} />
          New project
        </span>
        <ChevronRight size={14} className="opacity-60" />
      </NavLink>

      <div className="mb-2 flex items-center justify-between px-2">
        <span className="eyebrow">Projects</span>
        <FolderKanban size={14} className="text-neutral-400" />
      </div>

      <nav className="min-h-0 flex-1 space-y-1 overflow-y-auto">
        {projects.map((project) => {
          const active = location.pathname.startsWith(`/projects/${project.id}`);

          return (
            <NavLink
              key={project.id}
              to={`/projects/${project.id}/discovery`}
              className={[
                "block rounded-xl px-3 py-2.5 transition",
                active
                  ? "bg-white shadow-sm ring-1 ring-black/5"
                  : "text-neutral-600 hover:bg-white/65 hover:text-neutral-950",
              ].join(" ")}
            >
              <div className="flex items-center justify-between gap-3">
                <span className="truncate text-sm font-medium">{project.name}</span>
                <span
                  className={`status-dot shrink-0 ${statusClass[project.status]}`}
                />
              </div>
              <div className="mt-1 text-[11px] text-neutral-400">
                {statusLabel[project.status]}
              </div>
            </NavLink>
          );
        })}
      </nav>

      <div className="mt-3 border-t border-black/8 pt-3">
        <button
          type="button"
          className="flex w-full items-center gap-2 rounded-xl px-3 py-2 text-sm text-neutral-500 transition hover:bg-white/60 hover:text-neutral-900"
        >
          <Settings size={15} />
          Settings
        </button>
        <div className="mt-2 flex items-center gap-2 px-3 py-2 text-xs text-neutral-400">
          <Sparkles size={13} />
          Product discovery workspace
        </div>
      </div>
    </aside>
  );
}
