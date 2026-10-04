import { ArrowRight, FolderKanban, Plus, Trash2 } from "lucide-react";
import { Link } from "react-router-dom";

import { useDeleteProject, useProjects } from "@/features/projects/queries";

export function ProjectsPage() {
  const {
    data: projects = [],
    isLoading,
    isError,
    error,
    refetch,
  } = useProjects();
  const deleteProject = useDeleteProject();

  const handleDelete = (projectId: string, projectName: string) => {
    const confirmed = window.confirm(
      `Delete "${projectName}"? This permanently removes its discovery session and debug log.`,
    );
    if (!confirmed) return;
    deleteProject.mutate(projectId);
  };

  return (
    <div className="h-full overflow-y-auto p-8 lg:p-12">
      <div className="mx-auto max-w-5xl">
        <div className="mb-10 flex items-end justify-between gap-4">
          <div>
            <p className="eyebrow mb-2">Workspace</p>
            <h1 className="text-3xl font-semibold tracking-tight">Projects</h1>
            <p className="mt-2 max-w-xl text-sm leading-6 text-neutral-500">
              Each project keeps its discovery interview, product understanding,
              and PRD together.
            </p>
          </div>
          <Link
            to="/projects/new"
            className="inline-flex items-center gap-2 rounded-xl bg-neutral-950 px-4 py-2.5 text-sm font-medium text-white"
          >
            <Plus size={16} />
            New project
          </Link>
        </div>

        {deleteProject.isError && (
          <div className="mb-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {deleteProject.error instanceof Error
              ? deleteProject.error.message
              : "Project could not be deleted."}
          </div>
        )}

        {isLoading ? (
          <div className="surface p-6 text-sm text-neutral-500">Loading projects…</div>
        ) : isError ? (
          <div className="surface p-6">
            <p className="text-sm font-medium text-red-700">
              Could not load projects from Architect.
            </p>
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
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {projects.map((project) => {
              const deleting =
                deleteProject.isPending && deleteProject.variables === project.id;

              return (
                <div
                  key={project.id}
                  className="surface group relative transition hover:-translate-y-0.5 hover:shadow-md"
                >
                  <Link
                    to={`/projects/${project.id}/discovery`}
                    className="block p-5"
                  >
                    <div className="mb-8 flex items-start justify-between pr-9">
                      <span className="grid h-10 w-10 place-items-center rounded-xl bg-neutral-100">
                        <FolderKanban size={18} />
                      </span>
                      <ArrowRight
                        size={17}
                        className="text-neutral-300 transition group-hover:translate-x-0.5 group-hover:text-neutral-700"
                      />
                    </div>
                    <h2 className="font-semibold">{project.name}</h2>
                    <p className="mt-1 line-clamp-2 text-sm leading-6 text-neutral-500">
                      {project.description}
                    </p>
                  </Link>

                  <button
                    type="button"
                    onClick={() => handleDelete(project.id, project.name)}
                    disabled={deleting}
                    className="absolute right-4 top-4 grid h-8 w-8 place-items-center rounded-lg text-neutral-400 transition hover:bg-red-50 hover:text-red-600 disabled:cursor-wait disabled:opacity-50"
                    title={deleting ? "Deleting project" : "Delete project"}
                    aria-label={`Delete ${project.name}`}
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
