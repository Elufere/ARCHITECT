import { ArrowLeft, ArrowRight, Sparkles } from "lucide-react";
import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useCreateProject } from "@/features/projects/queries";
import { ArchitectApiError } from "@/lib/workspaceApi";

export function NewProjectPage() {
  const navigate = useNavigate();
  const createProject = useCreateProject();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [savedProjectId, setSavedProjectId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!name.trim() || !description.trim()) return;

    setSubmitError(null);
    setSavedProjectId(null);

    try {
      const workspace = await createProject.mutateAsync({ name, description });
      navigate(`/projects/${workspace.project.id}/discovery`);
    } catch (error) {
      if (error instanceof ArchitectApiError && error.projectId) {
        setSavedProjectId(error.projectId);
        setSubmitError(
          `${error.message} The project and founder idea were saved and can be resumed.`,
        );
        return;
      }

      setSubmitError(
        error instanceof Error ? error.message : "Project creation failed.",
      );
    }
  }

  return (
    <div className="h-full overflow-y-auto p-8 lg:p-12">
      <div className="mx-auto max-w-2xl">
        <Link
          to="/"
          className="mb-8 inline-flex items-center gap-2 text-sm text-neutral-500 hover:text-neutral-900"
        >
          <ArrowLeft size={15} />
          Back to projects
        </Link>

        <div className="mb-8">
          <span className="mb-4 grid h-11 w-11 place-items-center rounded-2xl bg-neutral-950 text-white">
            <Sparkles size={18} />
          </span>
          <p className="eyebrow mb-2">New project</p>
          <h1 className="text-3xl font-semibold tracking-tight">
            What are you building?
          </h1>
          <p className="mt-3 text-sm leading-6 text-neutral-500">
            Give Architect enough context to begin discovery. The description becomes
            the first founder message in the project.
          </p>
        </div>

        <form onSubmit={submit} className="surface space-y-6 p-6">
          <label className="block">
            <span className="mb-2 block text-sm font-medium">Project name</span>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="e.g. Escrow App"
              className="w-full rounded-xl border border-black/10 bg-neutral-50 px-3.5 py-3 text-sm outline-none transition focus:border-neutral-400 focus:bg-white"
            />
          </label>

          <label className="block">
            <span className="mb-2 block text-sm font-medium">Product idea</span>
            <textarea
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="Describe the product, who it is for, and the main outcome you want…"
              rows={7}
              className="w-full resize-none rounded-xl border border-black/10 bg-neutral-50 px-3.5 py-3 text-sm leading-6 outline-none transition focus:border-neutral-400 focus:bg-white"
            />
          </label>

          {submitError && (
            <div className="rounded-xl bg-red-50 p-3 text-sm text-red-700">
              <p>{submitError}</p>
              {savedProjectId && (
                <Link
                  to={`/projects/${savedProjectId}/discovery`}
                  className="mt-2 inline-block font-semibold underline"
                >
                  Open saved project
                </Link>
              )}
            </div>
          )}

          <button
            type="submit"
            disabled={
              !name.trim() || !description.trim() || createProject.isPending
            }
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-neutral-950 px-4 py-3 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
          >
            {createProject.isPending ? "Creating…" : "Start discovery"}
            {!createProject.isPending && <ArrowRight size={16} />}
          </button>
        </form>
      </div>
    </div>
  );
}
