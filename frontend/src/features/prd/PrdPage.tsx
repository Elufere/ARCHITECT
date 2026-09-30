import { FileText, LockKeyhole, Sparkles } from "lucide-react";
import { useParams } from "react-router-dom";

import {
  useGeneratePrd,
  useWorkspace,
} from "@/features/discovery/queries";

export function PrdPage() {
  const { projectId = "" } = useParams();
  const { data: workspace } = useWorkspace(projectId);
  const generatePrd = useGeneratePrd(projectId);

  if (!workspace) return null;

  if (workspace.prd.status === "generating") {
    return (
      <div className="grid h-full place-items-center p-8">
        <div className="max-w-md text-center">
          <span className="mx-auto mb-5 grid h-12 w-12 place-items-center rounded-2xl bg-violet-50 text-violet-600">
            <Sparkles size={19} />
          </span>
          <p className="eyebrow mb-2">PRD</p>
          <h2 className="text-xl font-semibold">Generation needs to finish</h2>
          <p className="mt-2 text-sm leading-6 text-neutral-500">
            Founder approval is already saved. Resume the verified compilation
            without replaying the confirmation turn.
          </p>
          {generatePrd.isError && (
            <p className="mt-3 text-sm text-red-700">
              {generatePrd.error instanceof Error
                ? generatePrd.error.message
                : "PRD generation failed."}
            </p>
          )}
          <button
            type="button"
            disabled={generatePrd.isPending}
            onClick={() => generatePrd.mutate()}
            className="mt-5 rounded-xl bg-neutral-950 px-4 py-2.5 text-sm font-medium text-white disabled:opacity-40"
          >
            {generatePrd.isPending ? "Generating…" : "Resume PRD generation"}
          </button>
        </div>
      </div>
    );
  }

  if (workspace.prd.status !== "ready") {
    return (
      <div className="grid h-full place-items-center p-8">
        <div className="max-w-md text-center">
          <span className="mx-auto mb-5 grid h-12 w-12 place-items-center rounded-2xl bg-neutral-100 text-neutral-500">
            <LockKeyhole size={19} />
          </span>
          <p className="eyebrow mb-2">PRD</p>
          <h2 className="text-xl font-semibold">Not generated yet</h2>
          <p className="mt-2 text-sm leading-6 text-neutral-500">
            Architect will first decide that discovery is sufficiently covered.
            You still make the final decision to generate the PRD.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6 md:p-10">
      <article className="mx-auto max-w-4xl">
        <div className="mb-10 flex items-start justify-between gap-6">
          <div>
            <div className="mb-3 flex items-center gap-2 text-neutral-400">
              <FileText size={16} />
              <span className="eyebrow">Product requirements document</span>
            </div>
            <h1 className="text-3xl font-semibold tracking-tight">
              {workspace.project.name}
            </h1>
          </div>
          <span className="inline-flex items-center gap-2 rounded-full bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700">
            <Sparkles size={13} />
            Generated
          </span>
        </div>

        <div className="space-y-8">
          {workspace.prd.sections.map((section) => (
            <section key={section.id} className="surface p-6">
              <h2 className="mb-3 text-lg font-semibold">{section.title}</h2>
              <p className="whitespace-pre-wrap text-sm leading-7 text-neutral-600">
                {section.body}
              </p>
            </section>
          ))}
        </div>
      </article>
    </div>
  );
}
