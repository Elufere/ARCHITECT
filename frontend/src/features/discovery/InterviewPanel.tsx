import {
  ArrowUp,
  Lightbulb,
  LoaderCircle,
  PauseCircle,
  Sparkles,
  WandSparkles,
} from "lucide-react";
import { FormEvent, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import {
  useRetryDiscovery,
  useSendDiscoveryTurn,
} from "@/features/discovery/queries";
import type { WorkspaceSnapshot } from "@/types/workspace";

interface Props {
  workspace: WorkspaceSnapshot;
}

export function InterviewPanel({ workspace }: Props) {
  const [draft, setDraft] = useState("");
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const sendTurn = useSendDiscoveryTurn(workspace.project.id);
  const retryDiscovery = useRetryDiscovery(workspace.project.id);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [workspace.discovery.messages.length]);

  function submit(event: FormEvent) {
    event.preventDefault();
    const message = draft.trim();
    if (!message) return;
    setDraft("");
    sendTurn.mutate({ type: "answer", message });
  }

  const processingError = sendTurn.isError || retryDiscovery.isError;
  const disabled =
    sendTurn.isPending || retryDiscovery.isPending || processingError;

  return (
    <section className="flex min-h-0 flex-col bg-white">
      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-7 md:px-10">
        <div className="mx-auto max-w-3xl">
          <div className="mb-8 flex items-center justify-between gap-4">
            <div>
              <p className="eyebrow mb-1">Product interview</p>
              <h2 className="text-lg font-semibold">Discovery</h2>
            </div>
            <div className="flex items-center gap-2 text-xs text-neutral-400">
              <span className="status-dot bg-emerald-500" />
              Saved
            </div>
          </div>

          <div className="space-y-7">
            {workspace.discovery.messages.map((message) => (
              <article
                key={message.id}
                className={
                  message.role === "architect"
                    ? "max-w-2xl"
                    : "ml-auto max-w-2xl rounded-2xl bg-[#f5f5f2] px-4 py-3"
                }
              >
                <div className="mb-2 flex items-center gap-2">
                  {message.role === "architect" && (
                    <span className="grid h-6 w-6 place-items-center rounded-lg bg-neutral-950 text-white">
                      <Sparkles size={12} />
                    </span>
                  )}
                  <span className="text-xs font-semibold text-neutral-400">
                    {message.role === "architect" ? "Architect" : "You"}
                  </span>
                </div>
                <p className="whitespace-pre-wrap text-[15px] leading-7 text-neutral-800">
                  {message.content}
                </p>
              </article>
            ))}

            {workspace.discovery.status === "ready_for_confirmation" && (
              <div className="surface border-violet-200 bg-violet-50/45 p-5">
                <div className="mb-3 flex items-center gap-2 text-violet-700">
                  <WandSparkles size={17} />
                  <span className="text-sm font-semibold">Ready for your review</span>
                </div>
                <p className="max-w-xl text-sm leading-6 text-neutral-600">
                  Architect believes the important product decisions are sufficiently
                  covered. You decide whether discovery is actually complete.
                </p>
                <div className="mt-4 flex flex-wrap gap-2">
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() => sendTurn.mutate({ type: "confirm_prd" })}
                    className="rounded-xl bg-neutral-950 px-4 py-2.5 text-sm font-medium text-white"
                  >
                    {sendTurn.isPending ? "Generating…" : "Yes, generate PRD"}
                  </button>
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() =>
                      sendTurn.mutate({ type: "continue_discovery" })
                    }
                    className="rounded-xl border border-black/10 bg-white px-4 py-2.5 text-sm font-medium"
                  >
                    There’s more to cover
                  </button>
                </div>
              </div>
            )}

            {(sendTurn.isPending || retryDiscovery.isPending) && (
              <div className="flex items-center gap-2 text-sm text-neutral-400">
                <LoaderCircle size={15} className="animate-spin" />
                Architect is thinking…
              </div>
            )}

            {processingError && (
              <div className="flex items-center justify-between gap-4 rounded-xl bg-red-50 px-3 py-2 text-xs text-red-700">
                <span>
                  Architect could not finish processing the saved turn. Your input
                  is already checkpointed.
                </span>
                <button
                  type="button"
                  disabled={retryDiscovery.isPending}
                  onClick={() => {
                    sendTurn.reset();
                    retryDiscovery.reset();
                    retryDiscovery.mutate();
                  }}
                  className="shrink-0 font-semibold"
                >
                  Retry saved work
                </button>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </div>
      </div>

      {workspace.discovery.status === "active" && (
        <div className="border-t border-black/8 bg-white px-5 py-4 md:px-10">
          <div className="mx-auto max-w-3xl">
            <div className="mb-2 flex flex-wrap gap-2">
              <button
                type="button"
                disabled={disabled}
                onClick={() => sendTurn.mutate({ type: "request_suggestion" })}
                className="inline-flex items-center gap-1.5 rounded-lg border border-black/8 px-2.5 py-1.5 text-xs font-medium text-neutral-600 hover:bg-neutral-50 disabled:opacity-40"
              >
                <Lightbulb size={13} />
                Suggest
              </button>
              <button
                type="button"
                disabled={disabled}
                onClick={() => sendTurn.mutate({ type: "unknown" })}
                className="inline-flex items-center gap-1.5 rounded-lg border border-black/8 px-2.5 py-1.5 text-xs font-medium text-neutral-600 hover:bg-neutral-50 disabled:opacity-40"
              >
                <PauseCircle size={13} />
                I haven’t decided
              </button>
              <button
                type="button"
                disabled={disabled}
                onClick={() => sendTurn.mutate({ type: "defer_design" })}
                className="inline-flex items-center gap-1.5 rounded-lg border border-black/8 px-2.5 py-1.5 text-xs font-medium text-neutral-600 hover:bg-neutral-50 disabled:opacity-40"
              >
                <WandSparkles size={13} />
                Leave to design / engineering
              </button>
            </div>

            <form onSubmit={submit} className="relative">
              <textarea
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }}
                placeholder="Answer Architect…"
                rows={3}
                disabled={disabled}
                className="w-full resize-none rounded-2xl border border-black/10 bg-[#fafaf8] px-4 py-3 pr-14 text-sm leading-6 outline-none transition focus:border-neutral-400 focus:bg-white disabled:opacity-60"
              />
              <button
                type="submit"
                disabled={!draft.trim() || disabled}
                aria-label="Send answer"
                className="absolute bottom-3 right-3 grid h-9 w-9 place-items-center rounded-xl bg-neutral-950 text-white disabled:opacity-30"
              >
                <ArrowUp size={17} />
              </button>
            </form>

          </div>
        </div>
      )}

      {workspace.discovery.status === "complete" && (
        <div className="border-t border-black/8 bg-emerald-50 px-6 py-4 text-sm text-emerald-800">
          Discovery is complete.{" "}
          <Link
            to={`/projects/${workspace.project.id}/prd`}
            className="font-semibold underline"
          >
            View the PRD
          </Link>
          .
        </div>
      )}
    </section>
  );
}
