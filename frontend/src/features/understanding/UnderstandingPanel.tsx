import {
  Check,
  Circle,
  Clock3,
  LockKeyhole,
  Minus,
} from "lucide-react";

import type {
  UnderstandingItem,
  UnderstandingState,
  WorkspaceSnapshot,
} from "@/types/workspace";

interface Props {
  workspace: WorkspaceSnapshot;
  compact?: boolean;
}

const iconByState: Record<UnderstandingState, typeof Check> = {
  confirmed: Check,
  active: Clock3,
  open: Circle,
  blocked: LockKeyhole,
  deferred: Minus,
};

const classByState: Record<UnderstandingState, string> = {
  confirmed: "bg-emerald-50 text-emerald-700",
  active: "bg-amber-50 text-amber-700",
  open: "bg-neutral-100 text-neutral-400",
  blocked: "bg-neutral-100 text-neutral-500",
  deferred: "bg-violet-50 text-violet-600",
};

function UnderstandingRow({ item }: { item: UnderstandingItem }) {
  const Icon = iconByState[item.state];

  return (
    <div className="flex gap-3 py-2.5">
      <span
        className={`mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full ${classByState[item.state]}`}
      >
        <Icon size={11} />
      </span>
      <div className="min-w-0">
        <div className="text-sm font-medium text-neutral-700">{item.label}</div>
        {item.detail && (
          <div className="mt-0.5 text-xs leading-5 text-neutral-400">
            {item.detail}
          </div>
        )}
      </div>
    </div>
  );
}

export function UnderstandingPanel({ workspace, compact = false }: Props) {
  return (
    <div className="h-full overflow-y-auto px-5 py-6">
      <div className="mb-6">
        <p className="eyebrow mb-1">Current understanding</p>
        <h2 className={compact ? "text-base font-semibold" : "text-xl font-semibold"}>
          What Architect knows
        </h2>
        {!compact && (
          <p className="mt-2 max-w-2xl text-sm leading-6 text-neutral-500">
            This is founder-facing product understanding, not the raw internal
            knowledge ledger.
          </p>
        )}
      </div>

      <div className="space-y-6">
        {workspace.understanding.sections.map((section) => (
          <section key={section.id}>
            <div className="mb-1 text-xs font-semibold text-neutral-400">
              {section.title}
            </div>
            <div className={compact ? "" : "surface divide-y divide-black/5 px-4"}>
              {section.items.map((item) => (
                <UnderstandingRow key={item.id} item={item} />
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
