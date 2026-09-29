import { useParams } from "react-router-dom";

import { InterviewPanel } from "@/features/discovery/InterviewPanel";
import { useWorkspace } from "@/features/discovery/queries";
import { UnderstandingPanel } from "@/features/understanding/UnderstandingPanel";

export function DiscoveryPage() {
  const { projectId = "" } = useParams();
  const { data: workspace } = useWorkspace(projectId);

  if (!workspace) return null;

  return (
    <div className="grid h-full min-h-0 grid-cols-1 xl:grid-cols-[minmax(0,1fr)_360px]">
      <InterviewPanel workspace={workspace} />
      <aside className="hidden min-h-0 border-l border-black/8 bg-[#fafaf8] xl:block">
        <UnderstandingPanel workspace={workspace} compact />
      </aside>
    </div>
  );
}
