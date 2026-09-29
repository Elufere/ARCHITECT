import { useParams } from "react-router-dom";

import { useWorkspace } from "@/features/discovery/queries";
import { UnderstandingPanel } from "@/features/understanding/UnderstandingPanel";

export function UnderstandingPage() {
  const { projectId = "" } = useParams();
  const { data: workspace } = useWorkspace(projectId);

  if (!workspace) return null;

  return (
    <div className="h-full overflow-y-auto p-5 md:p-8">
      <div className="mx-auto max-w-4xl">
        <UnderstandingPanel workspace={workspace} />
      </div>
    </div>
  );
}
