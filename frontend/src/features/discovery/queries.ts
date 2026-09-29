import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { workspaceApi } from "@/lib/workspaceApi";
import type { DiscoveryTurnInput, WorkspaceSnapshot } from "@/types/workspace";

export function useWorkspace(projectId: string) {
  return useQuery({
    queryKey: ["workspace", projectId],
    queryFn: () => workspaceApi.getWorkspace(projectId),
  });
}

function useWorkspaceMutation(
  projectId: string,
  mutationFn: () => Promise<WorkspaceSnapshot>,
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn,
    onSuccess: (workspace) => {
      queryClient.setQueryData(["workspace", projectId], workspace);
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
  });
}

export function useSendDiscoveryTurn(projectId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (input: DiscoveryTurnInput) =>
      workspaceApi.sendTurn(projectId, input),
    onSuccess: (workspace) => {
      queryClient.setQueryData(["workspace", projectId], workspace);
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
  });
}

export function useGeneratePrd(projectId: string) {
  return useWorkspaceMutation(projectId, () => workspaceApi.generatePrd(projectId));
}
