import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { workspaceApi } from "@/lib/workspaceApi";
import type { CreateProjectInput } from "@/types/workspace";

export function useProjects() {
  return useQuery({
    queryKey: ["projects"],
    queryFn: workspaceApi.listProjects,
  });
}

export function useCreateProject() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (input: CreateProjectInput) => workspaceApi.createProject(input),
    onSuccess: (workspace) => {
      queryClient.setQueryData(["workspace", workspace.project.id], workspace);
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
  });
}
