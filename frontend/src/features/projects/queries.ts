import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { workspaceApi } from "@/lib/workspaceApi";
import type { CreateProjectInput, ProjectSummary } from "@/types/workspace";

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
    onError: () => {
      // Project creation persists metadata/session before initial discovery runs.
      // Refresh so a saved project is visible even when that first model call fails.
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
  });
}



export function useDeleteProject() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (projectId: string) => workspaceApi.deleteProject(projectId),
    onSuccess: (_result, projectId) => {
      queryClient.removeQueries({ queryKey: ["workspace", projectId], exact: true });
      queryClient.setQueryData(
        ["projects"],
        (current: ProjectSummary[] | undefined) =>
          current?.filter((project) => project.id !== projectId) ?? [],
      );
      void queryClient.invalidateQueries({ queryKey: ["projects"] });
    },
  });
}
