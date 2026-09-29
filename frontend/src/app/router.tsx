import { Navigate, createBrowserRouter } from "react-router-dom";

import { AppShell } from "@/layouts/AppShell";
import { ProjectLayout } from "@/layouts/ProjectLayout";
import { ProjectsPage } from "@/features/projects/ProjectsPage";
import { NewProjectPage } from "@/features/projects/NewProjectPage";
import { DiscoveryPage } from "@/features/discovery/DiscoveryPage";
import { UnderstandingPage } from "@/features/understanding/UnderstandingPage";
import { PrdPage } from "@/features/prd/PrdPage";

export const router = createBrowserRouter([
  {
    element: <AppShell />,
    children: [
      { index: true, element: <ProjectsPage /> },
      { path: "projects/new", element: <NewProjectPage /> },
      {
        path: "projects/:projectId",
        element: <ProjectLayout />,
        children: [
          { index: true, element: <Navigate to="discovery" replace /> },
          { path: "discovery", element: <DiscoveryPage /> },
          { path: "understanding", element: <UnderstandingPage /> },
          { path: "prd", element: <PrdPage /> },
        ],
      },
    ],
  },
]);
