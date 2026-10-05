import {
  Link,
  Navigate,
  isRouteErrorResponse,
  useRouteError,
  createBrowserRouter,
} from "react-router-dom";

import { AppShell } from "@/layouts/AppShell";
import { ProjectLayout } from "@/layouts/ProjectLayout";
import { ProjectsPage } from "@/features/projects/ProjectsPage";
import { NewProjectPage } from "@/features/projects/NewProjectPage";
import { DiscoveryPage } from "@/features/discovery/DiscoveryPage";
import { UnderstandingPage } from "@/features/understanding/UnderstandingPage";
import { PrdPage } from "@/features/prd/PrdPage";

function RouteErrorPage() {
  const error = useRouteError();
  const status = isRouteErrorResponse(error) ? error.status : 500;
  const title = status === 404 ? "Page not found" : "Something went wrong";
  const message =
    status === 404
      ? "The page you opened does not exist or may have been removed."
      : error instanceof Error
        ? error.message
        : "Architect could not load this page.";

  return (
    <div className="grid h-full place-items-center p-8">
      <div className="surface max-w-md p-7 text-center">
        <p className="eyebrow mb-2">{status}</p>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        <p className="mt-2 text-sm leading-6 text-neutral-500">{message}</p>
        <Link
          to="/"
          className="mt-5 inline-flex rounded-lg bg-neutral-950 px-4 py-2 text-sm font-medium text-white"
        >
          Back to projects
        </Link>
      </div>
    </div>
  );
}

function NotFoundPage() {
  return (
    <div className="grid h-full place-items-center p-8">
      <div className="surface max-w-md p-7 text-center">
        <p className="eyebrow mb-2">404</p>
        <h1 className="text-xl font-semibold tracking-tight">Page not found</h1>
        <p className="mt-2 text-sm leading-6 text-neutral-500">
          The page you opened does not exist or may have been removed.
        </p>
        <Link
          to="/"
          className="mt-5 inline-flex rounded-lg bg-neutral-950 px-4 py-2 text-sm font-medium text-white"
        >
          Back to projects
        </Link>
      </div>
    </div>
  );
}

export const router = createBrowserRouter([
  {
    element: <AppShell />,
    errorElement: <RouteErrorPage />,
    children: [
      { index: true, element: <ProjectsPage /> },
      { path: "projects", element: <Navigate to="/" replace /> },
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
      { path: "*", element: <NotFoundPage /> },
    ],
  },
]);
