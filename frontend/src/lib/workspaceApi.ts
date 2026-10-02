import { seedProjects, seedWorkspaces } from "@/data/mock";
import type {
  CreateProjectInput,
  DiscoveryTurnInput,
  ProjectSummary,
  WorkspaceSnapshot,
} from "@/types/workspace";

const STORAGE_KEY = "architect.frontend.mock.v1";
const API_URL = ((import.meta.env.VITE_API_URL as string | undefined)?.trim() ?? "")
  .replace(/\/+$/, "");
const useMockApi = (import.meta.env.VITE_USE_MOCK_API ?? "false") === "true";

export class ArchitectApiError extends Error {
  readonly status: number;
  readonly projectId?: string;
  readonly retryable?: boolean;

  constructor(
    message: string,
    options: {
      status: number;
      projectId?: string;
      retryable?: boolean;
    },
  ) {
    super(message);
    this.name = "ArchitectApiError";
    this.status = options.status;
    this.projectId = options.projectId;
    this.retryable = options.retryable;
  }
}

type Store = {
  projects: ProjectSummary[];
  workspaces: Record<string, WorkspaceSnapshot>;
};

function cloneSeed(): Store {
  return JSON.parse(
    JSON.stringify({
      projects: seedProjects,
      workspaces: seedWorkspaces,
    }),
  ) as Store;
}

function loadStore(): Store {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    const seeded = cloneSeed();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(seeded));
    return seeded;
  }

  try {
    return JSON.parse(raw) as Store;
  } catch {
    const seeded = cloneSeed();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(seeded));
    return seeded;
  }
}

function saveStore(store: Store) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
}

function uid(prefix: string) {
  return `${prefix}-${crypto.randomUUID()}`;
}

function wait(ms = 250) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...init?.headers,
      },
    });
  } catch {
    throw new ArchitectApiError(
      "Could not reach the Architect API. Make sure the backend is running.",
      { status: 0, retryable: true },
    );
  }

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = body?.detail;
    const detailObject =
      detail && typeof detail === "object" ? detail : undefined;
    const message =
      (typeof detail === "string" ? detail : detailObject?.message) ??
      body?.message ??
      `Architect API request failed (${response.status})`;

    throw new ArchitectApiError(message, {
      status: response.status,
      projectId:
        typeof detailObject?.projectId === "string"
          ? detailObject.projectId
          : undefined,
      retryable:
        typeof detailObject?.retryable === "boolean"
          ? detailObject.retryable
          : response.status >= 500,
    });
  }

  return body as T;
}

const mockApi = {
  async listProjects() {
    await wait(100);
    return loadStore().projects;
  },

  async createProject(input: CreateProjectInput) {
    await wait();
    const store = loadStore();
    const id = `${input.name
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/(^-|-$)/g, "") || "project"}-${Date.now().toString(36)}`;

    const project: ProjectSummary = {
      id,
      name: input.name.trim(),
      description: input.description.trim(),
      status: "discovering",
      updatedAt: new Date().toISOString(),
    };

    const workspace: WorkspaceSnapshot = {
      project,
      discovery: {
        status: "active",
        messages: [
          {
            id: uid("message"),
            role: "founder",
            content: input.description.trim(),
            createdAt: new Date().toISOString(),
          },
          {
            id: uid("message"),
            role: "architect",
            content:
              "I have the starting idea. Who are the main people this product should serve?",
            createdAt: new Date().toISOString(),
          },
        ],
        activePrompt: "Who are the main people this product should serve?",
      },
      understanding: {
        sections: [
          {
            id: "open",
            title: "Open decisions",
            items: [{ id: uid("open"), label: "Primary users", state: "active" }],
          },
        ],
      },
      prd: { status: "not_generated", sections: [] },
    };

    store.projects.unshift(project);
    store.workspaces[id] = workspace;
    saveStore(store);
    return workspace;
  },

  async getWorkspace(projectId: string) {
    await wait(120);
    const workspace = loadStore().workspaces[projectId];
    if (!workspace) throw new Error("Project not found");
    return workspace;
  },

  async sendTurn(projectId: string, input: DiscoveryTurnInput) {
    await wait(350);
    const store = loadStore();
    const workspace = store.workspaces[projectId];
    if (!workspace) throw new Error("Project not found");

    const now = new Date().toISOString();
    let founderText = "";
    let architectText = "";

    if (input.type === "answer") {
      founderText = input.message;
      architectText =
        "Got it. I’ll treat that as part of the product decision. What is the next important rule or outcome you want us to settle?";
    } else if (input.type === "request_suggestion") {
      founderText = "What do you suggest?";
      architectText =
        "A sensible product-level default is to keep the rule simple, explicit, and reversible where money has not moved yet. You can accept that direction or tell me what should work differently.";
    } else if (input.type === "unknown") {
      founderText = "I haven’t decided yet.";
      architectText =
        "That’s fine. I’ll keep it as an open decision rather than forcing an answer, and move on to the next material part of the product.";
    } else if (input.type === "defer_design") {
      founderText = "Leave this to design or engineering.";
      architectText =
        "Understood. I’ll treat the interaction or implementation detail as deferred and stay at the product-decision level.";
    } else if (input.type === "confirm_prd") {
      founderText = "Yes, generate the PRD.";
      architectText = "PRD generated from the confirmed discovery decisions.";
      workspace.discovery.status = "complete";
      workspace.project.status = "prd_ready";
      workspace.prd = {
        status: "ready",
        sections: [
          {
            id: "overview",
            title: "Overview",
            body: `${workspace.project.name} is defined from the confirmed discovery decisions.`,
          },
        ],
      };
    } else {
      founderText = "There is more I want to cover.";
      architectText =
        "Sure. What product decision or area do you want to add or revisit?";
      workspace.discovery.status = "active";
      workspace.project.status = "discovering";
    }

    workspace.discovery.messages.push(
      {
        id: uid("message"),
        role: "founder",
        content: founderText,
        createdAt: now,
      },
      {
        id: uid("message"),
        role: "architect",
        content: architectText,
        createdAt: now,
      },
    );

    workspace.project.updatedAt = now;
    store.projects = store.projects.map((project) =>
      project.id === projectId ? workspace.project : project,
    );

    saveStore(store);
    return workspace;
  },

  async retryDiscovery(projectId: string) {
    await wait(120);
    const workspace = loadStore().workspaces[projectId];
    if (!workspace) throw new Error("Project not found");
    return workspace;
  },

  async generatePrd(projectId: string) {
    await wait(500);
    const store = loadStore();
    const workspace = store.workspaces[projectId];
    if (!workspace) throw new Error("Project not found");

    workspace.discovery.status = "complete";
    workspace.project.status = "prd_ready";
    workspace.prd = {
      status: "ready",
      sections: [
        {
          id: "overview",
          title: "Overview",
          body: `${workspace.project.name} is being defined through Architect's structured product discovery workflow.`,
        },
        {
          id: "users",
          title: "Users and roles",
          body:
            "This section will be generated from confirmed discovery knowledge once the live Architect API is connected.",
        },
        {
          id: "workflow",
          title: "Core workflow",
          body:
            "The final PRD will compile the confirmed lifecycle, business rules, integrations, constraints, exceptions, and MVP scope.",
        },
      ],
    };

    store.projects = store.projects.map((project) =>
      project.id === projectId ? workspace.project : project,
    );
    saveStore(store);
    return workspace;
  },
};

const liveApi = {
  listProjects: () => request<ProjectSummary[]>("/api/projects"),
  createProject: (input: CreateProjectInput) =>
    request<WorkspaceSnapshot>("/api/projects", {
      method: "POST",
      body: JSON.stringify(input),
    }),
  getWorkspace: (projectId: string) =>
    request<WorkspaceSnapshot>(`/api/projects/${projectId}/workspace`),
  sendTurn: (projectId: string, input: DiscoveryTurnInput) =>
    request<WorkspaceSnapshot>(`/api/projects/${projectId}/discovery/turn`, {
      method: "POST",
      body: JSON.stringify(input),
    }),
  retryDiscovery: (projectId: string) =>
    request<WorkspaceSnapshot>(`/api/projects/${projectId}/discovery/retry`, {
      method: "POST",
    }),
  generatePrd: (projectId: string) =>
    request<WorkspaceSnapshot>(`/api/projects/${projectId}/prd/generate`, {
      method: "POST",
    }),
};

export const workspaceApi = useMockApi ? mockApi : liveApi;

export function projectDebugLogUrl(projectId: string): string | null {
  if (useMockApi) return null;
  return `${API_URL}/api/projects/${encodeURIComponent(projectId)}/debug-log`;
}
