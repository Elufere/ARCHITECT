export type ProjectStatus =
  | "draft"
  | "discovering"
  | "ready_for_prd"
  | "prd_ready";

export type DiscoveryStatus =
  | "active"
  | "ready_for_confirmation"
  | "compiling"
  | "complete";

export type MessageRole = "architect" | "founder";

export interface ProjectSummary {
  id: string;
  name: string;
  description: string;
  status: ProjectStatus;
  updatedAt: string;
}

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  createdAt: string;
}

export type UnderstandingState =
  | "confirmed"
  | "active"
  | "open"
  | "deferred";

export interface UnderstandingItem {
  id: string;
  label: string;
  detail?: string;
  state: UnderstandingState;
}

export interface UnderstandingSection {
  id: string;
  title: string;
  items: UnderstandingItem[];
}

export interface PrdSection {
  id: string;
  title: string;
  body: string;
}

export interface WorkspaceSnapshot {
  project: ProjectSummary;
  discovery: {
    status: DiscoveryStatus;
    messages: Message[];
    activePrompt?: string;
  };
  understanding: {
    sections: UnderstandingSection[];
  };
  prd: {
    status: "not_generated" | "generating" | "ready";
    sections: PrdSection[];
  };
}

export type DiscoveryTurnInput =
  | { type: "answer"; message: string }
  | { type: "request_suggestion" }
  | { type: "unknown" }
  | { type: "defer_design" }
  | { type: "continue_discovery"; message?: string };

export interface CreateProjectInput {
  name: string;
  description: string;
}
