import type {
  ProjectSummary,
  WorkspaceSnapshot,
} from "@/types/workspace";

const now = new Date().toISOString();

export const seedProjects: ProjectSummary[] = [
  {
    id: "escrow-app",
    name: "Escrow App",
    description: "A customer-to-customer escrow transaction product.",
    status: "discovering",
    updatedAt: now,
  },
  {
    id: "tutordesk",
    name: "TutorDesk",
    description: "Workspace for independent tutors.",
    status: "ready_for_prd",
    updatedAt: now,
  },
  {
    id: "queueless",
    name: "QueueLess",
    description: "Digital queue and waitlist management.",
    status: "discovering",
    updatedAt: now,
  },
  {
    id: "sportqi",
    name: "SportQi",
    description: "Sports intelligence and pre-match insight engine.",
    status: "draft",
    updatedAt: now,
  },
];

export const seedWorkspaces: Record<string, WorkspaceSnapshot> = {
  "escrow-app": {
    project: seedProjects[0],
    discovery: {
      status: "active",
      messages: [
        {
          id: "m1",
          role: "founder",
          content:
            "I want to build an escrow app where two customers can act as buyer and seller.",
          createdAt: now,
        },
        {
          id: "m2",
          role: "architect",
          content:
            "I understand that a customer can participate in a deal as either buyer or seller. What should happen when one party disputes a transaction before completion?",
          createdAt: now,
        },
      ],
      activePrompt:
        "What should happen when one party disputes a transaction before completion?",
    },
    understanding: {
      sections: [
        {
          id: "users",
          title: "Users",
          items: [
            {
              id: "u1",
              label: "Customer",
              detail: "Can participate as buyer or seller depending on the transaction.",
              state: "confirmed",
            },
          ],
        },
        {
          id: "workflow",
          title: "Core workflow",
          items: [
            { id: "w1", label: "Deal is created", state: "confirmed" },
            { id: "w2", label: "Invited party accepts", state: "confirmed" },
            { id: "w3", label: "Buyer funds escrow", state: "confirmed" },
            { id: "w4", label: "Seller delivers", state: "confirmed" },
            { id: "w5", label: "Buyer confirms completion", state: "confirmed" },
            { id: "w6", label: "Funds are released", state: "confirmed" },
          ],
        },
        {
          id: "integrations",
          title: "Integrations",
          items: [
            {
              id: "i1",
              label: "Flutterwave",
              detail: "Payment gateway for funding escrow.",
              state: "confirmed",
            },
          ],
        },
        {
          id: "open",
          title: "Open decisions",
          items: [
            {
              id: "o1",
              label: "Dispute resolution",
              state: "active",
            },
            {
              id: "o2",
              label: "MVP scope",
              state: "open",
            },
          ],
        },
      ],
    },
    prd: {
      status: "not_generated",
      sections: [],
    },
  },
  tutordesk: {
    project: seedProjects[1],
    discovery: {
      status: "ready_for_confirmation",
      messages: [
        {
          id: "t1",
          role: "architect",
          content:
            "I think the important product decisions are covered. Do you think anything important is still missing before I generate the PRD?",
          createdAt: now,
        },
      ],
    },
    understanding: {
      sections: [
        {
          id: "users",
          title: "Users",
          items: [
            { id: "tu1", label: "Independent tutor", state: "confirmed" },
          ],
        },
        {
          id: "workflow",
          title: "Core workflow",
          items: [
            { id: "tw1", label: "Manage learners", state: "confirmed" },
            { id: "tw2", label: "Schedule sessions", state: "confirmed" },
          ],
        },
      ],
    },
    prd: { status: "not_generated", sections: [] },
  },
  queueless: {
    project: seedProjects[2],
    discovery: {
      status: "active",
      messages: [
        {
          id: "q1",
          role: "architect",
          content:
            "How should a customer know when it is time to return for service?",
          createdAt: now,
        },
      ],
    },
    understanding: {
      sections: [
        {
          id: "users",
          title: "Users",
          items: [
            { id: "qu1", label: "Customer", state: "confirmed" },
            { id: "qu2", label: "Business staff", state: "confirmed" },
          ],
        },
      ],
    },
    prd: { status: "not_generated", sections: [] },
  },
  sportqi: {
    project: seedProjects[3],
    discovery: {
      status: "active",
      messages: [],
      activePrompt: "Start by describing what SportQi should help users achieve.",
    },
    understanding: {
      sections: [
        {
          id: "open",
          title: "Open decisions",
          items: [{ id: "so1", label: "Primary user", state: "open" }],
        },
      ],
    },
    prd: { status: "not_generated", sections: [] },
  },
};
