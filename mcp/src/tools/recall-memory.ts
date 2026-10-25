
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { backendRequest, toolText } from "../backend.js";

export function registerRecallMemory(server: McpServer): void {
  server.registerTool(
    "recall_memory",
    {
      description:
        "Retrieve persistent, source-backed project decisions across chats " +
        "and restarts. Call BEFORE answering what a user previously chose, " +
        "configured, decided, saved or locked. Supply the project name " +
        "when known. Never invent missing values; present source IDs " +
        "and disclose missing or conflicting decisions.",
      inputSchema: {
        query: z.string().trim().min(1).max(5000),
        project: z.string().trim().min(1).max(100).optional(),
      },
    },
    async ({ query, project }) => {
      const result = await backendRequest<Record<string, unknown>>(
        "/memory/recall",
        {
          method: "POST",
          body: JSON.stringify({ query, project }),
        },
      );
      return toolText(result);
    },
  );
}
