
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { backendRequest, toolText } from "../backend.js";

export function registerObserveMessage(server: McpServer): void {
  server.registerTool(
    "observe_message",
    {
      description:
        "Inspect a USER-authored message for durable personal or project decisions. " +
        "Call when a user states a confirmed project choice, requirement, preference, " +
        "or uses remember/save/lock language, even without an explicit save request. " +
        "Do not use for ordinary knowledge questions, hypothetical options, " +
        "assistant-generated content, or secrets. " +
        "Backend decides whether to persist without model inference.",
      inputSchema: {
        message: z.string().trim().min(1).max(5000),
        project: z.string().trim().min(1).max(100).optional(),
        source: z.string().max(200).optional(),
      },
    },
    async ({ message, project, source }) => {
      const result = await backendRequest<Record<string, unknown>>(
        "/memory/observe",
        {
          method: "POST",
          body: JSON.stringify({
            message,
            project,
            source: source ?? "mcp:user",
          }),
        },
      );
      return toolText(result);
    },
  );
}
