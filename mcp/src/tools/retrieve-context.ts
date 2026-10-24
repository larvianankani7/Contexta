import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { backendRequest, toolText } from "../backend.js";

export function registerRetrieveContext(server: McpServer): void {
  server.registerTool(
    "retrieve_context",
    {
      description: "Retrieve a bounded set of ranked, provenance-backed evidence records.",
      inputSchema: {
        query: z.string().trim().min(1).max(5_000),
        limit: z.number().int().min(1).max(10).optional(),
      },
    },
    async ({ query, limit }) => {
      const result = await backendRequest<{ query: string; evidence: Array<Record<string, unknown>> }>(
        "/retrieve",
        { method: "POST", body: JSON.stringify({ query, limit: limit ?? 5 }) },
      );
      return toolText({
        ...result,
        evidence: result.evidence.map((item) => ({
          ...item,
          text: String(item.text).slice(0, 2_000),
        })),
      });
    },
  );
}
