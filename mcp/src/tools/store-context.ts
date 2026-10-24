import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { backendRequest, toolText } from "../backend.js";

export function registerStoreContext(server: McpServer): void {
  server.registerTool(
    "store_context",
    {
      description: "Extract and persist source-backed context with PNG and JSON artifacts.",
      inputSchema: {
        text: z.string().trim().min(1).max(50_000),
        source: z.string().max(500).optional(),
        metadata: z.record(z.string(), z.union([z.string(), z.number(), z.boolean(), z.null()])).optional(),
      },
    },
    async ({ text, source, metadata }) => {
      const result = await backendRequest<Record<string, unknown>>("/context", {
        method: "POST",
        body: JSON.stringify({ text, source, metadata }),
      });
      return toolText(result);
    },
  );
}
