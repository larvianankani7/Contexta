import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { backendRequest, toolText } from "../backend.js";

export function registerVerifyContext(server: McpServer): void {
  server.registerTool(
    "verify_context",
    {
      description: "Check whether a claim is supported by the referenced stored memories.",
      inputSchema: {
        claim: z.string().trim().min(1).max(2_000),
        evidence_ids: z.array(z.string().uuid()).min(1).max(20),
      },
    },
    async ({ claim, evidence_ids }) => {
      const result = await backendRequest<Record<string, unknown>>("/verify", {
        method: "POST",
        body: JSON.stringify({ claim, evidence_ids }),
      });
      return toolText(result);
    },
  );
}
