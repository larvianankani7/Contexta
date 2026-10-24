import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { backendRequest, toolText } from "../backend.js";

const maxContextCharacters = 8_000;

export function registerInjectContext(server: McpServer): void {
  server.registerTool(
    "inject_context",
    {
      description: "Return bounded evidence for a query or a small explicit set of memory IDs.",
      inputSchema: {
        query: z.string().trim().min(1).max(5_000).optional(),
        memory_ids: z.array(z.string().uuid()).min(1).max(5).optional(),
      },
    },
    async ({ query, memory_ids }) => {
      if (Boolean(query) === Boolean(memory_ids)) {
        throw new Error("Provide exactly one of query or memory_ids");
      }
      const evidence: Array<Record<string, unknown>> = [];
      if (query) {
        const result = await backendRequest<{ evidence: Array<Record<string, unknown>> }>("/retrieve", {
          method: "POST",
          body: JSON.stringify({ query, limit: 5 }),
        });
        evidence.push(...result.evidence);
      } else if (memory_ids) {
        const records = await Promise.all(
          memory_ids.map(async (id) => {
            const record = await backendRequest<Record<string, unknown>>(
              `/memory/${encodeURIComponent(id)}`,
            );
            return {
              memory_id: id,
              text: Array.isArray(record.facts)
                ? record.facts.map((fact) => String((fact as { text?: unknown }).text ?? "")).join(" ")
                : "",
              source: record.source,
              relevance: 1,
            };
          }),
        );
        evidence.push(...records);
      }
      let remaining = maxContextCharacters;
      const selected = evidence.slice(0, 5);
      let truncated = evidence.length > selected.length;
      const bounded = selected.map((item) => {
        const original = String(item.text);
        const text = original.slice(0, remaining);
        truncated ||= text.length < original.length;
        remaining -= text.length;
        return { ...item, text };
      }).filter((item) => item.text.length > 0);
      return toolText({ evidence: bounded, truncated });
    },
  );
}
