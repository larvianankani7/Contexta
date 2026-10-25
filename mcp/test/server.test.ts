import { after, before, test } from "node:test";
import assert from "node:assert/strict";
import {
  createServer as createHttpServer,
  type Server as HttpServer,
} from "node:http";
import { resolve } from "node:path";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createServer } from "../src/server.js";

const memoryId = "7c9e6679-7425-40de-944b-e07fc1f90ae7";
const exactId = "19c9e667-7425-40de-944b-e07fc1f90ae7";

let backend: HttpServer;
let backendRequests = 0;
let observed: Record<string, unknown> | null = null;
let recalled: Record<string, unknown> | null = null;

before(async () => {
  backend = createHttpServer(async (request, response) => {
    backendRequests++;

    let body = "";
    for await (const chunk of request) {
      body += chunk.toString();
    }

    const data = body ? JSON.parse(body) : {};

    const send = (status: number, payload: unknown) => {
      response.writeHead(status, {
        "content-type": "application/json",
      });
      response.end(JSON.stringify(payload));
    };

    if (request.url === "/context" && request.method === "POST") {
      send(201, { id: memoryId });
      return;
    }

    if (request.url === "/memory/observe" && request.method === "POST") {
      observed = data;

      const message = String(data.message ?? "");
      const general = /what.s caching/i.test(message);

      send(200, general ? {
        saved: false,
        reason: "question",
        memory_id: null,
        inference_ms: 0,
      } : {
        saved: true,
        reason: "project_decision",
        project: data.project ?? "Job Listing Portal",
        memory_id: memoryId,
        inference_ms: 0,
      });
      return;
    }

    if (request.url === "/memory/recall" && request.method === "POST") {
      recalled = data;
      send(200, {
        status: "source_backed_excerpts",
        project: data.project ?? "Job Listing Portal",
        answer: "Job Listing Portal uses MongoDB",
        evidence: [{
          memory_id: memoryId,
          text: "I'm gonna use MongoDB for my Job Listing Portal",
          source: "mcp:user",
          relevance: 1,
        }],
        inference_ms: 0,
      });
      return;
    }

    if (request.url === `/memory/${memoryId}`) {
      send(200, {
        source: "test",
        facts: [{ text: "x".repeat(10000) }],
      });
      return;
    }

    if (request.url === `/memory/${exactId}`) {
      send(200, {
        source: "test",
        facts: [{ text: "x".repeat(8000) }],
      });
      return;
    }

    if (request.url === "/retrieve") {
      send(200, {
        query: data.query ?? "test",
        evidence: [{
          memory_id: memoryId,
          text: "bounded",
          source: "test",
          relevance: 1,
        }],
      });
      return;
    }

    send(503, { detail: "unavailable" });
  });

  await new Promise<void>((done) =>
    backend.listen(0, "127.0.0.1", done)
  );

  const address = backend.address();
  if (!address || typeof address === "string") {
    throw new Error("Could not start test backend");
  }

  process.env.CONTEXTA_BACKEND_URL =
    `http://127.0.0.1:${address.port}`;
});

after(async () => {
  await new Promise<void>((done, reject) => {
    backend.close((error) =>
      error ? reject(error) : done()
    );
  });
});

async function connectedClient() {
  const server = createServer();
  const client = new Client({
    name: "contexta-test",
    version: "1.0.0",
  });

  const [serverTransport, clientTransport] =
    InMemoryTransport.createLinkedPair();

  await Promise.all([
    server.connect(serverTransport),
    client.connect(clientTransport),
  ]);

  return { client, server };
}

async function callTool(
  name: string,
  args: Record<string, unknown>,
) {
  const { client, server } = await connectedClient();

  try {
    return await client.callTool({
      name,
      arguments: args,
    });
  } finally {
    await client.close();
    await server.close();
  }
}


function toolResult(
  result: Awaited<ReturnType<typeof callTool>>,
): Record<string, any> {
  const content = result.content;

  assert.ok(Array.isArray(content), "Expected MCP content array");

  const item: unknown = content.find(
    (part: unknown) =>
      typeof part === "object" &&
      part !== null &&
      "type" in part &&
      part.type === "text",
  );

  assert.ok(
    typeof item === "object" &&
    item !== null &&
    "text" in item &&
    typeof item.text === "string",
    "Expected text result",
  );

  return JSON.parse(item.text);
}


test("MCP advertises all six Contexta tools", async () => {
  const { client, server } = await connectedClient();

  try {
    const result = await client.listTools();

    assert.deepEqual(
      result.tools.map((tool) => tool.name).sort(),
      [
        "inject_context",
        "observe_message",
        "recall_memory",
        "retrieve_context",
        "store_context",
        "verify_context",
      ],
    );
  } finally {
    await client.close();
    await server.close();
  }
});

test("store_context persists through backend API", async () => {
  const result = await callTool("store_context", {
    text: "Project Alpha uses PostgreSQL.",
    source: "test",
  });

  assert.equal(result.isError, undefined);
  assert.equal(toolResult(result).id, memoryId);
});

test("retrieve_context returns ranked evidence", async () => {
  const result = await callTool("retrieve_context", {
    query: "Project Alpha database",
    limit: 3,
  });

  assert.equal(result.isError, undefined);
  assert.equal(toolResult(result).evidence[0].text, "bounded");
});

test("observe_message stores project decisions", async () => {
  const result = await callTool("observe_message", {
    message: "I'm gonna use MongoDB for my Job Listing Portal",
    project: "Job Listing Portal",
  });

  const data = toolResult(result);

  assert.equal(result.isError, undefined);
  assert.equal(data.saved, true);
  assert.equal(data.memory_id, memoryId);
  assert.equal(data.inference_ms, 0);
  assert.equal(
    observed?.message,
    "I'm gonna use MongoDB for my Job Listing Portal",
  );
});

test("observe_message skips general knowledge questions", async () => {
  const result = await callTool("observe_message", {
    message: "What's caching?",
  });

  const data = toolResult(result);

  assert.equal(data.saved, false);
  assert.equal(data.reason, "question");
});

test("recall_memory retrieves persistent evidence", async () => {
  const result = await callTool("recall_memory", {
    query: "Which database did we choose?",
    project: "Job Listing Portal",
  });

  const data = toolResult(result);

  assert.equal(result.isError, undefined);
  assert.match(data.answer, /MongoDB/);
  assert.equal(data.evidence[0].memory_id, memoryId);
  assert.equal(data.inference_ms, 0);
  assert.equal(recalled?.project, "Job Listing Portal");
});

test("verify_context propagates backend errors", async () => {
  const result = await callTool("verify_context", {
    claim: "Project Alpha uses PostgreSQL",
    evidence_ids: [memoryId],
  });

  assert.equal(result.isError, true);
  assert.match(JSON.stringify(result), /503/);
});

test("invalid store_context input is rejected", async () => {
  const beforeCount = backendRequests;

  const result = await callTool("store_context", {
    text: "",
  });

  assert.equal(result.isError, true);
  assert.equal(backendRequests, beforeCount);
});

test("inject_context truncates oversized evidence", async () => {
  const result = await callTool("inject_context", {
    memory_ids: [memoryId],
  });

  const data = toolResult(result);

  assert.equal(result.isError, undefined);
  assert.equal(data.evidence[0].text.length, 8000);
  assert.equal(data.truncated, true);
});

test("inject_context preserves exact-size evidence", async () => {
  const result = await callTool("inject_context", {
    memory_ids: [exactId],
  });

  const data = toolResult(result);

  assert.equal(result.isError, undefined);
  assert.equal(data.evidence[0].text.length, 8000);
  assert.equal(data.truncated, false);
});

test("stdio MCP server starts correctly", async () => {
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [
      resolve("node_modules/tsx/dist/cli.mjs"),
      "src/server.ts",
    ],
    cwd: process.cwd(),
    stderr: "pipe",
  });

  const client = new Client({
    name: "stdio-test",
    version: "1.0.0",
  });

  await client.connect(transport);

  try {
    const result = await client.listTools();
    assert.equal(result.tools.length, 6);
  } finally {
    await client.close();
    await transport.close();
  }
});
