import { after, before, test } from "node:test";
import assert from "node:assert/strict";
import { createServer as createHttpServer, type Server as HttpServer } from "node:http";
import { resolve } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { createServer } from "../src/server.js";

let backend: HttpServer;
let backendRequests = 0;

before(async () => {
  backend = createHttpServer((request, response) => {
    backendRequests += 1;
    if (request.url === "/context" && request.method === "POST") {
      response.writeHead(201, { "content-type": "application/json" });
      response.end(JSON.stringify({ id: "7c9e6679-7425-40de-944b-e07fc1f90ae7" }));
      return;
    }
    if (request.url === "/memory/7c9e6679-7425-40de-944b-e07fc1f90ae7") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({
        source: "test",
        facts: [{ text: "x".repeat(10_000) }],
      }));
      return;
    }
    if (request.url === "/memory/19c9e667-7425-40de-944b-e07fc1f90ae7") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({
        source: "test",
        facts: [{ text: "x".repeat(8_000) }],
      }));
      return;
    }
    if (request.url === "/retrieve") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({
        query: "test",
        evidence: [{ memory_id: "memory", text: "bounded", source: "test", relevance: 1 }],
      }));
      return;
    }
    response.writeHead(503, { "content-type": "application/json" });
    response.end(JSON.stringify({ detail: "unavailable" }));
  });
  await new Promise<void>((resolve) => backend.listen(0, "127.0.0.1", resolve));
  const address = backend.address();
  if (!address || typeof address === "string") {
    throw new Error("Could not start backend test server");
  }
  process.env.CONTEXTA_BACKEND_URL = `http://127.0.0.1:${address.port}`;
});

after(async () => {
  await new Promise<void>((resolve, reject) => {
    backend.close((error) => (error ? reject(error) : resolve()));
  });
});

async function connectedClient(): Promise<{
  client: Client;
  server: ReturnType<typeof createServer>;
}> {
  const server = createServer();
  const client = new Client({ name: "contexta-test", version: "1.0.0" });
  const [serverTransport, clientTransport] = InMemoryTransport.createLinkedPair();
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  return { client, server };
}

test("MCP initializes and advertises the four Contexta tools", async () => {
  const { client, server } = await connectedClient();
  try {
    const result = await client.listTools();
    assert.deepEqual(
      result.tools.map((tool) => tool.name).sort(),
      ["inject_context", "retrieve_context", "store_context", "verify_context"],
    );
  } finally {
    await client.close();
    await server.close();
  }
});

test("store_context forwards the validated input to the backend", async () => {
  const { client, server } = await connectedClient();
  try {
    const result = await client.callTool({
      name: "store_context",
      arguments: { text: "Project Alpha uses PostgreSQL.", source: "test" },
    });
    assert.equal(result.isError, undefined);
    assert.match(result.content[0].type === "text" ? result.content[0].text : "", /7c9e6679/);
  } finally {
    await client.close();
    await server.close();
  }
});

test("retrieve_context returns ranked backend evidence", async () => {
  const { client, server } = await connectedClient();
  try {
    const result = await client.callTool({
      name: "retrieve_context",
      arguments: { query: "Project Alpha database", limit: 3 },
    });
    assert.equal(result.isError, undefined);
    const response = JSON.parse(result.content[0].type === "text" ? result.content[0].text : "{}");
    assert.equal(response.evidence[0].text, "bounded");
  } finally {
    await client.close();
    await server.close();
  }
});

test("backend errors are returned as MCP tool errors", async () => {
  const { client, server } = await connectedClient();
  try {
    const result = await client.callTool({
      name: "verify_context",
      arguments: {
        claim: "Project Alpha uses PostgreSQL",
        evidence_ids: ["7c9e6679-7425-40de-944b-e07fc1f90ae7"],
      },
    });

    test("tool input validation rejects empty context", async () => {
      const { client, server } = await connectedClient();
      try {
        const requestsBefore = backendRequests;
        await client.callTool({ name: "store_context", arguments: { text: "" } });
        assert.equal(backendRequests, requestsBefore);
      } finally {
        await client.close();
        await server.close();
      }
    });

    test("inject_context bounds selected memory output", async () => {
      const { client, server } = await connectedClient();
      try {
        const result = await client.callTool({
          name: "inject_context",
          arguments: { memory_ids: ["7c9e6679-7425-40de-944b-e07fc1f90ae7"] },
        });
        assert.equal(result.isError, undefined);
        const response = JSON.parse(result.content[0].type === "text" ? result.content[0].text : "{}");
        assert.equal(response.evidence[0].text.length, 8_000);
        assert.equal(response.truncated, true);
      } finally {
        await client.close();
        await server.close();
      }
    });

    test("inject_context does not report exact-limit text as truncated", async () => {
      const { client, server } = await connectedClient();
      try {
        const result = await client.callTool({
          name: "inject_context",
          arguments: { memory_ids: ["19c9e667-7425-40de-944b-e07fc1f90ae7"] },
        });
        assert.equal(result.isError, undefined);
        const response = JSON.parse(result.content[0].type === "text" ? result.content[0].text : "{}");
        assert.equal(response.evidence[0].text.length, 8_000);
        assert.equal(response.truncated, false);
      } finally {
        await client.close();
        await server.close();
      }
    });

    test("stdio process starts with protocol-only stdout and serves tool discovery", async () => {
      const transport = new StdioClientTransport({
        command: process.execPath,
        args: [resolve("node_modules/tsx/dist/cli.mjs"), "src/server.ts"],
        cwd: process.cwd(),
        stderr: "pipe",
      });
      const client = new Client({ name: "stdio-test", version: "1.0.0" });
      await client.connect(transport);
      try {
        const result = await client.listTools();
        assert.equal(result.tools.length, 4);
      } finally {
        await client.close();
        await transport.close();
      }
    });
    assert.match(JSON.stringify(result), /503/);
  } finally {
    await client.close();
    await server.close();
  }
});
