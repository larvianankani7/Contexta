const defaultTimeoutMs = 120_000;

export function backendUrl(): URL {
  const value = process.env.CONTEXTA_BACKEND_URL ?? "http://127.0.0.1:8000";
  const url = new URL(value);
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password) {
    throw new Error("CONTEXTA_BACKEND_URL must be an HTTP(S) URL without embedded credentials");
  }
  return url;
}

export async function backendRequest<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const url = new URL(path, backendUrl());
  const response = await fetch(url, {
    ...init,
    headers: {
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
    signal: AbortSignal.timeout(defaultTimeoutMs),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Contexta backend returned ${response.status}: ${detail.slice(0, 500)}`);
  }
  return (await response.json()) as T;
}

export function toolText(value: unknown) {
  return { content: [{ type: "text" as const, text: JSON.stringify(value) }] };
}
