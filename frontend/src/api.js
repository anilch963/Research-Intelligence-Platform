const API_URL = import.meta.env.VITE_API_URL ?? "";

async function request(path, options) {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const getHealth = () => request("/api/health");

export const query = (question, retriever) =>
  request("/api/query", {
    method: "POST",
    body: JSON.stringify({ question, retriever }),
  });
