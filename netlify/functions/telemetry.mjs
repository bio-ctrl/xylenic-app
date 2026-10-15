// Telemetry endpoint: runs a small schema-optimization pass and reports
// how long it actually took on the server.
export default async (req) => {
  const start = performance.now();

  const fields = Array.from({ length: 2048 }, (_, i) => ({ key: `f${i % 97}`, weight: (i * 31) % 17 }));
  const schema = new Map();
  for (const { key, weight } of fields) schema.set(key, (schema.get(key) ?? 0) + weight);
  const optimized = [...schema.entries()].sort((a, b) => b[1] - a[1]);

  const executionMs = performance.now() - start;

  return Response.json(
    {
      status: "ok",
      fieldsOptimized: optimized.length,
      executionMs: Number(executionMs.toFixed(3)),
      networkEgressBytes: 0,
      timestamp: new Date().toISOString(),
    },
    { headers: { "Cache-Control": "no-store" } },
  );
};

export const config = { path: "/api/telemetry" };
