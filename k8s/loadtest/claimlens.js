// k6 load test for ClaimLens on Kubernetes. Run it with k8s/loadtest/run.sh,
// which starts k6 in Docker on the kind network and saves the summary.
//
// Scenarios (SCENARIO=browse|detect|claims, comma-separated):
//   browse  logged-in customers reading their claims and policies:
//           edge -> frontend -> gateway -> agent -> Postgres, Redis sessions
//   detect  photo uploads to /api/detect: gateway -> object storage -> Redis
//           stream -> vision workers, which KEDA scales on the backlog
//   claims  full claim submissions: guardrails (OCR, classifier, CLIP), vision,
//           MCP tools, the agent graph. Run with the agent's LLM key removed
//           (run.sh does), so it measures this system, not Ollama Cloud.
//
// Every request goes through the same public path a browser uses.

import http from "k6/http";
import { check, sleep } from "k6";
import { SharedArray } from "k6/data";
import { Counter, Trend } from "k6/metrics";

const BASE = __ENV.BASE_URL || "http://claimlens-control-plane:30080";
const SCENARIOS = (__ENV.SCENARIO || "browse,detect").split(",");
// [{username, password}], from k8s/loadtest/run.sh (git-ignored data/ folder).
const users = new SharedArray("users", () => JSON.parse(open(__ENV.USERS || "/data/users.json")));
const photo = open(__ENV.PHOTO || "/photo.jpg", "b");

// Upload to result: includes waiting for a worker when the queue is long.
const detectTotal = new Trend("detect_total_ms", true);
const detectPending = new Counter("detect_returned_pending");
const claimDone = new Trend("claim_to_decision_ms", true);

// Step tests: hold each rate for a while, so each step's latency and errors
// can be read separately (k8s/loadtest/analyze.py) and the knee is visible.
// The VU caps keep k6 small: it runs in the same Docker VM as the cluster.
function steps(rates, hold) {
  return rates.flatMap((r) => [{ target: r, duration: "10s" }, { target: r, duration: hold }]);
}

export const STEPS = {
  browse: [25, 100, 200, 400], // iterations/s, 3 API requests each
  detect: [1, 3, 6, 10], // uploads/s
};

const ALL = {
  browse: {
    executor: "ramping-arrival-rate",
    exec: "browse",
    startRate: 1,
    timeUnit: "1s",
    preAllocatedVUs: 30,
    maxVUs: 250,
    stages: steps(STEPS.browse, "60s"),
  },
  detect: {
    executor: "ramping-arrival-rate",
    exec: "detect",
    startRate: 1,
    timeUnit: "1s",
    preAllocatedVUs: 20,
    maxVUs: 120,
    stages: steps(STEPS.detect, "90s"),
  },
  claims: {
    executor: "constant-arrival-rate",
    exec: "claim",
    rate: Number(__ENV.CLAIMS_PER_MIN || 30),
    timeUnit: "1m",
    duration: __ENV.CLAIMS_DURATION || "4m",
    preAllocatedVUs: 20,
    maxVUs: 100,
  },
};

export const options = {
  scenarios: Object.fromEntries(SCENARIOS.map((s) => [s, ALL[s]])),
  setupTimeout: "2m",
  // Pass/fail lines, recorded with the results. Not a gate (yet): they say
  // what "fine" meant for this run.
  thresholds: {
    "http_req_failed{scenario:browse}": ["rate<0.01"],
    "http_req_duration{scenario:browse}": ["p(95)<500"],
    "http_req_failed{scenario:detect}": ["rate<0.01"],
    detect_total_ms: ["p(95)<10000"],
    "http_req_failed{scenario:claims}": ["rate<0.02"],
  },
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max"],
};

const JSON_HEADERS = { "Content-Type": "application/json" };

export function setup() {
  // One session per user, shared by the VUs (sessions live in Redis).
  return users.map((u) => {
    const r = http.post(`${BASE}/api/auth/login`, JSON.stringify(u), { headers: JSON_HEADERS });
    check(r, { "login 200": (res) => res.status === 200 });
    const session = r.cookies.claimlens_session[0].value;
    const p = http.get(`${BASE}/api/me/policies`, { cookies: { claimlens_session: session } });
    const policies = p.json("policies") || [];
    const active = policies.find((x) => x.status === "active") || policies[0];
    return { session, policy: active && active.policy_id };
  });
}

function as(sessions) {
  const s = sessions[(__VU + __ITER) % sessions.length];
  return { s, params: { cookies: { claimlens_session: s.session } } };
}

export function browse(sessions) {
  const { params } = as(sessions);
  const responses = http.batch([
    ["GET", `${BASE}/api/auth/me`, null, { ...params, tags: { name: "me" } }],
    ["GET", `${BASE}/api/claims?limit=20`, null, { ...params, tags: { name: "claims" } }],
    ["GET", `${BASE}/api/me/policies`, null, { ...params, tags: { name: "policies" } }],
  ]);
  for (const r of responses) check(r, { "browse 200": (res) => res.status === 200 });
}

export function detect(sessions) {
  const { params } = as(sessions);
  const started = Date.now();
  let r = http.post(
    `${BASE}/api/detect`,
    { file: http.file(photo, "car.jpg", "image/jpeg") },
    { ...params, tags: { name: "detect" }, timeout: "60s" },
  );
  if (r.status === 202) {
    // The gateway waits up to VISION_TIMEOUT_S, then hands back a job to poll.
    detectPending.add(1);
    const id = r.json("job_id");
    for (let i = 0; i < 60 && r.status === 202; i++) {
      sleep(1);
      r = http.get(`${BASE}/api/jobs/${id}`, { ...params, tags: { name: "job" } });
    }
  }
  check(r, { "detect 200": (res) => res.status === 200 });
  if (r.status === 200) detectTotal.add(Date.now() - started);
}

export function claim(sessions) {
  const { s, params } = as(sessions);
  const started = Date.now();
  const r = http.post(
    `${BASE}/api/claims`,
    {
      policy_id: s.policy,
      incident_date: "2026-10-01",
      description: "Load test: rear bumper dented in a car park, no one hurt.",
      photos: http.file(photo, "car.jpg", "image/jpeg"),
    },
    { ...params, tags: { name: "claim" }, timeout: "60s" },
  );
  if (!check(r, { "claim 202": (res) => res.status === 202 })) return;
  const id = r.json("claim_id");
  // Until the agent stops for the adjuster (or retake/fails): the whole graph.
  for (let i = 0; i < 120; i++) {
    sleep(1);
    const g = http.get(`${BASE}/api/claims/${id}`, { ...params, tags: { name: "claim_status" } });
    const status = g.status === 200 ? g.json("status") : null;
    if (status && status !== "processing") {
      check(g, { "claim reached a decision point": () => status.startsWith("awaiting") });
      claimDone.add(Date.now() - started);
      return;
    }
  }
}
