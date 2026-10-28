import { describe, expect, it } from "vitest";

import { errorMessage } from "./api";
import { customerStep, day, money, statusLabel, titleCase } from "./format";
import { pickHeaders, upstreamUrl } from "./forward";
import { safeNext } from "./nav";
import { boxPercent, classColor } from "./overlay";
import { photoProblems } from "./photos";
import type { Detection, User } from "./types";

const GW = "http://gateway:8000";
const customer: User = { username: "a@example.com", role: "customer", name: "A", customer_id: "C00001" };
const adjuster: User = { username: "sam", role: "adjuster", name: "Sam", customer_id: null };

describe("upstreamUrl (the /api proxy's allowlist)", () => {
  it("maps allowed paths and keeps the query", () => {
    expect(upstreamUrl(["claims", "abc"], "?status=completed", GW)?.href).toBe(
      "http://gateway:8000/claims/abc?status=completed",
    );
    expect(upstreamUrl(["auth", "login"], "", GW)?.pathname).toBe("/auth/login");
  });

  it.each([
    [["docs"]],
    [["openapi.json"]],
    [["health"]],
    [[]],
    [["claims", ".."]],
    [["claims", "..", "docs"]],
    [["claims", "a/b"]],
    [["claims", "a\\b"]],
    [["claims", "%2e%2e"]],
    [["claims", "x?y"]],
    [["claims", ""]],
  ])("refuses %j", (segments) => {
    expect(upstreamUrl(segments, "", GW)).toBeNull();
  });

  it("can't be pointed at another host", () => {
    expect(upstreamUrl(["claims", "@evil.com"], "", GW)?.host).toBe("gateway:8000");
  });
});

describe("pickHeaders", () => {
  it("passes only listed headers", () => {
    const from = new Headers({ cookie: "s=1", "x-forwarded-for": "6.6.6.6", host: "evil", accept: "*/*" });
    const out = pickHeaders(from, ["accept", "cookie", "content-type"]);
    expect([...out.keys()].sort()).toEqual(["accept", "cookie"]);
  });
});

describe("safeNext (no open redirect after login)", () => {
  it.each([
    [null, customer, "/claims"],
    ["/claims/abc", customer, "/claims/abc"],
    ["/adjuster", customer, "/claims"], // not this role's area
    ["/claimsfake", customer, "/claims"],
    ["//evil.com", customer, "/claims"],
    ["/\\evil.com", customer, "/claims"],
    ["https://evil.com", adjuster, "/adjuster"],
    ["/adjuster/claims/x", adjuster, "/adjuster/claims/x"],
  ])("%s for %s -> %s", (next, user, expected) => {
    expect(safeNext(next as string | null, user as User)).toBe(expected);
  });
});

describe("overlay geometry", () => {
  const det: Detection = {
    class_id: 0,
    class_name: "dent",
    confidence: 0.9,
    box: { x1: 100, y1: 50, x2: 300, y2: 150 },
    area_pct: 2,
  };

  it("converts photo pixels to percent", () => {
    expect(boxPercent(det, 1000, 500)).toEqual({ left: 10, top: 10, width: 20, height: 20 });
  });

  it("clamps boxes that spill past the photo", () => {
    const p = boxPercent({ ...det, box: { x1: -10, y1: 0, x2: 1200, y2: 500 } }, 1000, 500);
    expect(p).toEqual({ left: 0, top: 0, width: 100, height: 100 });
  });

  it("has a colour per class and a fallback", () => {
    expect(classColor("scratch")).not.toBe(classColor("dent"));
    expect(classColor("unknown")).toBe(classColor("dent"));
  });
});

describe("photoProblems (mirrors the gateway's checks)", () => {
  const file = (name: string, type: string, size = 10) => new File([new Uint8Array(size)], name, { type });

  it("accepts good photos", () => {
    expect(photoProblems([file("a.jpg", "image/jpeg"), file("b.webp", "image/webp")])).toEqual([]);
  });

  it("flags type, size and count", () => {
    expect(photoProblems([])).toEqual(["Add at least one photo."]);
    expect(photoProblems([file("a.pdf", "application/pdf")])[0]).toMatch(/JPEG, PNG or WebP/);
    expect(photoProblems([file("big.jpg", "image/jpeg", 10 * 1024 * 1024 + 1)])[0]).toMatch(/10 MB/);
    expect(photoProblems(Array.from({ length: 11 }, (_, i) => file(`${i}.jpg`, "image/jpeg")))).toContain(
      "Up to 10 photos per upload.",
    );
  });
});

describe("formatting", () => {
  it("formats money and dates", () => {
    expect(money(1234.5)).toBe("$1,234.50");
    expect(money(null)).toBe("—");
    expect(day("2026-09-01")).toBe("Sep 1, 2026"); // no off-by-one from time zones
    expect(titleCase("glass shatter")).toBe("Glass Shatter");
    expect(titleCase("claim_burst")).toBe("Claim Burst");
  });

  it("describes statuses per audience", () => {
    expect(statusLabel("awaiting_approval", "customer")).toBe("With an adjuster");
    expect(statusLabel("awaiting_approval", "adjuster")).toBe("Needs decision");
    expect(statusLabel("failed", "customer")).toBe("Delayed");
    expect(customerStep("completed")).toBe(3);
  });

  it("turns FastAPI errors into a sentence", () => {
    expect(errorMessage("Unknown policy", 422)).toBe("Unknown policy");
    expect(errorMessage([{ msg: "too short", loc: ["body", "description"] }], 422)).toBe("description: too short");
    expect(errorMessage(undefined, 429)).toMatch(/Too many/);
  });
});
