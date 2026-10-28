import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import ClaimPage from "./page";
import type { CustomerClaim } from "@/lib/types";

vi.mock("next/navigation", () => ({ useParams: () => ({ id: "e".repeat(32) }) }));

const BASE: CustomerClaim = {
  claim_id: "e".repeat(32),
  status: "awaiting_approval",
  policy_id: "P00001",
  incident_date: "2026-09-01",
  reported_date: "2026-09-02",
  description: "Rear-ended at a red light.",
  claimed_amount: 900,
  photo_count: 2,
  question: null,
  outcome: null,
  message: null,
};

function serve(claim: CustomerClaim) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(claim), { status: 200 })),
  );
}

describe("customer claim page", () => {
  beforeEach(() => vi.useRealTimers());
  afterEach(() => vi.unstubAllGlobals());

  it("says a person is deciding, and shows the customer's own photos", async () => {
    serve(BASE);
    render(<ClaimPage />);
    expect(await screen.findByText("An adjuster is reviewing your claim")).toBeInTheDocument();
    expect(screen.getAllByText("With an adjuster").length).toBeGreaterThan(0);
    const photos = screen.getAllByRole("img", { name: /^Photo \d$/ });
    expect(photos.map((p) => p.getAttribute("src"))).toEqual([
      `/api/claims/${"e".repeat(32)}/photos/0`,
      `/api/claims/${"e".repeat(32)}/photos/1`,
    ]);
  });

  it("asks for retakes with the reasons", async () => {
    serve({
      ...BASE,
      status: "awaiting_retake",
      question: { type: "retake", photos: [{ photo_index: 1, problems: ["Photo is blurry; hold the phone steady."] }] },
    });
    render(<ClaimPage />);
    expect(await screen.findByText("Please retake one photo")).toBeInTheDocument();
    expect(screen.getByText(/Photo is blurry/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send new photos" })).toBeDisabled(); // nothing chosen yet
  });

  it("shows the outcome and the message once decided", async () => {
    serve({
      ...BASE,
      status: "completed",
      outcome: { action: "edit", amount: 412.5 },
      message: "Your claim has been approved for $412.50.",
    });
    render(<ClaimPage />);
    expect(await screen.findByText("Approved, amount edited")).toBeInTheDocument();
    expect(screen.getByText("$412.50")).toBeInTheDocument();
    expect(screen.getByText(/approved for \$412\.50/)).toBeInTheDocument();
  });
});
