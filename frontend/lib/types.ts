// Shapes returned by the gateway. They mirror services/gateway/claims.py
// (customer views) and services/agent/graph.py (the adjuster's full state).

export type Role = "customer" | "adjuster";

export interface User {
  username: string;
  role: Role;
  name: string;
  customer_id: string | null;
}

export type ClaimStatus =
  | "processing"
  | "awaiting_retake"
  | "awaiting_approval"
  | "completed"
  | "failed"
  | "interrupted";

export type DecisionAction = "approve" | "edit" | "reject";
export type Recommendation = "approve" | "review" | "deny" | "refer_to_investigator";

export interface Policy {
  policy_id: string;
  status: "active" | "lapsed" | "cancelled";
  start_date: string;
  end_date: string;
  collision: boolean;
  comprehensive: boolean;
  vehicle: string;
}

export interface RetakeQuestion {
  type: "retake";
  photos: { photo_index: number; problems: string[] }[];
}

export interface ApprovalQuestion {
  type: "approval";
  recommendation: { action: Recommendation; reasons: string[] };
  payable: Range;
  fraud: { score: number; level: FraudLevel };
}

// ---- customer views ----

export interface CustomerClaim {
  claim_id: string;
  status: ClaimStatus;
  policy_id: string;
  incident_date: string;
  reported_date: string;
  description: string;
  claimed_amount: number | null;
  photo_count: number;
  question: RetakeQuestion | null;
  outcome: { action: DecisionAction; amount: number } | null;
  message: string | null;
}

export interface CustomerClaimRow {
  claim_id: string;
  policy_id: string;
  status: ClaimStatus;
  created_at: string;
  updated_at: string;
  incident_date: string | null;
  damages: string[];
  decision: { action: DecisionAction; amount: number } | null;
}

// ---- adjuster views ----

export type FraudLevel = "low" | "medium" | "high";

export interface Range {
  low: number;
  expected: number;
  high: number;
}

export interface AdjusterClaimRow {
  claim_id: string;
  customer_id: string | null;
  policy_id: string;
  status: ClaimStatus;
  created_at: string;
  updated_at: string;
  summary: {
    incident_date?: string;
    damages?: string[];
    recommendation?: Recommendation | null;
    fraud_level?: FraudLevel | null;
    fraud_score?: number | null;
    payable_expected?: number | null;
    decision?: { action: DecisionAction; amount: number } | null;
    security_findings?: number;
  };
}

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  box: { x1: number; y1: number; x2: number; y2: number };
  area_pct: number;
  mask_png?: string | null; // base64 1-bit PNG of the mask inside the box
}

export interface VisionResult {
  width: number;
  height: number;
  detections: Detection[];
  damage_area_pct: number;
}

export interface Quality {
  ok: boolean;
  problems: string[];
  photo_index: number;
  width?: number;
  height?: number;
  sharpness?: number;
  brightness?: number;
}

export interface Damage {
  damage_class: string;
  severity: "minor" | "moderate" | "severe";
  area_pct: number;
  confidence: number;
  photo_index: number;
}

export interface Clause {
  id: string;
  wording_id: string;
  section: string;
  page_start: number;
  page_end: number;
  text: string;
  /** Flagged by guardrails as possible injection; never shown to the LLM. */
  withheld?: boolean;
}

export interface Opinion {
  assessment: "covered" | "not_covered" | "unclear";
  reasoning: string;
  citations: string[];
  concerns: string[];
  source: "llm" | "fallback" | "code";
  searches: { query?: string; added?: string[]; error?: string; refused?: string }[];
}

export interface Coverage {
  covered: boolean;
  in_force: boolean | null;
  peril_covered: boolean | null;
  deductible: number | null;
  reasons: string[];
  error?: string;
}

export interface Estimate {
  currency: string;
  labour_rate_per_hour: number;
  line_items: {
    damage_class: string;
    severity: string;
    operation: string;
    parts: number;
    labour_hours: number;
    paint_hours: number;
    cost: number;
  }[];
  total: number;
  low: number;
  high: number;
  assumptions: string[];
}

export interface FraudSignal {
  signal: string;
  weight: number;
  detail: string;
}

export interface TraceStep {
  node: string;
  at: string;
  summary: string;
  source?: string;
  problems?: string[];
  citations?: string[];
}

/** A guardrails finding (agent state["security"]). */
export type SecurityKind = "injection" | "visual_injection" | "rag_injection" | "forensics" | "guardrails_unavailable";

export interface SecurityFinding {
  node: string;
  source: string; // "description", "photo 0", "clause C3", "policy clauses"
  kind: SecurityKind;
  detail: string;
  excerpt?: string; // the flagged text itself: untrusted, render as text only
  rules?: string[];
  score?: number | null;
  code?: string;
}

export interface MetadataFinding {
  code: string;
  severity: "info" | "warning" | "high";
  detail: string;
}

export interface PhotoCheck {
  photo_index: number;
  error?: string;
  ocr_text?: string;
  text_scan?: { flagged: boolean; score: number | null; rules: string[] } | null;
  redactions?: { faces: number; text_regions: number; plates: number };
  metadata?: {
    camera: string | null;
    software: string | null;
    taken_at: string | null;
    has_exif: boolean;
    has_gps: boolean;
    findings: MetadataFinding[];
  };
  ai_generated?: { score: number; flagged: boolean; model: string } | null;
}

export interface PhotoReview {
  matches_description: "yes" | "no" | "unclear";
  damage_seen: string[];
  concerns: string[];
  summary: string;
  source: string;
  photos_shown: number;
}

export interface ClaimState {
  claim_id: string;
  customer_id?: string;
  policy_id: string;
  incident_date: string;
  reported_date: string;
  description: string;
  claimed_amount: number | null;
  photos: { key: string; job_id: string }[];
  retakes?: number;
  intake?: {
    peril: string;
    summary: string;
    mentioned_date: string | null;
    other_vehicle_involved: boolean;
    injuries_mentioned: boolean;
    screen?: "clean" | "flagged" | "unscanned";
  };
  quality?: Quality[];
  vision?: (VisionResult | null)[];
  damages?: Damage[];
  peril?: string;
  policy?: {
    policy_id: string;
    state: string;
    status: string;
    start_date: string;
    end_date: string;
    collision_deductible: number | null;
    comprehensive_deductible: number | null;
    vehicle: { make: string; model: string; year: number; segment: string };
    wording: { title?: string; insurer?: string } | null;
  } | null;
  coverage?: Coverage;
  clauses?: Clause[];
  opinion?: Opinion;
  estimate?: Estimate | null;
  estimate_error?: string | null;
  payable?: Range;
  fraud?: {
    score: number;
    level: FraudLevel;
    signals: FraudSignal[];
    history: { claims_last_12_months: number; total_paid: number; past_claims: number };
  };
  recommendation?: { action: Recommendation; reasons: string[] };
  decision?: { action: DecisionAction; amount: number; adjuster: string; notes: string };
  report?: { customer_message: string; adjuster_summary: string };
  security?: SecurityFinding[];
  photo_checks?: PhotoCheck[];
  photo_review?: PhotoReview;
}

export interface AdjusterClaim {
  claim_id: string;
  status: ClaimStatus;
  question: RetakeQuestion | ApprovalQuestion | null;
  error: string | null;
  state: ClaimState;
  trace: TraceStep[];
}
