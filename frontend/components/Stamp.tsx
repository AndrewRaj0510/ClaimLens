import { statusLabel, statusTone, type Tone } from "@/lib/format";
import type { ClaimStatus, Role } from "@/lib/types";

const TONES: Record<Tone, string> = {
  busy: "border-slate text-slate",
  attention: "border-signal text-signal",
  good: "border-moss text-moss",
  bad: "border-rust text-rust",
};

/** A claim's status, drawn as a rubber stamp on the file. */
export function Stamp({
  status,
  role,
  size = "sm",
}: {
  status: ClaimStatus;
  role: Role;
  size?: "sm" | "lg";
}) {
  const big = size === "lg";
  return (
    <span
      className={`inline-flex items-center gap-1.5 whitespace-nowrap border-2 font-mono uppercase ${TONES[statusTone(status)]} ${
        big ? "-rotate-2 px-3 py-1 text-sm tracking-[0.18em]" : "px-1.5 py-0.5 text-[0.6875rem] tracking-[0.1em]"
      }`}
    >
      {status === "processing" && (
        <span className="size-1.5 animate-pulse rounded-full bg-current" aria-hidden />
      )}
      {statusLabel(status, role)}
    </span>
  );
}
