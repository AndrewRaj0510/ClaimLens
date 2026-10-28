"use client";

import { useId, useState } from "react";

import { photoUrl } from "@/lib/api";
import { boxPercent, classColor, maskHref } from "@/lib/overlay";
import type { Detection, Quality, VisionResult } from "@/lib/types";

/** Corner brackets instead of a closed rectangle: the photo stays readable. */
function bracketPath(d: Detection) {
  const { x1, y1, x2, y2 } = d.box;
  const k = Math.max(6, Math.min(x2 - x1, y2 - y1) * 0.22);
  return [
    `M${x1} ${y1 + k}V${y1}H${x1 + k}`,
    `M${x2 - k} ${y1}H${x2}V${y1 + k}`,
    `M${x2} ${y2 - k}V${y2}H${x2 - k}`,
    `M${x1 + k} ${y2}H${x1}V${y2 - k}`,
  ].join("");
}

export function DamageOverlay({
  result,
  hidden,
  masks,
  boxes,
}: {
  result: VisionResult;
  hidden: Set<string>;
  masks: boolean;
  boxes: boolean;
}) {
  const uid = useId().replace(/:/g, "");
  const shown = result.detections.filter((d) => !hidden.has(d.class_name));
  return (
    <>
      {/* Drawn in photo pixels: the viewBox is the photo, stretched over the <img>. */}
      <svg
        className="pointer-events-none absolute inset-0 size-full"
        viewBox={`0 0 ${result.width} ${result.height}`}
        preserveAspectRatio="none"
        aria-hidden
      >
        <defs>
          {shown.map((d, i) => {
            const href = maskHref(d);
            if (!href) return null;
            const { x1, y1, x2, y2 } = d.box;
            // SVG masks are luminance masks: the PNG's white pixels let the colour through.
            return (
              <mask key={i} id={`${uid}-m${i}`} maskUnits="userSpaceOnUse" x="0" y="0" width={result.width} height={result.height}>
                <image href={href} x={x1} y={y1} width={x2 - x1} height={y2 - y1} preserveAspectRatio="none" />
              </mask>
            );
          })}
        </defs>
        {shown.map((d, i) => {
          const color = classColor(d.class_name);
          const { x1, y1, x2, y2 } = d.box;
          return (
            <g key={i} data-testid="detection">
              {masks && maskHref(d) && (
                <rect
                  data-testid="mask"
                  x={x1}
                  y={y1}
                  width={x2 - x1}
                  height={y2 - y1}
                  fill={color}
                  fillOpacity={0.5}
                  mask={`url(#${uid}-m${i})`}
                />
              )}
              {boxes && (
                <path
                  d={bracketPath(d)}
                  fill="none"
                  stroke={color}
                  strokeWidth={2.5}
                  vectorEffect="non-scaling-stroke"
                />
              )}
            </g>
          );
        })}
      </svg>
      {/* Labels in HTML, positioned in %, so text keeps its size at any zoom. */}
      {boxes &&
        shown.map((d, i) => {
          const p = boxPercent(d, result.width, result.height);
          return (
            <span
              key={i}
              data-testid="label"
              className="pointer-events-none absolute -translate-y-full whitespace-nowrap px-1 py-px font-mono text-[10px] uppercase tracking-wider text-white"
              style={{ left: `${p.left}%`, top: `${p.top}%`, background: classColor(d.class_name) }}
            >
              {d.class_name} {Math.round(d.confidence * 100)}%
            </span>
          );
        })}
    </>
  );
}

export function PhotoViewer({
  claimId,
  count,
  vision,
  quality,
}: {
  claimId: string;
  count: number;
  vision: (VisionResult | null)[];
  quality: Quality[];
}) {
  const [active, setActive] = useState(0);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [masks, setMasks] = useState(true);
  const [boxes, setBoxes] = useState(true);
  const result = vision[active] ?? null;
  const q = quality.find((x) => x.photo_index === active);
  const classes = [...new Set(vision.flatMap((v) => v?.detections.map((d) => d.class_name) ?? []))];

  function toggle(name: string) {
    const next = new Set(hidden);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    setHidden(next);
  }

  return (
    <div className="sheet">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-rule px-4 py-2.5">
        <span className="label">Evidence · photo {active + 1} of {count}</span>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          {classes.map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => toggle(name)}
              aria-pressed={!hidden.has(name)}
              className={`flex items-center gap-1.5 border px-2 py-0.5 font-mono text-[11px] uppercase transition-opacity ${
                hidden.has(name) ? "border-rule opacity-40" : "border-ink"
              }`}
            >
              <span className="size-2" style={{ background: classColor(name) }} />
              {name}
            </button>
          ))}
          <Toggle label="Masks" on={masks} set={setMasks} />
          <Toggle label="Boxes" on={boxes} set={setBoxes} />
        </div>
      </div>

      <div className="relative bg-ink">
        <div
          className="relative mx-auto"
          // Height is capped through the width, so the box always has the
          // photo's exact aspect ratio and the overlay lines up with the image.
          style={
            result
              ? {
                  aspectRatio: `${result.width} / ${result.height}`,
                  width: `min(100%, calc(34rem * ${result.width / result.height}))`,
                }
              : undefined
          }
        >
          <img
            key={active}
            src={photoUrl(claimId, active)}
            alt={`Claim photo ${active + 1}`}
            className="block size-full object-contain"
          />
          {result && <DamageOverlay result={result} hidden={hidden} masks={masks} boxes={boxes} />}
        </div>
        {!result && (
          <span className="absolute left-3 top-3 bg-rust px-2 py-0.5 font-mono text-xs uppercase text-white">
            Detection failed or pending
          </span>
        )}
      </div>

      <div className="grid gap-4 p-4 sm:grid-cols-[1fr_auto]">
        <div className="text-sm">
          {result ? (
            <p>
              <span className="font-mono">{result.detections.length}</span> detection
              {result.detections.length === 1 ? "" : "s"} · damage covers{" "}
              <span className="font-mono">{result.damage_area_pct.toFixed(1)}%</span> of the photo
              <span className="text-faint"> · {result.width}×{result.height}px</span>
            </p>
          ) : (
            <p className="text-rust">No vision result for this photo: check it by eye.</p>
          )}
          {q && !q.ok && (
            <p className="mt-1 text-amber">Quality: {q.problems.join(" ")}</p>
          )}
        </div>
        {count > 1 && (
          <ul className="flex gap-2">
            {Array.from({ length: count }, (_, i) => (
              <li key={i}>
                <button
                  type="button"
                  onClick={() => setActive(i)}
                  aria-label={`Show photo ${i + 1}`}
                  aria-current={i === active}
                  className={`relative block size-14 overflow-hidden border-2 ${i === active ? "border-signal" : "border-transparent opacity-70 hover:opacity-100"}`}
                >
                  <img src={photoUrl(claimId, i)} alt="" className="size-full object-cover" loading="lazy" />
                  <span className="absolute bottom-0 right-0 bg-ink px-1 font-mono text-[9px] text-sheet">
                    {vision[i]?.detections.length ?? "?"}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function Toggle({ label, on, set }: { label: string; on: boolean; set: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center gap-1.5 font-mono text-[11px] uppercase">
      <input type="checkbox" checked={on} onChange={(e) => set(e.target.checked)} className="accent-signal" />
      {label}
    </label>
  );
}
