import type { Detection } from "./types";

// One colour per CarDD class, picked to stay apart from each other and to
// read on both light and dark car paint.
export const CLASS_COLORS: Record<string, string> = {
  dent: "#e2471a",
  scratch: "#f2b705",
  crack: "#d6246e",
  "glass shatter": "#1fa3c7",
  "lamp broken": "#8e5bd8",
  "tire flat": "#3fb27f",
};

export function classColor(name: string): string {
  return CLASS_COLORS[name] ?? "#e2471a";
}

/**
 * A box in photo pixels -> its position in percent of the photo, so the
 * overlay scales with the displayed image at any size.
 */
export function boxPercent(det: Detection, width: number, height: number) {
  const clamp = (v: number) => Math.min(100, Math.max(0, v));
  const left = clamp((det.box.x1 / width) * 100);
  const top = clamp((det.box.y1 / height) * 100);
  return {
    left,
    top,
    width: clamp((det.box.x2 / width) * 100) - left,
    height: clamp((det.box.y2 / height) * 100) - top,
  };
}

export function maskHref(det: Detection): string | null {
  return det.mask_png ? `data:image/png;base64,${det.mask_png}` : null;
}
