"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { ALLOWED_TYPES, MAX_PHOTOS, photoProblems } from "@/lib/photos";

/** Drop zone + previews. Calls onChange with the chosen files. */
export function PhotoPicker({
  files,
  onChange,
  hint = "Close-ups of each damaged area, plus one of the whole car.",
}: {
  files: File[];
  onChange: (files: File[]) => void;
  hint?: string;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const previews = useMemo(() => files.map((f) => URL.createObjectURL(f)), [files]);
  useEffect(() => () => previews.forEach((u) => URL.revokeObjectURL(u)), [previews]);
  const problems = files.length ? photoProblems(files) : [];

  function add(list: FileList | null) {
    if (list) onChange([...files, ...Array.from(list)]);
  }

  return (
    <div className="space-y-3">
      <button
        type="button"
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          add(e.dataTransfer.files);
        }}
        className={`group relative flex w-full cursor-pointer flex-col items-center justify-center gap-2 border-2 border-dashed px-6 py-10 text-center transition-colors ${
          over ? "border-signal bg-signal/5" : "border-rule bg-white hover:border-ink"
        }`}
      >
        <svg className="size-8 text-faint transition-colors group-hover:text-signal" viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
          <path d="M3 9V3h6M23 3h6v6M29 23v6h-6M9 29H3v-6" />
          <circle cx="16" cy="16" r="5" />
        </svg>
        <span className="font-medium">Drop photos here or browse</span>
        <span className="text-sm text-graphite">{hint}</span>
        <span className="label">JPEG · PNG · WebP · up to {MAX_PHOTOS} · 10 MB each</span>
      </button>
      <input
        ref={input}
        type="file"
        accept={ALLOWED_TYPES.join(",")}
        multiple
        hidden
        onChange={(e) => {
          add(e.target.files);
          e.target.value = "";
        }}
        aria-label="Photos"
      />
      {files.length > 0 && (
        <ul className="grid grid-cols-3 gap-2 sm:grid-cols-5">
          {files.map((f, i) => (
            <li key={`${f.name}-${i}`} className="group relative aspect-square overflow-hidden border border-rule bg-white">
              <img src={previews[i]} alt={f.name} className="size-full object-cover" />
              <span className="absolute left-1 top-1 bg-ink px-1 font-mono text-[10px] text-sheet">
                {String(i + 1).padStart(2, "0")}
              </span>
              <button
                type="button"
                onClick={() => onChange(files.filter((_, j) => j !== i))}
                className="absolute right-1 top-1 bg-sheet/90 px-1.5 text-xs opacity-0 transition-opacity hover:text-signal group-hover:opacity-100 focus:opacity-100"
                aria-label={`Remove ${f.name}`}
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
      {problems.length > 0 && (
        <ul role="alert" className="space-y-1 border-l-2 border-signal pl-3 text-sm text-rust">
          {problems.map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
