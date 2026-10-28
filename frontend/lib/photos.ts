// Mirrors the gateway's upload checks (services/gateway/common.py and
// claims.py), so customers hear about a bad file before uploading it. The
// gateway still checks everything: this is convenience, not security.

export const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
export const MAX_PHOTOS = 10;
export const MAX_MB = 10;

export function photoProblems(files: File[]): string[] {
  const problems: string[] = [];
  if (files.length === 0) problems.push("Add at least one photo.");
  if (files.length > MAX_PHOTOS) problems.push(`Up to ${MAX_PHOTOS} photos per upload.`);
  for (const f of files) {
    if (!ALLOWED_TYPES.includes(f.type)) problems.push(`${f.name}: use a JPEG, PNG or WebP photo.`);
    else if (f.size > MAX_MB * 1024 * 1024) problems.push(`${f.name}: larger than ${MAX_MB} MB.`);
  }
  return problems;
}
