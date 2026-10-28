// Browser-side client for the gateway, through this app's /api proxy
// (app/api/[...path]/route.ts). Same origin, so the session cookie goes along
// automatically and there's no CORS.

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

type Detail = string | { msg: string; loc?: (string | number)[] }[] | undefined;

/** FastAPI errors are a string, or a list of validation errors. */
export function errorMessage(detail: Detail, status: number): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail
      .map((d) => {
        const field = d.loc?.filter((p) => p !== "body").at(-1);
        return field ? `${field}: ${d.msg}` : d.msg;
      })
      .join("; ");
  }
  return status === 429 ? "Too many attempts. Try again later." : `Request failed (${status})`;
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    credentials: "same-origin",
    headers: { Accept: "application/json", ...init.headers },
  });
  if (!res.ok) {
    let detail: Detail;
    try {
      detail = (await res.json()).detail;
    } catch {
      detail = undefined;
    }
    throw new ApiError(res.status, errorMessage(detail, res.status));
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export function postJson<T>(path: string, body: unknown): Promise<T> {
  return api<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function photoUrl(claimId: string, index: number): string {
  return `/api/claims/${claimId}/photos/${index}`;
}
