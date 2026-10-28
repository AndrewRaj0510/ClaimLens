import type { Role, User } from "./types";

export const HOME: Record<Role, string> = { customer: "/claims", adjuster: "/adjuster" };

/**
 * Where to go after logging in. Only a path on this site: "?next=//evil.com"
 * or "?next=https://..." would otherwise make the login page an open redirect.
 * And only into the area for this user's role.
 */
export function safeNext(next: string | null, user: User): string {
  const home = HOME[user.role];
  if (!next || !next.startsWith("/") || next.startsWith("//") || next.includes("\\")) return home;
  return next === home || next.startsWith(`${home}/`) ? next : home;
}
