"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createContext, useContext, useEffect, useState } from "react";

import { ApiError, api } from "@/lib/api";
import { HOME } from "@/lib/nav";
import type { Role, User } from "@/lib/types";

const UserContext = createContext<User | null>(null);

export function useUser(): User {
  const user = useContext(UserContext);
  if (!user) throw new Error("useUser outside <Shell>");
  return user;
}

/**
 * Loads the session and renders the page only for the right role. This is
 * navigation, not security: the gateway checks the session and role on every
 * request, whatever the browser shows.
 */
export function Shell({ role, children }: { role: Role; children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    api<User>("/auth/me")
      .then((me) => (me.role === role ? setUser(me) : router.replace(HOME[me.role])))
      .catch((e) => {
        if (e instanceof ApiError && e.status === 401) {
          router.replace(`/login?next=${encodeURIComponent(pathname)}`);
        }
      });
  }, [role, router, pathname]);

  async function logout() {
    await api("/auth/logout", { method: "POST" }).catch(() => undefined);
    router.replace("/login");
  }

  if (!user) {
    return (
      <div className="grid min-h-dvh place-items-center">
        <span className="label animate-pulse">Opening file…</span>
      </div>
    );
  }

  const nav =
    role === "adjuster"
      ? [{ href: "/adjuster", label: "Queue" }]
      : [
          { href: "/claims", label: "My claims" },
          { href: "/claims/new", label: "File a claim" },
        ];

  return (
    <UserContext.Provider value={user}>
      <header className="sticky top-0 z-30 border-b border-ink bg-paper/90 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center gap-8 px-5">
          <Link href={HOME[role]} className="flex items-baseline gap-2">
            <Wordmark />
            <span className="label hidden sm:inline">
              {role === "adjuster" ? "Adjuster desk" : "Claims"}
            </span>
          </Link>
          <nav className="flex gap-5 text-sm">
            {nav.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={
                  pathname === item.href
                    ? "border-b-2 border-signal pb-0.5 font-medium"
                    : "pb-0.5 text-graphite hover:text-ink"
                }
              >
                {item.label}
              </Link>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-4 text-sm">
            <span className="hidden text-graphite sm:inline">{user.name}</span>
            <button type="button" onClick={logout} className="label cursor-pointer hover:text-signal">
              Log out
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-5 py-8">{children}</main>
    </UserContext.Provider>
  );
}

export function Wordmark({ large = false }: { large?: boolean }) {
  return (
    <span className={`font-serif tracking-tight ${large ? "text-5xl" : "text-2xl"}`}>
      Claim<span className="italic text-signal">Lens</span>
    </span>
  );
}
