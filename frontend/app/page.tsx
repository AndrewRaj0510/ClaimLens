"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { api } from "@/lib/api";
import { HOME } from "@/lib/nav";
import type { User } from "@/lib/types";

export default function Home() {
  const router = useRouter();
  useEffect(() => {
    api<User>("/auth/me")
      .then((me) => router.replace(HOME[me.role]))
      .catch(() => router.replace("/login"));
  }, [router]);
  return null;
}
