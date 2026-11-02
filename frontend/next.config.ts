import type { NextConfig } from "next";

// Content-Security-Policy: everything from this origin only. Even if a script
// got injected, it couldn't load code from elsewhere or send data anywhere
// (connect-src, form-action), and no other site can frame these pages.
// 'unsafe-inline' stays for scripts and styles: Next.js inlines its hydration
// data, and server-rendered style attributes count as inline styles. Removing
// it needs a per-request nonce (set in proxy.ts), which makes every page
// render dynamically. Dev mode also needs eval, for React's error overlay.
const dev = process.env.NODE_ENV === "development";
const CSP = [
  "default-src 'self'",
  `script-src 'self' 'unsafe-inline'${dev ? " 'unsafe-eval'" : ""}`,
  "style-src 'self' 'unsafe-inline'",
  // blob: for the previews of photos being picked, before upload.
  "img-src 'self' data: blob:",
  "font-src 'self'",
  "connect-src 'self'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
].join("; ");

const config: NextConfig = {
  // A self-contained server in .next/standalone: the Docker image ships that
  // folder instead of node_modules.
  output: "standalone",
  poweredByHeader: false,
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "same-origin" },
          { key: "Content-Security-Policy", value: CSP },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        ],
      },
    ];
  },
};

export default config;
