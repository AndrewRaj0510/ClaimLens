import { Shell } from "@/components/Shell";

export default function AdjusterLayout({ children }: { children: React.ReactNode }) {
  return <Shell role="adjuster">{children}</Shell>;
}
