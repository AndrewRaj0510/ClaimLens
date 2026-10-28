import { Shell } from "@/components/Shell";

export default function CustomerLayout({ children }: { children: React.ReactNode }) {
  return <Shell role="customer">{children}</Shell>;
}
