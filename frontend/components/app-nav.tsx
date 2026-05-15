import Link from "next/link";

export function AppNav({ active }: { active: "home" | "evaluation" }) {
  const linkBase = "rounded-full border px-4 py-2 text-sm font-medium transition";

  return (
    <nav className="flex flex-wrap items-center gap-3 text-sm">
      <Link
        href="/"
        className={`${linkBase} ${active === "home" ? "border-ink bg-ink text-paper" : "border-ink/10 bg-paper text-ink hover:border-ink/30"}`}
      >
        Home
      </Link>
      <Link
        href="/evaluation"
        className={`${linkBase} ${active === "evaluation" ? "border-ember bg-ember text-paper" : "border-ink/10 bg-paper text-ink hover:border-ember/40"}`}
      >
        Evaluation Dashboard
      </Link>
    </nav>
  );
}