import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "PaperWeave",
  description: "GraphRAG benchmark dashboard for scientific research papers.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
