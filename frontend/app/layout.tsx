import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Northstar AI | Business assistant",
  description: "A focused AI assistant for modern teams.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
