import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)"],
      },
      colors: {
        ink: "#09111f",
        paper: "#f5f1e8",
        ember: "#bc5a2b",
        mist: "#97b9c8",
        pine: "#123c38",
      },
      boxShadow: {
        panel: "0 12px 40px rgba(9, 17, 31, 0.12)",
      },
      backgroundImage: {
        weave: "radial-gradient(circle at top left, rgba(188, 90, 43, 0.2), transparent 30%), radial-gradient(circle at bottom right, rgba(18, 60, 56, 0.18), transparent 35%)"
      }
    },
  },
  plugins: [],
};

export default config;
