import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)", panel: "var(--panel)", panel2: "var(--panel2)",
        line: "var(--line)", fg: "var(--fg)", mut: "var(--mut)",
        accent: "var(--accent)", ok: "var(--ok)", warn: "var(--warn)",
        danger: "var(--danger)",
      },
    },
  },
  plugins: [],
};
export default config;
