import type { NextConfig } from "next";
import { dirname } from "path";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  outputFileTracingRoot: dirname(__filename),
  turbopack: {
    root: dirname(__filename),
  },
};

export default nextConfig;
