/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The browser talks to Next, Next proxies to FastAPI. This keeps the API
  // origin out of the client bundle and avoids CORS entirely in production.
  async rewrites() {
    const api = process.env.OPSPILOT_API_URL ?? "http://127.0.0.1:8000";
    return [
      { source: "/api/v1/:path*", destination: `${api}/api/v1/:path*` },
      { source: "/metrics", destination: `${api}/metrics` },
      { source: "/ready", destination: `${api}/ready` },
    ];
  },
};
export default nextConfig;
