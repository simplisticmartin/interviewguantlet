import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  // "" prefix so VITE_API_BASE_URL is readable here as well as in the client bundle.
  const env = loadEnv(mode, ".", "");

  return {
    plugins: [react()],
    server: {
      port: 5173,
      // Proxy in dev so the browser sees a single origin and CORS never enters the
      // picture during local development.
      proxy: {
        "/api": {
          target: env.VITE_API_BASE_URL || "http://localhost:8000",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ""),
        },
      },
    },
    build: {
      outDir: "dist",
      sourcemap: true,
    },
  };
});
