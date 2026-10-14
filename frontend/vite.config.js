import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  // Proxy targets the FastAPI app. Default 8899 avoids collisions when another
  // process already owns 8000. Override: VITE_API_PROXY=http://127.0.0.1:8000 in frontend/.env
  const apiTarget = env.VITE_API_PROXY || "http://127.0.0.1:8899";

  return {
    plugins: [react()],
    build: {
      outDir: "../static/dist",
      emptyOutDir: true,
    },
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target: apiTarget,
          changeOrigin: true,
          ws: true,
        },
        "/output": {
          target: apiTarget,
          changeOrigin: true,
        },
      },
    },
  };
});
