import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: { rollupOptions: { input: ["index.html", "workbench.html"] } },
  server: {
    port: 5173,
    proxy: {
      "/workbench-api": "http://127.0.0.1:8194",
      "/api": (globalThis as unknown as { process?: { env: Record<string,string> } }).process?.env.VITE_API_PROXY_TARGET || "http://localhost:8190",
    },
  },
});
