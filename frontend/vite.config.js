// frontend/vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Proxy local dev requests tới backend để tránh CORS khi dev
  server: {
    proxy: {
      "/predict": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
