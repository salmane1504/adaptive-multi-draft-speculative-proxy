import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const api = "http://127.0.0.1:8000";

export default defineConfig({
  // Built files are served by FastAPI under /dashboard/.
  base: "/dashboard/",
  plugins: [react()],
  // Recharts makes a single ~600 kB bundle; fine for a local dashboard.
  build: { chunkSizeWarningLimit: 1000 },
  server: {
    proxy: { "/v1": api, "/metrics": api, "/health": api },
  },
});
