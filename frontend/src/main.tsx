import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";

createRoot(document.getElementById("root")!).render(<App />);

// Register Service Worker for PWA (vite-plugin-pwa)
if (import.meta.env.PROD) {
  // dynamic import to avoid affecting dev HMR
  import('virtual:pwa-register').then(({ registerSW }) => {
    registerSW({ immediate: true });
  }).catch(() => {
    // PWA registration failed, continue without it
  });
}