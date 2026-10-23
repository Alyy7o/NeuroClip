/**
 * Backend base URL from Vite env.
 * Always trim — trailing spaces in .env break fetch ("Failed to fetch").
 */
export const API_BASE = String(
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8040'
).trim().replace(/\/+$/, '');

/** Headers needed for ngrok free tunnels (skip interstitial HTML). */
export const NGROK_HEADERS: HeadersInit = {
  'ngrok-skip-browser-warning': 'true',
};
