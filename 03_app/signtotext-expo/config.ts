/**
 * Server configuration.
 * - Emulator Android : 10.0.2.2
 * - iPhone on same WiFi: your laptop's LAN IP, e.g. "192.168.1.42"
 *   Find it with:  ip a | grep "inet " | grep -v 127
 */
const SERVER_HOST = '172.20.10.12'; // ← change to your laptop's local IP
const SERVER_PORT = 8000;

export const BASE_URL = `http://${SERVER_HOST}:${SERVER_PORT}`;
export const PREDICT_URL = `${BASE_URL}/predict`;
export const HEALTH_URL = `${BASE_URL}/health`;
export const CLASSES_URL = `${BASE_URL}/classes`;

export const RECORDING_DURATION_MS = 3000;
