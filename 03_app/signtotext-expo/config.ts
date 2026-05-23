/**
 * Server configuration.
 * Set EXPO_PUBLIC_SERVER_HOST to the FastAPI host visible from the device.
 * Examples: 10.0.2.2 for Android emulator, or a LAN IP for a physical phone.
 */
const SERVER_HOST = process.env.EXPO_PUBLIC_SERVER_HOST ?? 'localhost';
const SERVER_PORT = process.env.EXPO_PUBLIC_SERVER_PORT ?? '8000';

export const BASE_URL = `http://${SERVER_HOST}:${SERVER_PORT}`;
export const PREDICT_URL = `${BASE_URL}/predict`;
export const PREDICT_SEQUENCE_URL = `${BASE_URL}/predict-sequence`;
export const HEALTH_URL = `${BASE_URL}/health`;
export const CLASSES_URL = `${BASE_URL}/classes`;
export const MODELS_URL = `${BASE_URL}/models`;

export const RECORDING_DURATION_SINGLE_MS = 3000;
export const RECORDING_DURATION_SEQUENCE_MS = 8000;
