import {
  HEALTH_URL,
  MODELS_URL,
  PREDICT_SEQUENCE_URL,
  PREDICT_URL,
} from '../config';
import {
  ModelsResponse,
  PredictResponse,
  SequencePredictResponse,
} from '../types';

const REQUEST_TIMEOUT_MS = 5000;

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init?: RequestInit,
  timeoutMs: number = REQUEST_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(input, {
      ...init,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

async function readJsonSafe(res: Response): Promise<any> {
  const text = await res.text();
  if (!text) return null;

  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetchWithTimeout(HEALTH_URL);
    return res.ok;
  } catch {
    return false;
  }
}

export async function fetchModels(): Promise<ModelsResponse> {
  const res = await fetchWithTimeout(MODELS_URL);
  const json = await readJsonSafe(res);

  if (!res.ok) {
    throw new Error(json?.detail ?? `Server error ${res.status}`);
  }

  return json as ModelsResponse;
}

function buildVideoFormData(
  videoUri: string,
  modelId: string,
  extraFields?: Record<string, string>,
): FormData {
  const formData = new FormData();
  formData.append('video', {
    uri: videoUri,
    name: 'recording.mp4',
    type: 'video/mp4',
  } as unknown as Blob);
  formData.append('model_id', modelId);
  if (extraFields) {
    Object.entries(extraFields).forEach(([key, value]) => {
      formData.append(key, value);
    });
  }
  return formData;
}

async function postVideo<T>(
  url: string,
  videoUri: string,
  modelId: string,
  extraFields?: Record<string, string>,
): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    body: buildVideoFormData(videoUri, modelId, extraFields),
    headers: { Accept: 'application/json' },
  });

  const json = await readJsonSafe(res);
  if (!res.ok) {
    throw new Error(json?.detail ?? `Server error ${res.status}`);
  }

  return json as T;
}

export function predict(videoUri: string, modelId: string): Promise<PredictResponse> {
  return postVideo<PredictResponse>(PREDICT_URL, videoUri, modelId);
}

export function predictSequence(videoUri: string, modelId: string): Promise<SequencePredictResponse> {
  return postVideo<SequencePredictResponse>(PREDICT_SEQUENCE_URL, videoUri, modelId, {
    debug: 'true',
    save_debug: 'true',
  });
}
