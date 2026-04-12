import { PREDICT_URL, HEALTH_URL } from '../config';
import { PredictResponse } from '../types';

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(HEALTH_URL, { signal: AbortSignal.timeout(5000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function predict(videoUri: string): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append('video', {
    uri: videoUri,
    name: 'recording.mp4',
    type: 'video/mp4',
  } as unknown as Blob);

  const res = await fetch(PREDICT_URL, {
    method: 'POST',
    body: formData,
    headers: { Accept: 'application/json' },
  });

  const json = await res.json();

  if (!res.ok) {
    throw new Error(json?.detail ?? `Server error ${res.status}`);
  }

  return json as PredictResponse;
}
