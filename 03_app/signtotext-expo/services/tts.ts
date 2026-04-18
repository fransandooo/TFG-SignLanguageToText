import * as Speech from 'expo-speech';

import { PredictionResult } from '../types';

export function buildSpeechText(result: PredictionResult): string {
  if (result.mode === 'sequence') {
    return result.sequence_text.trim();
  }
  return result.predictions[0]?.word ?? '';
}

export function stopSpeech(): void {
  Speech.stop();
}

export function speakResult(result: PredictionResult): void {
  const text = buildSpeechText(result);
  if (!text) return;

  Speech.stop();
  Speech.speak(text, {
    language: 'en-US',
    rate: 0.95,
    pitch: 1.0,
  });
}
