export interface Prediction {
  word: string;
  confidence: number;
  rank: number;
}

export interface PredictResponse {
  predictions: Prediction[];
  detection_rate: number;
  frames_processed: number;
  attention_weights: number[];   // 64 values, one per sampled frame
  processing_time_ms: number;
}
