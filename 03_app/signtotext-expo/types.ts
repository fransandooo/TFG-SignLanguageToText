export type CaptureMode = 'isolated' | 'sequence';

export interface Prediction {
  word: string;
  confidence: number;
  rank: number;
}

export interface ModelSummary {
  model_id: string;
  label: string;
  architecture: string;
  num_classes: number;
  dataset_name: string;
  val_top1: number;
  experimental: boolean;
  available_attention: boolean;
}

export interface ModelsResponse {
  default_model_id: string;
  models: ModelSummary[];
}

interface BaseResult {
  mode: CaptureMode;
  model_id: string;
  model_label: string;
  architecture: string;
  num_classes: number;
  frames_processed: number;
  processing_time_ms: number;
}

export interface PredictResponse extends BaseResult {
  mode: 'isolated';
  predictions: Prediction[];
  detection_rate: number;
  attention_weights: number[] | null;
}

export interface SequenceSegment {
  word: string;
  start_ms: number;
  end_ms: number;
  confidence: number;
  windows: number;
}

export interface SequencePredictResponse extends BaseResult {
  mode: 'sequence';
  sequence_text: string;
  segments: SequenceSegment[];
  window_count: number;
  avg_detection_rate: number;
}

export type PredictionResult = PredictResponse | SequencePredictResponse;
