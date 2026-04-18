import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import {
  CircleIcon,
  FlipHorizontal2,
  Images,
  Mic2,
  MicOff,
  Timer,
  Waves,
} from 'lucide-react-native';
import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

import {
  RECORDING_DURATION_SEQUENCE_MS,
  RECORDING_DURATION_SINGLE_MS,
} from '../config';
import { predict, predictSequence } from '../services/api';
import {
  CaptureMode,
  ModelSummary,
  PredictionResult,
} from '../types';

interface Props {
  captureMode: CaptureMode;
  models: ModelSummary[];
  modelsError: string | null;
  onModeChange: (mode: CaptureMode) => void;
  onResult: (response: PredictionResult) => void;
  onSelectModel: (modelId: string) => void;
  onToggleTts: () => void;
  selectedModel: ModelSummary | null;
  ttsEnabled: boolean;
}

const PRE_COUNTDOWN_OPTIONS = [0, 3, 5] as const;
type PreCountdown = (typeof PRE_COUNTDOWN_OPTIONS)[number];

export default function RecordScreen({
  captureMode,
  models,
  modelsError,
  onModeChange,
  onResult,
  onSelectModel,
  onToggleTts,
  selectedModel,
  ttsEnabled,
}: Props) {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<'back' | 'front'>('back');
  const [preCountdown, setPreCountdown] = useState<PreCountdown>(0);
  const [phase, setPhase] = useState<'idle' | 'pre' | 'recording' | 'processing'>('idle');
  const [preTick, setPreTick] = useState(0);
  const [recTick, setRecTick] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const cameraRef = useRef<CameraView>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => () => {
    clearTick();
  }, []);

  function clearTick() {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }

  if (!permission) return <View style={s.permissionContainer} />;

  if (!permission.granted) {
    return (
      <View style={s.permissionContainer}>
        <Text style={s.permissionText}>Camera access is required to record signs.</Text>
        <TouchableOpacity style={s.primaryBtn} onPress={requestPermission}>
          <Text style={s.primaryBtnText}>Grant permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const busy = phase !== 'idle';
  const isRecording = phase === 'recording';
  const durationMs =
    captureMode === 'sequence' ? RECORDING_DURATION_SEQUENCE_MS : RECORDING_DURATION_SINGLE_MS;

  async function handleRecord() {
    if (phase !== 'idle' || !cameraRef.current || !selectedModel) return;
    setError(null);

    if (preCountdown > 0) {
      setPreTick(preCountdown);
      setPhase('pre');

      await new Promise<void>((resolve) => {
        let remaining = preCountdown;
        intervalRef.current = setInterval(() => {
          remaining -= 1;
          setPreTick(remaining);
          if (remaining <= 0) {
            clearTick();
            resolve();
          }
        }, 1000);
      });
    }

    setRecTick(Math.round(durationMs / 1000));
    setPhase('recording');

    const stopTimer = setTimeout(() => {
      clearTick();
      cameraRef.current?.stopRecording();
    }, durationMs);

    intervalRef.current = setInterval(() => {
      setRecTick((count) => (count > 1 ? count - 1 : 0));
    }, 1000);

    try {
      const video = await cameraRef.current.recordAsync({
        maxDuration: durationMs / 1000,
      });

      clearTimeout(stopTimer);
      clearTick();

      if (!video?.uri) throw new Error('No video recorded');
      await sendAndPredict(video.uri);
    } catch (err: unknown) {
      clearTimeout(stopTimer);
      clearTick();
      setPhase('idle');
      const message = err instanceof Error ? err.message : 'Unknown error';
      if (!message.includes('aborted')) setError(message);
    }
  }

  async function handleGallery() {
    if (phase !== 'idle' || !selectedModel) return;
    setError(null);

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['videos'],
      allowsEditing: false,
      quality: 1,
    });

    if (result.canceled || !result.assets?.[0]) return;
    await sendAndPredict(result.assets[0].uri);
  }

  async function sendAndPredict(uri: string) {
    if (!selectedModel) {
      setError('No model selected');
      return;
    }

    setPhase('processing');
    try {
      const response =
        captureMode === 'sequence'
          ? await predictSequence(uri, selectedModel.model_id)
          : await predict(uri, selectedModel.model_id);
      onResult(response);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(message);
    } finally {
      setPhase('idle');
    }
  }

  return (
    <View style={s.container}>
      <View style={s.cameraContainer}>
        <CameraView ref={cameraRef} style={s.camera} mode="video" facing={facing} />

        {phase === 'pre' && (
          <View style={s.overlay}>
            <Text style={s.bigCountdown}>{preTick}</Text>
            <Text style={s.overlayHint}>Get ready...</Text>
          </View>
        )}

        <TouchableOpacity
          style={[s.flipBtn, busy && s.btnDisabled]}
          onPress={() => setFacing((current) => (current === 'back' ? 'front' : 'back'))}
          disabled={busy}
        >
          <FlipHorizontal2 size={22} color="#fff" />
        </TouchableOpacity>
      </View>

      <ScrollView style={s.controls} contentContainerStyle={s.controlsContent}>
        {phase === 'processing' ? (
          <View style={s.statusRow}>
            <ActivityIndicator color="#fff" size="small" />
            <Text style={s.statusText}>  Analyzing...</Text>
          </View>
        ) : isRecording ? (
          <View style={s.statusRow}>
            <View style={s.recDot} />
            <Text style={s.statusText}>  Recording {recTick}s</Text>
          </View>
        ) : error ? (
          <Text style={s.errorText}>{error}</Text>
        ) : (
          <Text style={s.hintText}>
            {captureMode === 'sequence'
              ? 'Record a continuous phrase-like sequence'
              : `Tap to record ${Math.round(durationMs / 1000)}s isolated sign clip`}
          </Text>
        )}

        <View style={s.section}>
          <Text style={s.sectionLabel}>Capture mode</Text>
          <View style={s.modeRow}>
            <TouchableOpacity
              style={[s.modeChip, captureMode === 'isolated' && s.modeChipActive]}
              onPress={() => onModeChange('isolated')}
              disabled={busy}
            >
              <Text style={[s.modeChipText, captureMode === 'isolated' && s.modeChipTextActive]}>
                Single sign
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[s.modeChip, captureMode === 'sequence' && s.modeChipActive]}
              onPress={() => onModeChange('sequence')}
              disabled={busy}
            >
              <Waves size={14} color={captureMode === 'sequence' ? '#fff' : '#aaa'} />
              <Text style={[s.modeChipText, captureMode === 'sequence' && s.modeChipTextActive]}>
                Continuous
              </Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={s.section}>
          <View style={s.inlineHeader}>
            <Text style={s.sectionLabel}>Speech output</Text>
            <TouchableOpacity style={s.ttsToggle} onPress={onToggleTts}>
              {ttsEnabled ? <Mic2 size={16} color="#fff" /> : <MicOff size={16} color="#aaa" />}
              <Text style={s.ttsText}>{ttsEnabled ? 'Auto voice on' : 'Auto voice off'}</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={s.section}>
          <Text style={s.sectionLabel}>Model</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={s.modelRow}>
            {models.map((model) => {
              const isSelected = selectedModel?.model_id === model.model_id;
              return (
                <TouchableOpacity
                  key={model.model_id}
                  style={[s.modelCard, isSelected && s.modelCardActive]}
                  onPress={() => onSelectModel(model.model_id)}
                  disabled={busy}
                >
                  <Text style={[s.modelTitle, isSelected && s.modelTitleActive]}>{model.label}</Text>
                  <Text style={s.modelMeta}>{model.num_classes} classes</Text>
                  <Text style={s.modelMeta}>Val Top-1 {model.val_top1.toFixed(1)}%</Text>
                  {model.experimental ? <Text style={s.experimental}>Experimental</Text> : null}
                </TouchableOpacity>
              );
            })}
          </ScrollView>
          {!models.length && !modelsError ? (
            <Text style={s.loadingText}>Loading models from server...</Text>
          ) : null}
        </View>

        <View style={s.timerRow}>
          <Timer size={14} color="#888" />
          <Text style={s.timerLabel}>Delay:</Text>
          {PRE_COUNTDOWN_OPTIONS.map((option) => (
            <TouchableOpacity
              key={option}
              style={[s.timerChip, preCountdown === option && s.timerChipActive]}
              onPress={() => setPreCountdown(option)}
              disabled={busy}
            >
              <Text style={[s.timerChipText, preCountdown === option && s.timerChipTextActive]}>
                {option === 0 ? 'Off' : `${option}s`}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <View style={s.bottomRow}>
          <TouchableOpacity
            style={[s.sideBtn, (busy || !selectedModel) && s.btnDisabled]}
            onPress={handleGallery}
            disabled={busy || !selectedModel}
          >
            <Images size={26} color="#fff" />
            <Text style={s.sideBtnLabel}>Gallery</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[s.recordBtn, isRecording && s.recordBtnActive, !selectedModel && s.btnDisabled]}
            onPress={handleRecord}
            disabled={busy || !selectedModel}
            activeOpacity={0.8}
          >
            {isRecording ? (
              <View style={s.stopIcon} />
            ) : (
              <CircleIcon size={38} color="#e53935" fill="#e53935" />
            )}
          </TouchableOpacity>

          <View style={s.sideBtn}>
            <Text style={s.sideHint}>{Math.round(durationMs / 1000)}s</Text>
          </View>
        </View>

        <Text style={s.hintBottom}>
          {captureMode === 'sequence'
            ? 'Continuous mode uses sliding windows and merges repeated detections.'
            : 'Keep the signing hand visible and centered.'}
        </Text>
      </ScrollView>
    </View>
  );
}

const PURPLE = '#7c3aed';

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  permissionContainer: {
    flex: 1,
    backgroundColor: '#000',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  permissionText: { color: '#ccc', textAlign: 'center', marginBottom: 20, fontSize: 15 },
  primaryBtn: { backgroundColor: PURPLE, borderRadius: 10, paddingHorizontal: 24, paddingVertical: 12 },
  primaryBtnText: { color: '#fff', fontWeight: '600', fontSize: 15 },
  cameraContainer: { flex: 1, margin: 16, borderRadius: 20, overflow: 'hidden', backgroundColor: '#111' },
  camera: { flex: 1 },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0,0,0,0.4)',
  },
  bigCountdown: { color: '#fff', fontSize: 72, fontWeight: '800' },
  overlayHint: { color: '#ddd', fontSize: 16, marginTop: 8 },
  flipBtn: {
    position: 'absolute',
    top: 16,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.45)',
    borderRadius: 24,
    padding: 12,
  },
  controls: { flex: 1, paddingHorizontal: 16 },
  controlsContent: { paddingBottom: 24 },
  statusRow: { flexDirection: 'row', alignItems: 'center', minHeight: 20, marginBottom: 14 },
  statusText: { color: '#fff', fontSize: 14 },
  hintText: { color: '#bbb', fontSize: 14, marginBottom: 14 },
  errorText: { color: '#ef5350', fontSize: 14, marginBottom: 14 },
  section: { marginBottom: 16 },
  sectionLabel: { color: '#fff', fontSize: 13, fontWeight: '700', marginBottom: 8 },
  inlineHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  modeRow: { flexDirection: 'row', gap: 10 },
  modeChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#151515',
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: '#232323',
  },
  modeChipActive: { backgroundColor: '#2d1b69', borderColor: '#7c3aed' },
  modeChipText: { color: '#aaa', fontSize: 13, fontWeight: '600' },
  modeChipTextActive: { color: '#fff' },
  ttsToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#151515',
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  ttsText: { color: '#ddd', fontSize: 12, fontWeight: '600' },
  modelRow: { gap: 10 },
  modelCard: {
    width: 170,
    backgroundColor: '#151515',
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: '#232323',
  },
  modelCardActive: { borderColor: '#7c3aed', backgroundColor: '#1d133b' },
  modelTitle: { color: '#fff', fontSize: 14, fontWeight: '700', marginBottom: 6 },
  modelTitleActive: { color: '#fff' },
  modelMeta: { color: '#999', fontSize: 12, marginBottom: 2 },
  experimental: { color: '#f5b942', fontSize: 11, marginTop: 6, fontWeight: '700' },
  loadingText: { color: '#777', fontSize: 12, marginTop: 8 },
  timerRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 18 },
  timerLabel: { color: '#888', fontSize: 13, marginRight: 2 },
  timerChip: {
    borderRadius: 18,
    paddingHorizontal: 12,
    paddingVertical: 7,
    backgroundColor: '#151515',
  },
  timerChipActive: { backgroundColor: PURPLE },
  timerChipText: { color: '#aaa', fontSize: 12, fontWeight: '600' },
  timerChipTextActive: { color: '#fff' },
  bottomRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  sideBtn: { width: 80, alignItems: 'center', justifyContent: 'center', gap: 6 },
  sideBtnLabel: { color: '#fff', fontSize: 12, fontWeight: '600' },
  sideHint: { color: '#777', fontSize: 13, fontWeight: '600' },
  recordBtn: {
    width: 84,
    height: 84,
    borderRadius: 42,
    borderWidth: 3,
    borderColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#111',
  },
  recordBtnActive: { borderColor: '#e53935' },
  stopIcon: { width: 28, height: 28, borderRadius: 6, backgroundColor: '#e53935' },
  hintBottom: { color: '#666', textAlign: 'center', fontSize: 12, marginTop: 18, lineHeight: 18 },
  recDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#e53935' },
  btnDisabled: { opacity: 0.45 },
});
