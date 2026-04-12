import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { CircleIcon, FlipHorizontal2, Images, Timer } from 'lucide-react-native';
import { useRef, useState, useEffect } from 'react';
import {
  ActivityIndicator,
  Alert,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

import { RECORDING_DURATION_MS } from '../config';
import { predict } from '../services/api';
import { PredictResponse } from '../types';

interface Props {
  onResult: (response: PredictResponse) => void;
}

const PRE_COUNTDOWN_OPTIONS = [0, 3, 5] as const;
type PreCountdown = (typeof PRE_COUNTDOWN_OPTIONS)[number];

export default function RecordScreen({ onResult }: Props) {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<'back' | 'front'>('back');
  const [preCountdown, setPreCountdown] = useState<PreCountdown>(0);

  const [phase, setPhase] = useState<'idle' | 'pre' | 'recording' | 'processing'>('idle');
  const [preTick, setPreTick] = useState(0);
  const [recTick, setRecTick] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const cameraRef = useRef<CameraView>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => () => { clearTick(); }, []);

  function clearTick() {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }

  // ── Permission screen ──────────────────────────────────────────────────────
  if (!permission) return <View style={s.permissionContainer} />;

  if (!permission.granted) {
    return (
      <View style={s.permissionContainer}>
        <Text style={s.permissionText}>Camera access is required to record signs.</Text>
        <TouchableOpacity style={s.purpleBtn} onPress={requestPermission}>
          <Text style={s.purpleBtnText}>Grant permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // ── Handlers ───────────────────────────────────────────────────────────────

  async function handleRecord() {
    if (phase !== 'idle' || !cameraRef.current) return;
    setError(null);

    if (preCountdown > 0) {
      // Pre-recording countdown
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

    // Start recording
    setRecTick(Math.round(RECORDING_DURATION_MS / 1000));
    setPhase('recording');

    const stopTimer = setTimeout(() => {
      clearTick();
      cameraRef.current?.stopRecording();
    }, RECORDING_DURATION_MS);

    intervalRef.current = setInterval(() => {
      setRecTick((c) => (c > 1 ? c - 1 : 0));
    }, 1000);

    try {
      const video = await cameraRef.current.recordAsync({
        maxDuration: RECORDING_DURATION_MS / 1000,
      });

      clearTimeout(stopTimer);
      clearTick();

      if (!video?.uri) throw new Error('No video recorded');
      await sendAndPredict(video.uri);
    } catch (err: unknown) {
      clearTimeout(stopTimer);
      clearTick();
      setPhase('idle');
      const msg = err instanceof Error ? err.message : 'Unknown error';
      if (!msg.includes('aborted')) setError(msg);
    }
  }

  async function handleGallery() {
    if (phase !== 'idle') return;
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
    setPhase('processing');
    try {
      const response = await predict(uri);
      onResult(response);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      setError(msg);
    } finally {
      setPhase('idle');
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  const busy = phase !== 'idle';
  const isRecording = phase === 'recording';

  return (
    <View style={s.container}>
      {/* Camera + overlays */}
      <View style={s.cameraContainer}>
        <CameraView ref={cameraRef} style={s.camera} mode="video" facing={facing} />

        {/* Pre-countdown overlay */}
        {phase === 'pre' && (
          <View style={s.overlay}>
            <Text style={s.bigCountdown}>{preTick}</Text>
            <Text style={s.overlayHint}>Get ready…</Text>
          </View>
        )}

        {/* Flip button */}
        <TouchableOpacity
          style={[s.flipBtn, busy && s.btnDisabled]}
          onPress={() => setFacing((f) => (f === 'back' ? 'front' : 'back'))}
          disabled={busy}
        >
          <FlipHorizontal2 size={22} color="#fff" />
        </TouchableOpacity>
      </View>

      {/* Controls */}
      <View style={s.controls}>
        {/* Status */}
        {phase === 'processing' ? (
          <View style={s.statusRow}>
            <ActivityIndicator color="#fff" size="small" />
            <Text style={s.statusText}>  Analyzing…</Text>
          </View>
        ) : isRecording ? (
          <View style={s.statusRow}>
            <View style={s.recDot} />
            <Text style={s.statusText}>  Recording  {recTick}s</Text>
          </View>
        ) : error ? (
          <Text style={s.errorText}>{error}</Text>
        ) : (
          <Text style={s.hintText}>
            Tap to record {Math.round(RECORDING_DURATION_MS / 1000)}s clip
          </Text>
        )}

        {/* Timer selector */}
        <View style={s.timerRow}>
          <Timer size={14} color="#888" />
          <Text style={s.timerLabel}>Delay:</Text>
          {PRE_COUNTDOWN_OPTIONS.map((opt) => (
            <TouchableOpacity
              key={opt}
              style={[s.timerChip, preCountdown === opt && s.timerChipActive]}
              onPress={() => setPreCountdown(opt)}
              disabled={busy}
            >
              <Text style={[s.timerChipText, preCountdown === opt && s.timerChipTextActive]}>
                {opt === 0 ? 'Off' : `${opt}s`}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Bottom row: gallery — record — spacer */}
        <View style={s.bottomRow}>
          {/* Gallery button */}
          <TouchableOpacity
            style={[s.sideBtn, busy && s.btnDisabled]}
            onPress={handleGallery}
            disabled={busy}
          >
            <Images size={26} color="#fff" />
            <Text style={s.sideBtnLabel}>Gallery</Text>
          </TouchableOpacity>

          {/* Record button */}
          <TouchableOpacity
            style={[s.recordBtn, isRecording && s.recordBtnActive]}
            onPress={handleRecord}
            disabled={busy}
            activeOpacity={0.8}
          >
            {isRecording ? (
              <View style={s.stopIcon} />
            ) : (
              <CircleIcon size={38} color="#e53935" fill="#e53935" />
            )}
          </TouchableOpacity>

          {/* Spacer to balance layout */}
          <View style={s.sideBtn} />
        </View>

        <Text style={s.hintBottom}>Hold your hand up and sign a word</Text>
      </View>
    </View>
  );
}

// ── Styles ─────────────────────────────────────────────────────────────────────

const PURPLE = '#7c3aed';

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },

  // Permission
  permissionContainer: { flex: 1, backgroundColor: '#000', alignItems: 'center', justifyContent: 'center', padding: 24 },
  permissionText: { color: '#ccc', textAlign: 'center', marginBottom: 20, fontSize: 15 },
  purpleBtn: { backgroundColor: PURPLE, borderRadius: 10, paddingHorizontal: 24, paddingVertical: 12 },
  purpleBtnText: { color: '#fff', fontWeight: '600', fontSize: 15 },

  // Camera
  cameraContainer: { flex: 1, position: 'relative' },
  camera: { flex: 1 },
  flipBtn: {
    position: 'absolute', top: 16, right: 16,
    width: 44, height: 44, borderRadius: 22,
    backgroundColor: 'rgba(0,0,0,0.45)',
    alignItems: 'center', justifyContent: 'center',
  },
  btnDisabled: { opacity: 0.3 },

  // Pre-countdown overlay
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.65)',
    alignItems: 'center', justifyContent: 'center',
  },
  bigCountdown: { color: '#fff', fontSize: 96, fontWeight: '800' },
  overlayHint: { color: '#aaa', fontSize: 18, marginTop: 8 },

  // Controls bar
  controls: {
    backgroundColor: '#000',
    paddingTop: 20, paddingBottom: 28, paddingHorizontal: 24,
    alignItems: 'center', gap: 14,
  },

  // Status
  statusRow: { flexDirection: 'row', alignItems: 'center' },
  recDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: '#e53935' },
  statusText: { color: '#fff', fontSize: 16 },
  hintText: { color: '#888', fontSize: 14 },
  hintBottom: { color: '#555', fontSize: 12 },
  errorText: { color: '#ff6b6b', textAlign: 'center', fontSize: 13 },

  // Timer selector
  timerRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  timerLabel: { color: '#888', fontSize: 13, marginLeft: 4 },
  timerChip: {
    paddingHorizontal: 10, paddingVertical: 4,
    borderRadius: 12, borderWidth: 1, borderColor: '#333',
  },
  timerChipActive: { borderColor: PURPLE, backgroundColor: PURPLE + '33' },
  timerChipText: { color: '#888', fontSize: 13 },
  timerChipTextActive: { color: '#c4b5fd' },

  // Bottom row
  bottomRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', width: '100%' },
  sideBtn: { width: 64, alignItems: 'center', gap: 4 },
  sideBtnLabel: { color: '#888', fontSize: 11 },

  // Record button
  recordBtn: {
    width: 72, height: 72, borderRadius: 36,
    borderWidth: 3, borderColor: '#fff',
    alignItems: 'center', justifyContent: 'center',
  },
  recordBtnActive: { borderColor: '#e53935' },
  stopIcon: { width: 24, height: 24, borderRadius: 4, backgroundColor: '#fff' },
});
