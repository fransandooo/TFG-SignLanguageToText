import { StatusBar } from 'expo-status-bar';
import { useEffect, useState } from 'react';
import { SafeAreaView, StyleSheet, Text, View } from 'react-native';

import RecordScreen from './screens/RecordScreen';
import ResultScreen from './screens/ResultScreen';
import { fetchModels } from './services/api';
import { speakResult, stopSpeech } from './services/tts';
import {
  CaptureMode,
  ModelSummary,
  PredictionResult,
} from './types';

export default function App() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [captureMode, setCaptureMode] = useState<CaptureMode>('isolated');
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  useEffect(() => {
    let alive = true;

    async function loadModels() {
      try {
        const response = await fetchModels();
        if (!alive) return;
        setModels(response.models);
        setSelectedModelId((current) => current ?? response.default_model_id);
        setModelsError(null);
      } catch (err: unknown) {
        if (!alive) return;
        const message = err instanceof Error ? err.message : 'Unable to load models';
        setModelsError(message);
      }
    }

    loadModels();
    return () => {
      alive = false;
      stopSpeech();
    };
  }, []);

  useEffect(() => {
    if (!result || !ttsEnabled) return;
    speakResult(result);
  }, [result, ttsEnabled]);

  const selectedModel = models.find((model) => model.model_id === selectedModelId) ?? null;

  function handleToggleTts() {
    setTtsEnabled((current) => {
      const next = !current;
      if (!next) stopSpeech();
      return next;
    });
  }

  function handleReset() {
    stopSpeech();
    setResult(null);
  }

  function handleReplay() {
    if (result) speakResult(result);
  }

  const subtitle = selectedModel
    ? `${selectedModel.label} · ${captureMode === 'sequence' ? 'Continuous' : 'Single sign'}`
    : 'Loading models...';

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar style="light" />

      <View style={styles.header}>
        <Text style={styles.title}>Sign to Text</Text>
        <Text style={styles.subtitle}>{subtitle}</Text>
        {modelsError ? <Text style={styles.error}>{modelsError}</Text> : null}
      </View>

      {result === null ? (
        <RecordScreen
          captureMode={captureMode}
          models={models}
          modelsError={modelsError}
          onModeChange={setCaptureMode}
          onResult={setResult}
          onSelectModel={setSelectedModelId}
          onToggleTts={handleToggleTts}
          selectedModel={selectedModel}
          ttsEnabled={ttsEnabled}
        />
      ) : (
        <ResultScreen
          onReplay={handleReplay}
          onReset={handleReset}
          onToggleTts={handleToggleTts}
          response={result}
          ttsEnabled={ttsEnabled}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#222',
  },
  title: { color: '#fff', fontSize: 20, fontWeight: '700' },
  subtitle: { color: '#666', fontSize: 12, marginTop: 2 },
  error: { color: '#ef5350', fontSize: 12, marginTop: 6 },
});
