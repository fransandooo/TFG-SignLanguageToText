import { StatusBar } from 'expo-status-bar';
import { useState } from 'react';
import { SafeAreaView, StyleSheet, Text, View } from 'react-native';

import RecordScreen from './screens/RecordScreen';
import ResultScreen from './screens/ResultScreen';
import { PredictResponse } from './types';

export default function App() {
  const [result, setResult] = useState<PredictResponse | null>(null);

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar style="light" />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Sign to Text</Text>
        <Text style={styles.subtitle}>ASL · 30 words</Text>
      </View>

      {/* Screens */}
      {result === null ? (
        <RecordScreen onResult={setResult} />
      ) : (
        <ResultScreen response={result} onReset={() => setResult(null)} />
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
});
