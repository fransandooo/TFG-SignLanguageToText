import { Brain, Clock, Film, Hand, RotateCcw, Trophy } from 'lucide-react-native';
import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

import { PredictResponse, Prediction } from '../types';

interface Props {
  response: PredictResponse;
  onReset: () => void;
}

const PURPLE = '#7c3aed';
const PURPLE_LIGHT = '#c4b5fd';

export default function ResultScreen({ response, onReset }: Props) {
  const top = response.predictions[0];
  const rest = response.predictions.slice(1);

  return (
    <ScrollView style={s.scroll} contentContainerStyle={s.content} showsVerticalScrollIndicator={false}>

      {/* ── Top prediction ── */}
      <View style={s.heroCard}>
        <Text style={s.heroLabel}>Detected Sign</Text>
        <Text style={s.heroWord}>{top.word.toUpperCase()}</Text>
        <ConfidenceBadge confidence={top.confidence} />
      </View>

      {/* ── Stats grid ── */}
      <View style={s.statsGrid}>
        <StatCard icon={<Film size={18} color={PURPLE_LIGHT} />} label="Frames" value={String(response.frames_processed)} />
        <StatCard icon={<Hand size={18} color={PURPLE_LIGHT} />} label="Hand detected" value={`${Math.round(response.detection_rate * 100)}%`} />
        <StatCard icon={<Clock size={18} color={PURPLE_LIGHT} />} label="Processing" value={response.processing_time_ms != null ? `${response.processing_time_ms.toFixed(0)} ms` : '—'} />
        <StatCard icon={<Brain size={18} color={PURPLE_LIGHT} />} label="Model" value="GRU+Attn" />
      </View>

      {/* ── Temporal attention heatmap ── */}
      <SectionTitle icon={<Brain size={16} color={PURPLE_LIGHT} />} title="Temporal attention" />
      {response.attention_weights?.length > 0
        ? <AttentionHeatmap weights={response.attention_weights} />
        : <Text style={s.heatmapHint}>Not available — restart the server to enable this.</Text>
      }
      {response.attention_weights?.length > 0 && (
        <Text style={s.heatmapHint}>Brighter segments = frames the model focused on most</Text>
      )}

      {/* ── All predictions ── */}
      <SectionTitle icon={<Trophy size={16} color={PURPLE_LIGHT} />} title="All predictions" />
      {response.predictions.map((p) => (
        <PredictionRow key={p.rank} prediction={p} isTop={p.rank === 1} />
      ))}

      {/* ── Try again ── */}
      <TouchableOpacity style={s.retryBtn} onPress={onReset}>
        <RotateCcw size={18} color="#fff" />
        <Text style={s.retryText}>Try again</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

// ── Attention heatmap ──────────────────────────────────────────────────────────

function AttentionHeatmap({ weights }: { weights: number[] }) {
  const max = Math.max(...weights);
  const normalized = weights.map((w) => (max > 0 ? w / max : 0));

  return (
    <View style={hm.container}>
      <View style={hm.row}>
        {normalized.map((v, i) => (
          <View
            key={i}
            style={[
              hm.bar,
              {
                backgroundColor: `rgba(139, 92, 246, ${0.08 + v * 0.92})`,
                borderTopWidth: v > 0.6 ? 2 : 0,
                borderTopColor: '#c4b5fd',
              },
            ]}
          />
        ))}
      </View>
      <View style={hm.labels}>
        <Text style={hm.label}>Frame 1</Text>
        <Text style={hm.label}>Frame {weights.length}</Text>
      </View>
    </View>
  );
}

const hm = StyleSheet.create({
  container: { marginBottom: 6 },
  row: { flexDirection: 'row', height: 48, gap: 1.5, borderRadius: 8, overflow: 'hidden' },
  bar: { flex: 1, borderRadius: 2 },
  labels: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 4 },
  label: { color: '#555', fontSize: 11 },
});

// ── Sub-components ─────────────────────────────────────────────────────────────

function SectionTitle({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <View style={sec.row}>
      {icon}
      <Text style={sec.text}>{title}</Text>
    </View>
  );
}

const sec = StyleSheet.create({
  row: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 10, marginTop: 24 },
  text: { color: '#fff', fontSize: 16, fontWeight: '600' },
});

function ConfidenceBadge({ confidence }: { confidence: number }) {
  const pct = confidence * 100;
  const color = pct >= 70 ? '#4caf50' : pct >= 40 ? '#ff9800' : '#f44336';
  return (
    <View style={[badge.container, { borderColor: color + '99' }]}>
      <Text style={[badge.text, { color }]}>{pct.toFixed(1)}% confidence</Text>
    </View>
  );
}

const badge = StyleSheet.create({
  container: { borderWidth: 1, borderRadius: 20, paddingHorizontal: 14, paddingVertical: 5, marginTop: 8 },
  text: { fontWeight: '600', fontSize: 14 },
});

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <View style={stat.card}>
      {icon}
      <Text style={stat.value}>{value}</Text>
      <Text style={stat.label}>{label}</Text>
    </View>
  );
}

const stat = StyleSheet.create({
  card: { flex: 1, alignItems: 'center', gap: 4, backgroundColor: '#1a1a1a', borderRadius: 12, padding: 14 },
  value: { color: '#fff', fontWeight: 'bold', fontSize: 15 },
  label: { color: '#666', fontSize: 11, textAlign: 'center' },
});

function PredictionRow({ prediction, isTop }: { prediction: Prediction; isTop: boolean }) {
  return (
    <View style={[row.container, isTop && row.containerTop]}>
      <View style={[row.rank, isTop && row.rankTop]}>
        <Text style={row.rankText}>{prediction.rank}</Text>
      </View>
      <Text style={[row.word, isTop && row.wordTop]}>{prediction.word}</Text>
      <View style={row.barGroup}>
        <Text style={[row.pct, isTop && row.pctTop]}>{(prediction.confidence * 100).toFixed(1)}%</Text>
        <View style={row.barBg}>
          <View style={[row.barFill, { width: `${prediction.confidence * 100}%` }, isTop && row.barFillTop]} />
        </View>
      </View>
    </View>
  );
}

const row = StyleSheet.create({
  container: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: '#1a1a1a', borderRadius: 10, padding: 12, marginBottom: 8 },
  containerTop: { backgroundColor: '#2d1b69', borderWidth: 1, borderColor: PURPLE + '66' },
  rank: { width: 28, height: 28, borderRadius: 14, backgroundColor: '#333', alignItems: 'center', justifyContent: 'center' },
  rankTop: { backgroundColor: PURPLE },
  rankText: { color: '#fff', fontSize: 12, fontWeight: 'bold' },
  word: { flex: 1, color: '#ccc', fontSize: 16 },
  wordTop: { color: '#fff', fontWeight: '600' },
  barGroup: { alignItems: 'flex-end', gap: 4 },
  pct: { color: '#888', fontSize: 13, fontWeight: '500' },
  pctTop: { color: '#fff' },
  barBg: { width: 90, height: 5, backgroundColor: '#333', borderRadius: 3, overflow: 'hidden' },
  barFill: { height: '100%', backgroundColor: '#555', borderRadius: 3 },
  barFillTop: { backgroundColor: PURPLE },
});

// ── Page styles ────────────────────────────────────────────────────────────────

const s = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: '#0d0d0d' },
  content: { padding: 20, paddingBottom: 48 },

  heroCard: { borderRadius: 20, padding: 28, alignItems: 'center', marginBottom: 16, backgroundColor: '#1e1b4b' },
  heroLabel: { color: '#a5b4fc', fontSize: 14 },
  heroWord: { color: '#fff', fontSize: 48, fontWeight: 'bold', letterSpacing: 2, marginTop: 8 },

  statsGrid: { flexDirection: 'row', gap: 8, marginBottom: 4, flexWrap: 'wrap' },

  heatmapHint: { color: '#555', fontSize: 11, marginBottom: 4 },

  retryBtn: { marginTop: 28, backgroundColor: PURPLE, borderRadius: 12, padding: 16, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
  retryText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
