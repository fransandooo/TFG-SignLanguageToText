import {
  Brain,
  Clock,
  Film,
  Hand,
  RotateCcw,
  Trophy,
  Volume2,
  VolumeX,
  Waves,
} from 'lucide-react-native';
import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

import {
  Prediction,
  PredictionResult,
  SequenceSegment,
} from '../types';

interface Props {
  onReplay: () => void;
  onReset: () => void;
  onToggleTts: () => void;
  response: PredictionResult;
  ttsEnabled: boolean;
}

const PURPLE = '#7c3aed';
const PURPLE_LIGHT = '#c4b5fd';

export default function ResultScreen({
  onReplay,
  onReset,
  onToggleTts,
  response,
  ttsEnabled,
}: Props) {
  const sequenceResponse = response.mode === 'sequence' ? response : null;
  const isolatedResponse = response.mode === 'isolated' ? response : null;
  const top = isolatedResponse?.predictions[0] ?? null;
  const heroText = sequenceResponse ? sequenceResponse.sequence_text || 'NO SEGMENTS' : top?.word.toUpperCase() ?? 'N/A';
  const heroLabel = sequenceResponse ? 'Detected Sequence' : 'Detected Sign';
  const confidence = sequenceResponse ? averageConfidence(sequenceResponse.segments) : top?.confidence ?? 0;

  return (
    <ScrollView style={s.scroll} contentContainerStyle={s.content} showsVerticalScrollIndicator={false}>
      <View style={s.heroCard}>
        <Text style={s.heroLabel}>{heroLabel}</Text>
        <Text style={s.heroWord}>{heroText.toUpperCase()}</Text>
        <ConfidenceBadge confidence={confidence} />
        <Text style={s.heroMeta}>
          {response.model_label} · {response.num_classes} classes
        </Text>
      </View>

      <View style={s.actionRow}>
        <TouchableOpacity style={s.secondaryBtn} onPress={onReplay}>
          <Volume2 size={18} color="#fff" />
          <Text style={s.secondaryBtnText}>Replay voice</Text>
        </TouchableOpacity>

        <TouchableOpacity style={s.secondaryBtn} onPress={onToggleTts}>
          {ttsEnabled ? <Volume2 size={18} color="#fff" /> : <VolumeX size={18} color="#aaa" />}
          <Text style={s.secondaryBtnText}>{ttsEnabled ? 'Auto voice on' : 'Auto voice off'}</Text>
        </TouchableOpacity>
      </View>

      <View style={s.statsGrid}>
        <StatCard
          icon={<Film size={18} color={PURPLE_LIGHT} />}
          label="Frames"
          value={String(response.frames_processed)}
        />
        <StatCard
          icon={<Hand size={18} color={PURPLE_LIGHT} />}
          label={sequenceResponse ? 'Avg detect' : 'Hand detect'}
          value={sequenceResponse
            ? `${Math.round(sequenceResponse.avg_detection_rate * 100)}%`
            : `${Math.round((isolatedResponse?.detection_rate ?? 0) * 100)}%`}
        />
        <StatCard
          icon={<Clock size={18} color={PURPLE_LIGHT} />}
          label="Processing"
          value={`${response.processing_time_ms.toFixed(0)} ms`}
        />
        <StatCard
          icon={sequenceResponse ? <Waves size={18} color={PURPLE_LIGHT} /> : <Brain size={18} color={PURPLE_LIGHT} />}
          label={sequenceResponse ? 'Windows' : 'Model'}
          value={sequenceResponse ? String(sequenceResponse.window_count) : response.architecture}
        />
      </View>

      {sequenceResponse ? (
        <>
          <SectionTitle icon={<Waves size={16} color={PURPLE_LIGHT} />} title="Detected segments" />
          {sequenceResponse.segments.map((segment, index) => (
            <SegmentRow key={`${segment.word}-${segment.start_ms}-${index}`} segment={segment} />
          ))}
        </>
      ) : (
        <>
          <SectionTitle icon={<Brain size={16} color={PURPLE_LIGHT} />} title="Temporal attention" />
          {isolatedResponse?.attention_weights?.length ? (
            <>
              <AttentionHeatmap weights={isolatedResponse.attention_weights} />
              <Text style={s.heatmapHint}>Brighter segments = frames the model focused on most</Text>
            </>
          ) : (
            <Text style={s.heatmapHint}>Not available for this architecture.</Text>
          )}

          <SectionTitle icon={<Trophy size={16} color={PURPLE_LIGHT} />} title="All predictions" />
          {isolatedResponse?.predictions.map((prediction) => (
            <PredictionRow
              key={prediction.rank}
              isTop={prediction.rank === 1}
              prediction={prediction}
            />
          ))}
        </>
      )}

      <TouchableOpacity style={s.retryBtn} onPress={onReset}>
        <RotateCcw size={18} color="#fff" />
        <Text style={s.retryText}>Try again</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

function averageConfidence(segments: SequenceSegment[]): number {
  if (!segments.length) return 0;
  return segments.reduce((sum, segment) => sum + segment.confidence, 0) / segments.length;
}

function AttentionHeatmap({ weights }: { weights: number[] }) {
  const max = Math.max(...weights);
  const normalized = weights.map((weight) => (max > 0 ? weight / max : 0));

  return (
    <View style={hm.container}>
      <View style={hm.row}>
        {normalized.map((value, index) => (
          <View
            key={index}
            style={[
              hm.bar,
              {
                backgroundColor: `rgba(139, 92, 246, ${0.08 + value * 0.92})`,
                borderTopWidth: value > 0.6 ? 2 : 0,
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
    <View style={[badge.container, { borderColor: `${color}99` }]}>
      <Text style={[badge.text, { color }]}>{pct.toFixed(1)}% confidence</Text>
    </View>
  );
}

const badge = StyleSheet.create({
  container: {
    borderWidth: 1,
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 5,
    marginTop: 8,
  },
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
  card: {
    flex: 1,
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 14,
    minWidth: 150,
  },
  value: { color: '#fff', fontWeight: 'bold', fontSize: 15, textAlign: 'center' },
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
          <View
            style={[
              row.barFill,
              { width: `${prediction.confidence * 100}%` },
              isTop && row.barFillTop,
            ]}
          />
        </View>
      </View>
    </View>
  );
}

const row = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: '#1a1a1a',
    borderRadius: 10,
    padding: 12,
    marginBottom: 8,
  },
  containerTop: { backgroundColor: '#2d1b69', borderWidth: 1, borderColor: `${PURPLE}66` },
  rank: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: '#333',
    alignItems: 'center',
    justifyContent: 'center',
  },
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

function SegmentRow({ segment }: { segment: SequenceSegment }) {
  return (
    <View style={seg.row}>
      <View>
        <Text style={seg.word}>{segment.word}</Text>
        <Text style={seg.meta}>
          {formatMs(segment.start_ms)} - {formatMs(segment.end_ms)} · {segment.windows} windows
        </Text>
      </View>
      <Text style={seg.confidence}>{(segment.confidence * 100).toFixed(1)}%</Text>
    </View>
  );
}

function formatMs(value: number): string {
  const totalSeconds = value / 1000;
  return `${totalSeconds.toFixed(1)}s`;
}

const seg = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#1a1a1a',
    borderRadius: 10,
    padding: 14,
    marginBottom: 8,
  },
  word: { color: '#fff', fontSize: 16, fontWeight: '700', marginBottom: 4 },
  meta: { color: '#888', fontSize: 12 },
  confidence: { color: '#c4b5fd', fontSize: 14, fontWeight: '700' },
});

const s = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: '#0d0d0d' },
  content: { padding: 20, paddingBottom: 48 },
  heroCard: {
    borderRadius: 20,
    padding: 28,
    alignItems: 'center',
    marginBottom: 16,
    backgroundColor: '#1e1b4b',
  },
  heroLabel: { color: '#a5b4fc', fontSize: 14 },
  heroWord: {
    color: '#fff',
    fontSize: 36,
    fontWeight: 'bold',
    letterSpacing: 1.2,
    marginTop: 8,
    textAlign: 'center',
  },
  heroMeta: { color: '#9aa3c7', fontSize: 12, marginTop: 10, textAlign: 'center' },
  actionRow: { flexDirection: 'row', gap: 10, marginBottom: 16 },
  secondaryBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#171717',
    borderRadius: 12,
    padding: 14,
  },
  secondaryBtnText: { color: '#fff', fontSize: 13, fontWeight: '600' },
  statsGrid: { flexDirection: 'row', gap: 8, marginBottom: 4, flexWrap: 'wrap' },
  heatmapHint: { color: '#555', fontSize: 11, marginBottom: 4 },
  retryBtn: {
    marginTop: 28,
    backgroundColor: PURPLE,
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  retryText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
