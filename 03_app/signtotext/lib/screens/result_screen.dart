import 'package:flutter/material.dart';

import '../models/prediction.dart';

class ResultScreen extends StatelessWidget {
  final PredictResponse response;

  const ResultScreen({super.key, required this.response});

  @override
  Widget build(BuildContext context) {
    final top = response.topPrediction;
    final rest = response.predictions.skip(1).toList();

    return Scaffold(
      backgroundColor: const Color(0xFF0D0D0D),
      appBar: AppBar(
        backgroundColor: const Color(0xFF0D0D0D),
        title: const Text('Result', style: TextStyle(color: Colors.white)),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // ── Top prediction card ────────────────────────────────────────
            Container(
              padding: const EdgeInsets.all(28),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    Colors.deepPurple.shade800,
                    Colors.indigo.shade700,
                  ],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Column(
                children: [
                  const Text(
                    'Detected Sign',
                    style: TextStyle(color: Colors.white70, fontSize: 14),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    top.word.toUpperCase(),
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 48,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 2,
                    ),
                  ),
                  const SizedBox(height: 8),
                  _ConfidenceChip(confidence: top.confidence),
                ],
              ),
            ),

            const SizedBox(height: 32),

            // ── Stats row ─────────────────────────────────────────────────
            Row(
              children: [
                _StatCard(
                  label: 'Frames',
                  value: '${response.framesProcessed}',
                  icon: Icons.video_library_outlined,
                ),
                const SizedBox(width: 12),
                _StatCard(
                  label: 'Hand detected',
                  value: '${(response.detectionRate * 100).round()}%',
                  icon: Icons.back_hand_outlined,
                ),
              ],
            ),

            const SizedBox(height: 32),

            // ── Top-5 list ────────────────────────────────────────────────
            const Text(
              'All predictions',
              style: TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 12),
            _PredictionRow(prediction: top, isTop: true),
            ...rest.map((p) => _PredictionRow(prediction: p)),

            const SizedBox(height: 32),

            // ── Try again button ──────────────────────────────────────────
            ElevatedButton.icon(
              onPressed: () => Navigator.pop(context),
              icon: const Icon(Icons.replay),
              label: const Text('Try again'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.deepPurple,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ConfidenceChip extends StatelessWidget {
  final double confidence;

  const _ConfidenceChip({required this.confidence});

  Color get _color {
    if (confidence >= 0.7) return Colors.greenAccent;
    if (confidence >= 0.4) return Colors.orangeAccent;
    return Colors.redAccent;
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
      decoration: BoxDecoration(
        color: _color.withAlpha(51),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: _color.withAlpha(153)),
      ),
      child: Text(
        '${(confidence * 100).toStringAsFixed(1)}% confidence',
        style: TextStyle(
          color: _color,
          fontWeight: FontWeight.w600,
          fontSize: 14,
        ),
      ),
    );
  }
}

class _StatCard extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;

  const _StatCard({
    required this.label,
    required this.value,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: const Color(0xFF1A1A1A),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          children: [
            Icon(icon, color: Colors.deepPurpleAccent, size: 20),
            const SizedBox(width: 10),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(value,
                    style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 16)),
                Text(label,
                    style:
                        TextStyle(color: Colors.grey[500], fontSize: 12)),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _PredictionRow extends StatelessWidget {
  final Prediction prediction;
  final bool isTop;

  const _PredictionRow({required this.prediction, this.isTop = false});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: isTop
              ? Colors.deepPurple.withAlpha(38)
              : const Color(0xFF1A1A1A),
          borderRadius: BorderRadius.circular(10),
          border: isTop
              ? Border.all(color: Colors.deepPurpleAccent.withAlpha(102))
              : null,
        ),
        child: Row(
          children: [
            // Rank badge
            Container(
              width: 28,
              height: 28,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: isTop
                    ? Colors.deepPurpleAccent
                    : Colors.grey.shade800,
              ),
              child: Center(
                child: Text(
                  '${prediction.rank}',
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.bold),
                ),
              ),
            ),
            const SizedBox(width: 12),

            // Word
            Expanded(
              child: Text(
                prediction.word,
                style: TextStyle(
                  color: isTop ? Colors.white : Colors.grey[300],
                  fontSize: 16,
                  fontWeight:
                      isTop ? FontWeight.w600 : FontWeight.normal,
                ),
              ),
            ),

            // Confidence bar + label
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  '${(prediction.confidence * 100).toStringAsFixed(1)}%',
                  style: TextStyle(
                    color: isTop ? Colors.white : Colors.grey[400],
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 4),
                SizedBox(
                  width: 90,
                  height: 5,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(3),
                    child: LinearProgressIndicator(
                      value: prediction.confidence,
                      backgroundColor: Colors.grey.shade800,
                      color: isTop
                          ? Colors.deepPurpleAccent
                          : Colors.grey.shade500,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
