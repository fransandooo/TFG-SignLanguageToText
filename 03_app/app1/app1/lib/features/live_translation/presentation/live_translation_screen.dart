import 'package:flutter/material.dart';

import '../../../core/theme/app_colors.dart';
import '../../../core/widgets/luxury_card.dart';
import '../../../core/widgets/section_heading.dart';
import '../../../shared/data/mock_data.dart';

class LiveTranslationScreen extends StatelessWidget {
  const LiveTranslationScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return ListView(
      padding: const EdgeInsets.fromLTRB(24, 28, 24, 120),
      children: [
        const SectionHeading(
          eyebrow: 'Live Session',
          title:
              'Traducción en directo lista para una futura integración de cámara.',
          subtitle:
              'Esta pantalla ya contempla los bloques clave de producto: vista previa, confianza, texto incremental y acciones de revisión.',
        ),
        const SizedBox(height: 24),
        const _CameraStage(),
        const SizedBox(height: 18),
        Row(
          children: [
            Expanded(
              child: _StatusPill(
                icon: Icons.shield_outlined,
                label: 'Confiabilidad alta',
                value: '94%',
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _StatusPill(
                icon: Icons.timelapse_rounded,
                label: 'Latencia mock',
                value: '280 ms',
              ),
            ),
          ],
        ),
        const SizedBox(height: 18),
        LuxuryCard(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Text('Transcript', style: textTheme.titleLarge),
                  const Spacer(),
                  FilledButton.tonal(
                    onPressed: () {},
                    child: const Text('Guardar borrador'),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              ...MockData.liveTranscript.map(
                (line) => Padding(
                  padding: const EdgeInsets.only(bottom: 14),
                  child: Text(
                    line,
                    style: textTheme.bodyLarge?.copyWith(color: AppColors.ink),
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 18),
        LuxuryCard(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Controles de sesión', style: textTheme.titleLarge),
              const SizedBox(height: 16),
              Wrap(
                spacing: 12,
                runSpacing: 12,
                children: [
                  FilledButton.icon(
                    onPressed: () {},
                    icon: const Icon(Icons.play_arrow_rounded),
                    label: const Text('Iniciar captura'),
                  ),
                  OutlinedButton.icon(
                    onPressed: () {},
                    icon: const Icon(Icons.pause_circle_outline_rounded),
                    label: const Text('Pausar'),
                  ),
                  OutlinedButton.icon(
                    onPressed: () {},
                    icon: const Icon(Icons.content_copy_rounded),
                    label: const Text('Copiar texto'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _CameraStage extends StatelessWidget {
  const _CameraStage();

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return Container(
      height: 300,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(32),
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [AppColors.midnight, AppColors.ink],
        ),
        boxShadow: const [
          BoxShadow(
            color: Color(0x160F1720),
            blurRadius: 30,
            offset: Offset(0, 14),
          ),
        ],
      ),
      child: Stack(
        children: [
          Positioned.fill(
            child: DecoratedBox(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(32),
                border: Border.all(
                  color: AppColors.goldMuted.withValues(alpha: 0.35),
                ),
              ),
            ),
          ),
          Positioned(
            top: 24,
            left: 24,
            right: 24,
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 8,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.08),
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Text(
                    'Camera Placeholder',
                    style: textTheme.labelLarge?.copyWith(
                      color: AppColors.ivory,
                    ),
                  ),
                ),
                const Spacer(),
                const Icon(
                  Icons.fiber_manual_record_rounded,
                  color: AppColors.gold,
                ),
              ],
            ),
          ),
          Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(
                  Icons.sign_language_rounded,
                  size: 64,
                  color: AppColors.gold,
                ),
                const SizedBox(height: 18),
                Text(
                  'Espacio preparado para integrar cámara y superposiciones.',
                  textAlign: TextAlign.center,
                  style: textTheme.titleMedium?.copyWith(
                    color: AppColors.ivory,
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  'Bounding boxes, landmarks, estados y feedback de sesión cabrán aquí.',
                  textAlign: TextAlign.center,
                  style: textTheme.bodyMedium?.copyWith(
                    color: AppColors.parchment.withValues(alpha: 0.8),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _StatusPill extends StatelessWidget {
  const _StatusPill({
    required this.icon,
    required this.label,
    required this.value,
  });

  final IconData icon;
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return LuxuryCard(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
      child: Row(
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: AppColors.midnight,
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(icon, color: AppColors.ivory),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label, style: textTheme.bodyMedium),
                const SizedBox(height: 4),
                Text(value, style: textTheme.titleMedium),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
