import 'package:flutter/material.dart';

import '../../../core/theme/app_colors.dart';
import '../../../core/widgets/luxury_card.dart';
import '../../../core/widgets/section_heading.dart';
import '../../../shared/data/mock_data.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return CustomScrollView(
      slivers: [
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(24, 28, 24, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SectionHeading(
                  eyebrow: 'Signum Counsel',
                  title:
                      'Traducción de lengua de signos con presencia institucional.',
                  subtitle:
                      'Una base móvil pensada para entornos exigentes: clara, elegante y preparada para crecer hacia captura en directo, revisión y trazabilidad.',
                  action: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 14,
                      vertical: 10,
                    ),
                    decoration: BoxDecoration(
                      color: AppColors.midnight,
                      borderRadius: BorderRadius.circular(999),
                    ),
                    child: Text(
                      'Demo UI',
                      style: textTheme.labelLarge?.copyWith(
                        color: AppColors.ivory,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 24),
                LuxuryCard(
                  backgroundColor: AppColors.midnight,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Panel ejecutivo',
                        style: textTheme.labelLarge?.copyWith(
                          color: AppColors.gold,
                          letterSpacing: 1.4,
                        ),
                      ),
                      const SizedBox(height: 14),
                      Text(
                        'La experiencia móvil debe transmitir confianza antes incluso de conectar la cámara o el modelo.',
                        style: textTheme.titleLarge?.copyWith(
                          color: AppColors.ivory,
                          height: 1.4,
                        ),
                      ),
                      const SizedBox(height: 20),
                      Wrap(
                        spacing: 12,
                        runSpacing: 12,
                        children: MockData.metrics
                            .map((metric) => _MetricChip(metric: metric))
                            .toList(),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 28),
                Text('Siguientes módulos', style: textTheme.titleLarge),
                const SizedBox(height: 16),
                const _FeatureGrid(),
                const SizedBox(height: 28),
                Text('Frases destacadas', style: textTheme.titleLarge),
                const SizedBox(height: 16),
              ],
            ),
          ),
        ),
        SliverPadding(
          padding: const EdgeInsets.fromLTRB(24, 0, 24, 120),
          sliver: SliverList.separated(
            itemBuilder: (context, index) {
              final phrase = MockData.savedPhrases[index];
              return LuxuryCard(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      phrase.category.toUpperCase(),
                      style: textTheme.labelLarge?.copyWith(letterSpacing: 1.3),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      phrase.text,
                      style: textTheme.bodyLarge?.copyWith(
                        color: AppColors.ink,
                      ),
                    ),
                  ],
                ),
              );
            },
            separatorBuilder: (_, _) => const SizedBox(height: 14),
            itemCount: MockData.savedPhrases.length,
          ),
        ),
      ],
    );
  }
}

class _MetricChip extends StatelessWidget {
  const _MetricChip({required this.metric});

  final DashboardMetric metric;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return Container(
      width: 170,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.06),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: Colors.white.withValues(alpha: 0.12)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            metric.label,
            style: textTheme.bodyMedium?.copyWith(color: AppColors.parchment),
          ),
          const SizedBox(height: 10),
          Text(
            metric.value,
            style: textTheme.headlineMedium?.copyWith(color: AppColors.ivory),
          ),
          const SizedBox(height: 6),
          Text(
            metric.caption,
            style: textTheme.bodyMedium?.copyWith(
              color: AppColors.parchment.withValues(alpha: 0.8),
            ),
          ),
        ],
      ),
    );
  }
}

class _FeatureGrid extends StatelessWidget {
  const _FeatureGrid();

  @override
  Widget build(BuildContext context) {
    final features = [
      (
        'Captura en directo',
        'Preview de cámara, estados de sesión y traducción progresiva.',
        Icons.videocam_rounded,
      ),
      (
        'Revisión posterior',
        'Edición del texto antes de copiar, compartir o archivar.',
        Icons.fact_check_rounded,
      ),
      (
        'Historial trazable',
        'Sesiones con fecha, duración y nivel de confianza.',
        Icons.library_books_rounded,
      ),
      (
        'Modo institucional',
        'Una UI contenida para despachos, atención al cliente o demostraciones.',
        Icons.apartment_rounded,
      ),
    ];

    return GridView.builder(
      itemCount: features.length,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        mainAxisSpacing: 14,
        crossAxisSpacing: 14,
        childAspectRatio: 0.98,
      ),
      itemBuilder: (context, index) {
        final feature = features[index];
        return LuxuryCard(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: AppColors.midnight,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Icon(feature.$3, color: AppColors.ivory),
              ),
              const Spacer(),
              Text(feature.$1, style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              Text(feature.$2, style: Theme.of(context).textTheme.bodyMedium),
            ],
          ),
        );
      },
    );
  }
}
