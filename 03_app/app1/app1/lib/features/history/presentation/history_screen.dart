import 'package:flutter/material.dart';

import '../../../core/theme/app_colors.dart';
import '../../../core/widgets/luxury_card.dart';
import '../../../core/widgets/section_heading.dart';
import '../../../shared/data/mock_data.dart';

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return ListView.separated(
      padding: const EdgeInsets.fromLTRB(24, 28, 24, 120),
      itemCount: MockData.sessions.length + 1,
      separatorBuilder: (_, _) => const SizedBox(height: 14),
      itemBuilder: (context, index) {
        if (index == 0) {
          return const SectionHeading(
            eyebrow: 'Record archive',
            title: 'Historial preparado para revisión y trazabilidad.',
            subtitle:
                'Cada sesión se presenta como un registro claro, útil para guardar borradores, añadir metadatos o conectar futuros informes.',
          );
        }

        final session = MockData.sessions[index - 1];
        return LuxuryCard(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Text(session.title, style: textTheme.titleLarge),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 8,
                    ),
                    decoration: BoxDecoration(
                      color: AppColors.midnight,
                      borderRadius: BorderRadius.circular(999),
                    ),
                    child: Text(
                      '${(session.confidence * 100).round()}%',
                      style: textTheme.labelLarge?.copyWith(
                        color: AppColors.ivory,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              Text(
                '${session.timestamp}  •  ${session.duration}',
                style: textTheme.bodyMedium,
              ),
              const SizedBox(height: 14),
              Text(
                session.excerpt,
                style: textTheme.bodyLarge?.copyWith(color: AppColors.ink),
              ),
              const SizedBox(height: 18),
              Row(
                children: [
                  TextButton(
                    onPressed: () {},
                    child: const Text('Abrir sesión'),
                  ),
                  const SizedBox(width: 8),
                  TextButton(
                    onPressed: () {},
                    child: const Text('Exportar texto'),
                  ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }
}
