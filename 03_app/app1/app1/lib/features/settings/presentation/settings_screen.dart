import 'package:flutter/material.dart';

import '../../../core/theme/app_colors.dart';
import '../../../core/widgets/luxury_card.dart';
import '../../../core/widgets/section_heading.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool _demoMode = true;
  bool _highContrast = false;
  bool _saveDrafts = true;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return ListView(
      padding: const EdgeInsets.fromLTRB(24, 28, 24, 120),
      children: [
        const SectionHeading(
          eyebrow: 'Settings',
          title: 'Controles de producto y señales de confianza.',
          subtitle:
              'Aquí se concentra lo que más adelante conectará con permisos, accesibilidad, entorno técnico y preferencias del despacho o usuario final.',
        ),
        const SizedBox(height: 24),
        LuxuryCard(
          child: Column(
            children: [
              _SettingSwitchTile(
                title: 'Modo demostración',
                subtitle:
                    'Mantiene datos simulados para diseñar la experiencia sin backend.',
                value: _demoMode,
                onChanged: (value) => setState(() => _demoMode = value),
              ),
              const Divider(height: 32),
              _SettingSwitchTile(
                title: 'Mayor contraste',
                subtitle:
                    'Preparado para una futura variante visual orientada a accesibilidad.',
                value: _highContrast,
                onChanged: (value) => setState(() => _highContrast = value),
              ),
              const Divider(height: 32),
              _SettingSwitchTile(
                title: 'Guardar borradores automáticamente',
                subtitle:
                    'Pensado para sesiones de interpretación o atención al cliente.',
                value: _saveDrafts,
                onChanged: (value) => setState(() => _saveDrafts = value),
              ),
            ],
          ),
        ),
        const SizedBox(height: 18),
        LuxuryCard(
          backgroundColor: AppColors.midnight,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Próximas integraciones',
                style: textTheme.titleLarge?.copyWith(color: AppColors.ivory),
              ),
              const SizedBox(height: 16),
              ...const [
                'Permisos de cámara y micrófono.',
                'Selección de entorno: local, laboratorio o producción.',
                'Control de idioma de interfaz y formato de exportación.',
                'Ficha técnica del modelo y versión del sistema.',
              ].map(
                (item) => Padding(
                  padding: EdgeInsets.only(bottom: 10),
                  child: _BulletRow(text: item),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _SettingSwitchTile extends StatelessWidget {
  const _SettingSwitchTile({
    required this.title,
    required this.subtitle,
    required this.value,
    required this.onChanged,
  });

  final String title;
  final String subtitle;
  final bool value;
  final ValueChanged<bool> onChanged;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: textTheme.titleMedium),
              const SizedBox(height: 8),
              Text(subtitle, style: textTheme.bodyMedium),
            ],
          ),
        ),
        const SizedBox(width: 16),
        Switch(value: value, onChanged: onChanged),
      ],
    );
  }
}

class _BulletRow extends StatelessWidget {
  const _BulletRow({required this.text});

  final String text;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Padding(
          padding: EdgeInsets.only(top: 8),
          child: Icon(Icons.square, size: 8, color: AppColors.gold),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Text(
            text,
            style: Theme.of(
              context,
            ).textTheme.bodyLarge?.copyWith(color: AppColors.parchment),
          ),
        ),
      ],
    );
  }
}
