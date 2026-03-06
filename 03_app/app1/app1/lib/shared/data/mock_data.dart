class DashboardMetric {
  const DashboardMetric({
    required this.label,
    required this.value,
    required this.caption,
  });

  final String label;
  final String value;
  final String caption;
}

class SessionRecord {
  const SessionRecord({
    required this.title,
    required this.timestamp,
    required this.duration,
    required this.excerpt,
    required this.confidence,
  });

  final String title;
  final String timestamp;
  final String duration;
  final String excerpt;
  final double confidence;
}

class SavedPhrase {
  const SavedPhrase({required this.category, required this.text});

  final String category;
  final String text;
}

abstract final class MockData {
  static const metrics = [
    DashboardMetric(
      label: 'Sesiones hoy',
      value: '12',
      caption: '3 pendientes de revisión manual',
    ),
    DashboardMetric(
      label: 'Precisión demo',
      value: '94.2%',
      caption: 'Basado en el último entorno de prueba visual',
    ),
    DashboardMetric(
      label: 'Tiempo medio',
      value: '18s',
      caption: 'Desde inicio hasta texto revisable',
    ),
  ];

  static const liveTranscript = [
    'Good afternoon, I would like legal assistance regarding a contract review.',
    'Please note that I need the summary in plain language for my family.',
    'Can we save this transcript and send it after verification?',
  ];

  static const sessions = [
    SessionRecord(
      title: 'Consulta inicial',
      timestamp: 'Hoy, 09:40',
      duration: '02:14',
      excerpt: 'I need support understanding the next steps after the hearing.',
      confidence: 0.96,
    ),
    SessionRecord(
      title: 'Revisión de documentación',
      timestamp: 'Ayer, 18:05',
      duration: '01:32',
      excerpt: 'Please retain the current version and mark the urgent clauses.',
      confidence: 0.91,
    ),
    SessionRecord(
      title: 'Seguimiento de expediente',
      timestamp: '3 marzo, 12:20',
      duration: '03:08',
      excerpt:
          'I would like to confirm the translated record before submission.',
      confidence: 0.94,
    ),
  ];

  static const savedPhrases = [
    SavedPhrase(
      category: 'Atención al cliente',
      text:
          'Please take a seat. Your interpreter-assisted consultation will begin shortly.',
    ),
    SavedPhrase(
      category: 'Proceso',
      text: 'Would you like this transcript attached to your case summary?',
    ),
    SavedPhrase(
      category: 'Accesibilidad',
      text: 'We can repeat the capture and refine the text before saving it.',
    ),
  ];
}
