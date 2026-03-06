import 'package:app1/app.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('renders premium shell and primary sections', (tester) async {
    await tester.pumpWidget(const SignLanguageApp());

    expect(find.text('SIGNUM COUNSEL'), findsOneWidget);
    expect(find.text('Demo UI'), findsOneWidget);
    expect(find.text('Inicio'), findsOneWidget);
    expect(find.text('Directo'), findsOneWidget);
    expect(find.text('Historial'), findsOneWidget);
    expect(find.text('Ajustes'), findsOneWidget);
  });
}
