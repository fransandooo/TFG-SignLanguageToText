import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import 'screens/record_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Permission.camera.request();

  List<CameraDescription> cameras = [];
  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Could not initialise cameras: $e');
  }

  runApp(SignToTextApp(cameras: cameras));
}

class SignToTextApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const SignToTextApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Sign to Text',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        colorScheme: const ColorScheme.dark(
          primary: Colors.deepPurple,
          secondary: Colors.deepPurpleAccent,
        ),
        scaffoldBackgroundColor: Colors.black,
      ),
      home: RecordScreen(cameras: cameras),
    );
  }
}
