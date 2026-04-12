/// App-wide configuration. Change SERVER_HOST to your laptop's local IP
/// when testing on a physical device. For emulator use 10.0.2.2.
class AppConfig {
  // Change this to your local network IP when using a physical device
  // e.g. "192.168.1.42"
  static const String serverHost = "10.0.2.2";
  static const int serverPort = 8000;

  static String get baseUrl => "http://$serverHost:$serverPort";
  static String get predictUrl => "$baseUrl/predict";
  static String get healthUrl => "$baseUrl/health";
  static String get classesUrl => "$baseUrl/classes";

  static const int recordingDurationSeconds = 3;
}
