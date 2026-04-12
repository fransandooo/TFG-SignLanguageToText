import 'dart:io';

import 'package:http/http.dart' as http;
import 'dart:convert';

import '../config.dart';
import '../models/prediction.dart';

class ApiService {
  static Future<bool> checkHealth() async {
    try {
      final res = await http
          .get(Uri.parse(AppConfig.healthUrl))
          .timeout(const Duration(seconds: 5));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<PredictResponse> predict(File videoFile) async {
    final uri = Uri.parse(AppConfig.predictUrl);
    final request = http.MultipartRequest('POST', uri);

    request.files.add(
      await http.MultipartFile.fromPath('video', videoFile.path),
    );

    final streamedResponse = await request.send().timeout(
          const Duration(seconds: 60),
        );
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      final json = jsonDecode(response.body) as Map<String, dynamic>;
      return PredictResponse.fromJson(json);
    } else {
      // Try to extract detail message from FastAPI error
      String detail = 'Server error ${response.statusCode}';
      try {
        final json = jsonDecode(response.body) as Map<String, dynamic>;
        detail = json['detail'] as String? ?? detail;
      } catch (_) {}
      throw Exception(detail);
    }
  }
}
