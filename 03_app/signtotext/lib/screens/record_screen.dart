import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../config.dart';
import '../services/api_service.dart';
import 'result_screen.dart';

class RecordScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const RecordScreen({super.key, required this.cameras});

  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  bool _isRecording = false;
  bool _isProcessing = false;
  int _countdown = 0;
  Timer? _countdownTimer;
  String? _statusMessage;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  Future<void> _initCamera() async {
    if (widget.cameras.isEmpty) {
      setState(() => _statusMessage = 'No camera available');
      return;
    }
    // Prefer back camera
    final cam = widget.cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => widget.cameras.first,
    );
    _controller = CameraController(
      cam,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    try {
      await _controller!.initialize();
      if (mounted) setState(() {});
    } catch (e) {
      setState(() => _statusMessage = 'Camera error: $e');
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _countdownTimer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _controller!.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _startRecording() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (_isRecording || _isProcessing) return;

    setState(() {
      _isRecording = true;
      _countdown = AppConfig.recordingDurationSeconds;
      _statusMessage = null;
    });

    await _controller!.startVideoRecording();

    _countdownTimer = Timer.periodic(const Duration(seconds: 1), (t) async {
      if (!mounted) {
        t.cancel();
        return;
      }
      setState(() => _countdown--);
      if (_countdown <= 0) {
        t.cancel();
        await _stopAndPredict();
      }
    });
  }

  Future<void> _stopAndPredict() async {
    if (!_isRecording) return;
    setState(() {
      _isRecording = false;
      _isProcessing = true;
      _statusMessage = 'Processing…';
    });

    try {
      final videoFile = await _controller!.stopVideoRecording();
      final file = File(videoFile.path);
      final response = await ApiService.predict(file);

      if (mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => ResultScreen(response: response),
          ),
        );
      }
    } on Exception catch (e) {
      if (mounted) {
        setState(() => _statusMessage = e.toString().replaceFirst('Exception: ', ''));
      }
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final ctrl = _controller;
    final cameraReady = ctrl != null && ctrl.value.isInitialized;

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: const Text(
          'Sign to Text',
          style: TextStyle(color: Colors.white),
        ),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: Column(
        children: [
          // Camera preview
          Expanded(
            child: cameraReady
                ? ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: CameraPreview(ctrl),
                  )
                : Center(
                    child: Text(
                      _statusMessage ?? 'Initializing camera…',
                      style: const TextStyle(color: Colors.white),
                    ),
                  ),
          ),

          // Bottom controls
          Container(
            color: Colors.black,
            padding: const EdgeInsets.symmetric(vertical: 32, horizontal: 24),
            child: Column(
              children: [
                // Status / countdown
                _buildStatusRow(),
                const SizedBox(height: 20),
                // Record button
                _buildRecordButton(cameraReady),
                const SizedBox(height: 12),
                Text(
                  'Hold up your hand and sign a word',
                  style: TextStyle(color: Colors.grey[400], fontSize: 13),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusRow() {
    if (_isProcessing) {
      return const Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
            width: 16,
            height: 16,
            child: CircularProgressIndicator(
                strokeWidth: 2, color: Colors.white),
          ),
          SizedBox(width: 8),
          Text('Analyzing…', style: TextStyle(color: Colors.white)),
        ],
      );
    }
    if (_isRecording) {
      return Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.circle, color: Colors.red, size: 12),
          const SizedBox(width: 6),
          Text(
            'Recording  $_countdown s',
            style: const TextStyle(color: Colors.white, fontSize: 16),
          ),
        ],
      );
    }
    if (_statusMessage != null) {
      return Text(
        _statusMessage!,
        style: const TextStyle(color: Colors.orangeAccent),
        textAlign: TextAlign.center,
      );
    }
    return Text(
      'Tap to record ${AppConfig.recordingDurationSeconds}s clip',
      style: TextStyle(color: Colors.grey[400]),
    );
  }

  Widget _buildRecordButton(bool enabled) {
    final canRecord = enabled && !_isRecording && !_isProcessing;
    return GestureDetector(
      onTap: canRecord ? _startRecording : null,
      child: Container(
        width: 72,
        height: 72,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white, width: 3),
          color: _isRecording ? Colors.red : Colors.transparent,
        ),
        child: _isRecording
            ? const Icon(Icons.stop, color: Colors.white, size: 32)
            : const Icon(Icons.fiber_manual_record,
                color: Colors.red, size: 40),
      ),
    );
  }
}
