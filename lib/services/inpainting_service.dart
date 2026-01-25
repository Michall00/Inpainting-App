import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';
import '../utils/tensor_utils.dart';
import '../utils/app_logger.dart';
import 'execution_provider.dart';

class InpaintingResult {
  final Uint8List bytes;
  final int inferenceDurationMs;

  InpaintingResult({
    required this.bytes,
    required this.inferenceDurationMs,
  });
}

class InpaintingService {
  static bool _envInitialized = false;
  static String lastExecutionProvider = 'unknown';
  static ExecutionProvider preferredExecutionProvider =
      ExecutionProvider.auto;

  static void _ensureEnvironmentInitialized() {
    if (_envInitialized) return;
    OrtEnv.instance.init();
    _envInitialized = true;
  }

  static OrtSessionOptions _createSessionOptions({bool preferCoreML = true}) {
    final options = OrtSessionOptions();

    if (preferCoreML) {
      try {
        options.appendCoreMLProvider(CoreMLFlags.enableOnSubgraph);
      } catch (e) {
        AppLogger.log('CoreML provider skipped: $e');
      }
    }
    return options;
  }

  static OrtSession _createSession(ByteData modelData) {
    if (preferredExecutionProvider == ExecutionProvider.cpu) {
      return _createCpuSession(modelData);
    }

    final buffer = modelData.buffer.asUint8List();

    OrtSessionOptions? options;
    try {
      options = _createSessionOptions(preferCoreML: true);
      final session = OrtSession.fromBuffer(
        buffer,
        options,
      );
      lastExecutionProvider = 'coreml';
      AppLogger.log(
        'Created CoreML inpainting session addr=${session.address}',
      );
      return session;
    } catch (error) {
      AppLogger.log(
        'CoreML session failed, falling back to CPU. Error: $error',
      );
    } finally {
      options?.release();
    }
    // Fallback to CPU
    return _createCpuSession(modelData);
  }

  static OrtSession _createCpuSession(ByteData modelData) {
    final buffer = modelData.buffer.asUint8List();
    final cpuOptions = _createSessionOptions(preferCoreML: false);
    try {
      final session = OrtSession.fromBuffer(buffer, cpuOptions);
      lastExecutionProvider = 'cpu';
      AppLogger.log('Created CPU inpainting session addr=${session.address}');
      return session;
    } finally {
      cpuOptions.release();
    }
  }

  static Future<InpaintingResult> runInpainting({
    required img.Image original,
    required img.Image mask,
    required ByteData modelData,
  }) async {
    _ensureEnvironmentInitialized();

    var session = _createSession(modelData);

    final imageTensor = convertImageToUint8NCHW(original);
    final maskTensor = convertMaskToUint8NCHW(mask);

    final runOptions = OrtRunOptions();
    final inferenceStart = DateTime.now();

    List<OrtValue?>? result;

    try {
      try {
        result = session.run(
          runOptions,
          {'image': imageTensor, 'mask': maskTensor},
          ['result'],
        );
      } catch (error) {
        if (lastExecutionProvider == 'coreml') {
          AppLogger.log(
              'CoreML inference failed (${error.runtimeType}); retrying on CPU.');

          session.release();

          session = _createCpuSession(modelData);
          result = session.run(
            runOptions,
            {'image': imageTensor, 'mask': maskTensor},
            ['result'],
          );
        } else {
          rethrow;
        }
      }

      final inferenceMs =
          DateTime.now().difference(inferenceStart).inMilliseconds;

      AppLogger.log(
          'Inpainting inference completed in ${inferenceMs}ms using $lastExecutionProvider');

      if (result.isEmpty) {
        throw Exception('Inference returned empty result');
      }

      final output = result[0]!.value as List;
      final imgOut = convertNCHWtoImage(output);

      return InpaintingResult(
        bytes: Uint8List.fromList(img.encodeJpg(imgOut)),
        inferenceDurationMs: inferenceMs,
      );
    } finally {
      imageTensor.release();
      maskTensor.release();
      runOptions.release();
      session.release();

      if (result != null) {
        for (final value in result) {
          value?.release();
        }
      }
    }
  }
}
