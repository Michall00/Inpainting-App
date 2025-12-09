import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';
import '../utils/tensor_utils.dart';
import '../utils/app_logger.dart';

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

  static void _ensureEnvironmentInitialized() {
    if (_envInitialized) return;
    OrtEnv.instance.init();
    _envInitialized = true;
  }

  static OrtSessionOptions _createSessionOptions({bool preferCoreML = true}) {
    final options = OrtSessionOptions();
    if (preferCoreML) {
      try {
        options.appendCoreMLProvider(CoreMLFlags.useNone);
      } catch (_) {
        // CoreML provider not available; fallback handled by session builder.
      }
    }
    return options;
  }

  static OrtSession _createSession(ByteData modelData) {
    final buffer = modelData.buffer.asUint8List();

    // Try CoreML first, then fall back to CPU-only session options.
    try {
      final session = OrtSession.fromBuffer(
        buffer,
        _createSessionOptions(preferCoreML: true),
      );
      lastExecutionProvider = 'coreml';
      return session;
    } catch (error) {
      AppLogger.log(
        'Primary CoreML session failed, retrying with CPU. Error: $error',
      );
      final cpuOptions = _createSessionOptions(preferCoreML: false);
      try {
        final session = OrtSession.fromBuffer(buffer, cpuOptions);
        lastExecutionProvider = 'cpu';
        return session;
      } catch (_) {
        cpuOptions.release();
        rethrow;
      }
    }
  }

  static Future<InpaintingResult> runInpainting({
    required img.Image original,
    required img.Image mask,
    required ByteData modelData,
  }) async {
    _ensureEnvironmentInitialized();
    OrtSession session;
    try {
      session = _createSession(modelData);
      lastExecutionProvider = 'coreml';
    } catch (_) {
      lastExecutionProvider = 'cpu';
      rethrow;
    }

    final imageTensor = convertImageToUint8NCHW(original);
    final maskTensor = convertMaskToUint8NCHW(mask);

    final runOptions = OrtRunOptions();
    final inferenceStart = DateTime.now();
    final result = session.run(
      runOptions,
      {'image': imageTensor, 'mask': maskTensor},
      ['result'],
    );
    final inferenceMs =
        DateTime.now().difference(inferenceStart).inMilliseconds;
    AppLogger.log('Inpainting model inference completed in ${inferenceMs}ms');

    imageTensor.release();
    maskTensor.release();
    runOptions.release();
    session.release();

    final output = result[0]!.value as List;
    final imgOut = convertNCHWtoImage(output);
    return InpaintingResult(
      bytes: Uint8List.fromList(img.encodeJpg(imgOut)),
      inferenceDurationMs: inferenceMs,
    );
  }
}
