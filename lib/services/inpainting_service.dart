import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';
import '../utils/tensor_utils.dart';

class InpaintingService {
  static bool _envInitialized = false;

  static void _ensureEnvironmentInitialized() {
    if (_envInitialized) return;
    OrtEnv.instance.init();
    _envInitialized = true;
  }

  static OrtSessionOptions _createSessionOptions() {
    final options = OrtSessionOptions();
    try {
      options.appendCoreMLProvider(CoreMLFlags.useNone);
    } catch (_) {
      // CoreML provider not available; CPU fallback will handle execution.
    }
    options.appendCPUProvider(CPUFlags.useArena);
    return options;
  }

  static Future<Uint8List> runInpainting({
    required img.Image original,
    required img.Image mask,
    required ByteData modelData,
  }) async {
    _ensureEnvironmentInitialized();
    final session = OrtSession.fromBuffer(
      modelData.buffer.asUint8List(),
      _createSessionOptions(),
    );

    final imageTensor = convertImageToUint8NCHW(original);
    final maskTensor = convertMaskToUint8NCHW(mask);

    final result = session.run(
      OrtRunOptions(),
      {'image': imageTensor, 'mask': maskTensor},
      ['result'],
    );

    imageTensor.release();
    maskTensor.release();
    session.release();

    final output = result[0]!.value as List;
    final imgOut = convertNCHWtoImage(output);
    return Uint8List.fromList(img.encodeJpg(imgOut));
  }
}
