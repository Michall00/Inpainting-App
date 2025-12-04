import 'dart:io';
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

import '../utils/tensor_utils.dart';
import '../utils/app_logger.dart';

class SegmentationResult {
  final Uint8List maskBytes;
  final Float32List lowResMask;

  SegmentationResult({
    required this.maskBytes,
    required this.lowResMask,
  });
}

enum SegmentationMode {
  onnxCpu,
  onnxCoreML,
  prunedCoreML,
}

class SegmentationService {
  static bool _envInitialized = false;

  static void _ensureEnvironmentInitialized() {
    if (_envInitialized) return;
    OrtEnv.instance.init();
    _envInitialized = true;
  }

  static OrtSessionOptions _createOptions(SegmentationMode mode) {
    final options = OrtSessionOptions();
    if (mode == SegmentationMode.onnxCoreML ||
        mode == SegmentationMode.prunedCoreML) {
      try {
        options.appendCoreMLProvider(CoreMLFlags.useNone);
      } catch (_) {
        // CoreML provider not available; fallback handled by session builder.
      }
    }
    return options;
  }

  static OrtSession _createSession(
    ByteData modelData,
    SegmentationMode mode,
  ) {
    final buffer = modelData.buffer.asUint8List();

    // Try preferred provider first, then retry with default CPU-only options.
    try {
      return OrtSession.fromBuffer(buffer, _createOptions(mode));
    } catch (error) {
      if (mode == SegmentationMode.onnxCoreML ||
          mode == SegmentationMode.prunedCoreML) {
        AppLogger.log(
          'Primary CoreML session failed, retrying with CPU. Error: $error',
        );
        final cpuOptions = OrtSessionOptions();
        try {
          return OrtSession.fromBuffer(buffer, cpuOptions);
        } catch (_) {
          cpuOptions.release();
          rethrow;
        }
      }
      rethrow;
    }
  }

  static Future<SegmentationResult> segmentFromPoint({
    required File imageFile,
    required Offset clickPoint,
    required ByteData encoderData,
    required ByteData decoderData,
    SegmentationMode mode = SegmentationMode.onnxCpu,
  }) {
    return segmentWithPoints(
      imageFile: imageFile,
      bboxPx: null,
      positivePoints: [clickPoint],
      negativePoints: const <Offset>[],
      lowResMaskInput: null,
      encoderData: encoderData,
      decoderData: decoderData,
      mode: mode,
    );
  }

  static Future<SegmentationResult> segmentFromBbox({
    required File imageFile,
    required Rect bboxPx,
    required ByteData encoderData,
    required ByteData decoderData,
    SegmentationMode mode = SegmentationMode.onnxCpu,
  }) {
    return segmentWithPoints(
      imageFile: imageFile,
      bboxPx: bboxPx,
      positivePoints: const <Offset>[],
      negativePoints: const <Offset>[],
      lowResMaskInput: null,
      encoderData: encoderData,
      decoderData: decoderData,
      mode: mode,
    );
  }

  static Future<SegmentationResult> segmentWithPoints({
    required File imageFile,
    Rect? bboxPx,
    required List<Offset> positivePoints,
    required List<Offset> negativePoints,
    Float32List? lowResMaskInput,
    required ByteData encoderData,
    required ByteData decoderData,
    SegmentationMode mode = SegmentationMode.onnxCpu,
  }) async {
    _ensureEnvironmentInitialized();
    final imageBytes = await imageFile.readAsBytes();
    final image = img.decodeImage(imageBytes)!;

    final encoderSession = _createSession(encoderData, mode);

    final encoderInput = convertImageToFloatNCHW(image);

    final embeddings = encoderSession.run(
      OrtRunOptions(),
      {'input_image': encoderInput},
      ['image_embeddings'],
    );

    encoderInput.release();
    encoderSession.release();

    final imageEmbeddings = embeddings[0]!;

    final sx = 1024.0 / image.width;
    final sy = 1024.0 / image.height;

    final coords = <double>[];
    final labels = <double>[];

    if (bboxPx != null && !bboxPx.isEmpty) {
      final x0 = bboxPx.left.clamp(0.0, image.width.toDouble()).toDouble() * sx;
      final y0 = bboxPx.top.clamp(0.0, image.height.toDouble()).toDouble() * sy;
      final x1 =
          bboxPx.right.clamp(0.0, image.width.toDouble()).toDouble() * sx;
      final y1 =
          bboxPx.bottom.clamp(0.0, image.height.toDouble()).toDouble() * sy;
      coords.addAll([x0, y0, x1, y1]);
      labels.addAll([2.0, 3.0]);
    }

    for (final point in positivePoints) {
      final px = point.dx.clamp(0.0, image.width.toDouble()).toDouble() * sx;
      final py = point.dy.clamp(0.0, image.height.toDouble()).toDouble() * sy;
      coords.addAll([px, py]);
      labels.add(1.0);
    }

    for (final point in negativePoints) {
      final px = point.dx.clamp(0.0, image.width.toDouble()).toDouble() * sx;
      final py = point.dy.clamp(0.0, image.height.toDouble()).toDouble() * sy;
      coords.addAll([px, py]);
      labels.add(0.0);
    }

    if (coords.isEmpty) {
      imageEmbeddings.release();
      throw ArgumentError(
          'Segmentation requires at least one point or bounding box.');
    }

    final pointCount = coords.length ~/ 2;
    final pointCoords = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList(coords),
      [1, pointCount, 2],
    );
    final pointLabels = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList(labels),
      [1, pointCount],
    );

    const maskElements = 256 * 256;
    final maskData = Float32List(maskElements);
    var hasPreviousMask = false;
    if (lowResMaskInput != null && lowResMaskInput.length == maskElements) {
      maskData.setAll(0, lowResMaskInput);
      hasPreviousMask = true;
    }
    final maskInput = OrtValueTensor.createTensorWithDataList(
      maskData,
      [1, 1, 256, 256],
    );
    final hasMaskInput = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList([hasPreviousMask ? 1.0 : 0.0]),
      [1],
    );
    final origImSize = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList([image.height.toDouble(), image.width.toDouble()]),
      [2],
    );

    final decoderSession = _createSession(decoderData, mode);

    late final Uint8List encodedMask;
    var lowResMaskOutput = Float32List(maskElements);
    try {
      final decoderInputs = {
        'image_embeddings': imageEmbeddings,
        'point_coords': pointCoords,
        'point_labels': pointLabels,
        'mask_input': maskInput,
        'has_mask_input': hasMaskInput,
        'orig_im_size': origImSize,
      };

      final outputs = decoderSession.run(OrtRunOptions(), decoderInputs);
      try {
        if (outputs.isEmpty || outputs[0] == null) {
          throw StateError('Masks output was not produced by decoder.');
        }
        final masksOrt = outputs[0]!;
        final rawMask = masksOrt.value as List;
        final binary = <int>[];
        for (final row in rawMask[0][0] as List) {
          for (final v in row as List) {
            binary.add((v as double) > 0 ? 0 : 255);
          }
        }
        final maskBytes = Uint8List.fromList(binary);
        final mask = img.Image.fromBytes(
          width: image.width,
          height: image.height,
          bytes: maskBytes.buffer,
          numChannels: 1,
          format: img.Format.uint8,
        );
        encodedMask = Uint8List.fromList(img.encodePng(mask));

        dynamic lowResCandidate;
        if (outputs.length >= 3 && outputs[2] != null) {
          lowResCandidate = outputs[2]!.value;
        } else if (outputs.length >= 2 && outputs[1] != null) {
          lowResCandidate = outputs[1]!.value;
        }
        if (lowResCandidate != null) {
          final flattened = _extractLowResMask(lowResCandidate);
          if (flattened.length == maskElements) {
            lowResMaskOutput = flattened;
          }
        }
      } finally {
        for (final value in outputs) {
          value?.release();
        }
      }
    } finally {
      pointCoords.release();
      pointLabels.release();
      maskInput.release();
      hasMaskInput.release();
      origImSize.release();
      imageEmbeddings.release();
      decoderSession.release();
    }

    return SegmentationResult(
      maskBytes: encodedMask,
      lowResMask: lowResMaskOutput,
    );
  }

  static Float32List _extractLowResMask(dynamic value) {
    final collected = <double>[];
    void collect(dynamic item) {
      if (item is List) {
        for (final child in item) {
          collect(child);
        }
      } else if (item is num) {
        collected.add(item.toDouble());
      }
    }

    collect(value);
    final result = Float32List(collected.length);
    for (var i = 0; i < collected.length; i++) {
      result[i] = collected[i];
    }
    return result;
  }
}
