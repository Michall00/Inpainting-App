import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' show Offset, Rect;

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

class SegmentationService {
  static bool _envInitialized = false;

  static void _ensureEnvironmentInitialized() {
    if (_envInitialized) return;
    OrtEnv.instance.init();
    _envInitialized = true;
  }

  static Future<Uint8List> segmentFromPoint({
    required File imageFile,
    required Offset clickPoint,
    required ByteData encoderData,
    required ByteData decoderData,
  }) async {
    _ensureEnvironmentInitialized();
    final imageBytes = await imageFile.readAsBytes();
    final image = img.decodeImage(imageBytes)!;
    final pixels = image.getBytes(order: img.ChannelOrder.rgb);

    final imgFloat = Float32List(pixels.length);
    for (int i = 0; i < pixels.length; i++) {
      imgFloat[i] = pixels[i].toDouble();
    }

    final encoderSession = OrtSession.fromBuffer(
      encoderData.buffer.asUint8List(),
      OrtSessionOptions(),
    );

    final encoderInput = OrtValueTensor.createTensorWithDataList(
      imgFloat,
      [image.height, image.width, 3],
    );

    final embeddings = encoderSession.run(
      OrtRunOptions(),
      {'input_image': encoderInput},
      ['image_embeddings'],
    );

    encoderInput.release();
    encoderSession.release();

    final imageEmbeddings = embeddings[0]!;

    final coords = Float32List.fromList([
      clickPoint.dx * (1024 / image.width),
      clickPoint.dy * (1024 / image.height),
      0.0,
      0.0,
    ]);
    final labels = Float32List.fromList([1.0, -1.0]);

    final decoderSession = OrtSession.fromBuffer(
      decoderData.buffer.asUint8List(),
      OrtSessionOptions(),
    );

    final pointCoords =
        OrtValueTensor.createTensorWithDataList(coords, [1, 2, 2]);
    final pointLabels = OrtValueTensor.createTensorWithDataList(labels, [1, 2]);
    final maskInput = OrtValueTensor.createTensorWithDataList(
        Float32List(1 * 1 * 256 * 256), [1, 1, 256, 256]);
    final hasMaskInput = OrtValueTensor.createTensorWithDataList(
        Float32List.fromList([0.0]), [1]);
    final origImSize = OrtValueTensor.createTensorWithDataList(
        Float32List.fromList([image.height.toDouble(), image.width.toDouble()]),
        [2]);

    late final Uint8List encodedMask;
    try {
      final decoderInputs = {
        'image_embeddings': imageEmbeddings,
        'point_coords': pointCoords,
        'point_labels': pointLabels,
        'mask_input': maskInput,
        'has_mask_input': hasMaskInput,
        'orig_im_size': origImSize,
      };

      final maskOutput =
          decoderSession.run(OrtRunOptions(), decoderInputs, ['masks']);
      try {
        final masksOrt = maskOutput[0]!;
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
      } finally {
        for (final value in maskOutput) {
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

    return encodedMask;
  }

  static Future<Uint8List> segmentFromBbox({
    required File imageFile,
    required Rect bboxPx,
    required ByteData encoderData,
    required ByteData decoderData,
  }) async {
    _ensureEnvironmentInitialized();
    final imageBytes = await imageFile.readAsBytes();
    final image = img.decodeImage(imageBytes)!;
    final pixels = image.getBytes(order: img.ChannelOrder.rgb);

    final imgFloat = Float32List(pixels.length);
    for (int i = 0; i < pixels.length; i++) {
      imgFloat[i] = pixels[i].toDouble();
    }

    final encoderSession = OrtSession.fromBuffer(
      encoderData.buffer.asUint8List(),
      OrtSessionOptions(),
    );

    final encoderInput = OrtValueTensor.createTensorWithDataList(
      imgFloat,
      [image.height, image.width, 3],
    );

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

    final x0 = (bboxPx.left).clamp(0.0, image.width.toDouble()) * sx;
    final y0 = (bboxPx.top).clamp(0.0, image.height.toDouble()) * sy;
    final x1 = (bboxPx.right).clamp(0.0, image.width.toDouble()) * sx;
    final y1 = (bboxPx.bottom).clamp(0.0, image.height.toDouble()) * sy;

    final pointCoords = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList([x0, y0, x1, y1]),
      [1, 2, 2],
    );
    final pointLabels = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList([2.0, 3.0]),
      [1, 2],
    );

    final maskInput = OrtValueTensor.createTensorWithDataList(
      Float32List(1 * 1 * 256 * 256),
      [1, 1, 256, 256],
    );
    final hasMaskInput = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList([0.0]),
      [1],
    );
    final origImSize = OrtValueTensor.createTensorWithDataList(
      Float32List.fromList([image.height.toDouble(), image.width.toDouble()]),
      [2],
    );

    final decoderSession = OrtSession.fromBuffer(
      decoderData.buffer.asUint8List(),
      OrtSessionOptions(),
    );

    late final Uint8List encodedMask;
    try {
      final decoderInputs = {
        'image_embeddings': imageEmbeddings,
        'point_coords': pointCoords,
        'point_labels': pointLabels,
        'mask_input': maskInput,
        'has_mask_input': hasMaskInput,
        'orig_im_size': origImSize,
      };

      final maskOutput = decoderSession.run(
        OrtRunOptions(),
        decoderInputs,
        ['masks'],
      );
      try {
        final masksOrt = maskOutput[0]!;
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
      } finally {
        for (final value in maskOutput) {
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

    return encodedMask;
  }
}
