import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

import 'package:inpainting_app/utils/tensor_utils.dart';

bool _isOnnxAvailable() {
  try {
    OrtEnv.instance;
    return true;
  } catch (_) {
    return false;
  }
}

final bool _onnxAvailable = _isOnnxAvailable();

List<num> _flattenTensorValue(dynamic value) {
  if (value is Uint8List || value is Float32List) {
    return value.cast<num>();
  }
  if (value is List) {
    final out = <num>[];
    void walk(dynamic v) {
      if (v is List) {
        for (final entry in v) {
          walk(entry);
        }
      } else if (v is Uint8List || v is Float32List) {
        out.addAll(v.cast<num>());
      } else if (v is num) {
        out.add(v);
      } else {
        throw StateError('Unsupported tensor value type: ${v.runtimeType}');
      }
    }

    walk(value);
    return out;
  }
  throw StateError('Unsupported tensor value type: ${value.runtimeType}');
}

void main() {
  group('convertImageToUint8NCHW', () {
    test('reorders pixels into NCHW', () {
      final image = img.Image(width: 2, height: 2);
      image.setPixelRgb(0, 0, 10, 20, 30);
      image.setPixelRgb(1, 0, 40, 50, 60);
      image.setPixelRgb(0, 1, 70, 80, 90);
      image.setPixelRgb(1, 1, 100, 110, 120);

      final tensor = convertImageToUint8NCHW(image);
      expect(tensor, isA<OrtValueTensor>());

      final data = _flattenTensorValue(tensor.value);
      expect(data, hasLength(12));
      expect(
        data,
        [10, 40, 70, 100, 20, 50, 80, 110, 30, 60, 90, 120],
      );
    }, skip: _onnxAvailable ? false : 'ONNX Runtime library not available.');
  });

  group('convertImageToFloatNCHW', () {
    test('normalizes pixels into float NCHW', () {
      final image = img.Image(width: 1, height: 1);
      image.setPixelRgb(0, 0, 100, 110, 120);

      final tensor = convertImageToFloatNCHW(image);
      expect(tensor, isA<OrtValueTensor>());

      final data = _flattenTensorValue(tensor.value);
      expect(data, hasLength(3));

      final expected = [
        (100 - 123.675) / 58.395,
        (110 - 116.28) / 57.12,
        (120 - 103.53) / 57.375,
      ];
      for (var i = 0; i < expected.length; i++) {
        expect((data[i] as num).toDouble(),
            closeTo(expected[i], 1e-4),
            reason: 'Channel $i differs');
      }
    }, skip: _onnxAvailable ? false : 'ONNX Runtime library not available.');
  });

  group('convertMaskToUint8NCHW', () {
    test('converts single channel mask to NCHW', () {
      final mask = img.Image(width: 2, height: 2, numChannels: 1);
      mask.setPixelRgba(0, 0, 0, 0, 0, 255);
      mask.setPixelRgba(1, 0, 64, 64, 64, 255);
      mask.setPixelRgba(0, 1, 128, 128, 128, 255);
      mask.setPixelRgba(1, 1, 255, 255, 255, 255);

      final tensor = convertMaskToUint8NCHW(mask);
      expect(tensor, isA<OrtValueTensor>());

      final data = _flattenTensorValue(tensor.value);
      expect(data, [0, 64, 128, 255]);
    }, skip: _onnxAvailable ? false : 'ONNX Runtime library not available.');
  });

  group('convertNCHWtoImage', () {
    test('recreates image from NCHW list', () {
      final data = [
        [
          [
            [10, 40],
            [70, 100],
          ],
          [
            [20, 50],
            [80, 110],
          ],
          [
            [30, 60],
            [90, 120],
          ],
        ],
      ];

      final image = convertNCHWtoImage(data);
      final p00 = image.getPixel(0, 0);
      final p10 = image.getPixel(1, 0);
      final p01 = image.getPixel(0, 1);
      final p11 = image.getPixel(1, 1);

      expect([p00.r, p00.g, p00.b], [10, 20, 30]);
      expect([p10.r, p10.g, p10.b], [40, 50, 60]);
      expect([p01.r, p01.g, p01.b], [70, 80, 90]);
      expect([p11.r, p11.g, p11.b], [100, 110, 120]);
    }, skip: _onnxAvailable ? false : 'ONNX Runtime library not available.');
  });
}
