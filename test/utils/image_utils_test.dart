import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

import 'package:inpainting_app/utils/image_utils.dart';

void main() {
  group('bboxFromPoints', () {
    test('returns zero rect when no finite points', () {
      final rect = bboxFromPoints(const [Offset.infinite, Offset.infinite]);
      expect(rect, Rect.zero);
    });

    test('computes bounds from finite points', () {
      final rect = bboxFromPoints(const [
        Offset(10, 20),
        Offset(30, 5),
        Offset(25, 40),
      ]);
      expect(rect.left, 10);
      expect(rect.top, 5);
      expect(rect.right, 30);
      expect(rect.bottom, 40);
    });
  });

  group('dilateMask', () {
    test('expands black pixels within radius', () {
      final mask = img.Image(width: 5, height: 5, numChannels: 1);
      for (var y = 0; y < 5; y++) {
        for (var x = 0; x < 5; x++) {
          mask.setPixelRgba(x, y, 255, 255, 255, 255);
        }
      }
      mask.setPixelRgba(2, 2, 0, 0, 0, 255);

      final result = dilateMask(mask, radius: 1);

      expect(result.getPixel(2, 2).r, 0);
      expect(result.getPixel(1, 2).r, 0);
      expect(result.getPixel(2, 1).r, 0);
      expect(result.getPixel(3, 2).r, 0);
      expect(result.getPixel(2, 3).r, 0);
      expect(result.getPixel(0, 0).r, 255);
      expect(result.getPixel(4, 4).r, 255);
    });
  });
}
