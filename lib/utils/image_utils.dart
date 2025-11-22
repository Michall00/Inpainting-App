import 'package:image/image.dart' as img;
import 'dart:ui';

img.Image dilateMask(img.Image mask, {int radius = 5}) {
  final w = mask.width;
  final h = mask.height;
  final result = img.Image(width: w, height: h, numChannels: 1);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      result.setPixelRgba(x, y, 255, 255, 255, 255);
    }
  }

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      if (mask.getPixel(x, y).r == 0) {
        for (int dy = -radius; dy <= radius; dy++) {
          for (int dx = -radius; dx <= radius; dx++) {
            final nx = (x + dx).clamp(0, w - 1);
            final ny = (y + dy).clamp(0, h - 1);
            result.setPixelRgba(nx, ny, 0, 0, 0, 255);
          }
        }
      }
    }
  }

  return result;
}

Rect bboxFromPoints(List<Offset> points) {
  double minX = double.infinity, minY = double.infinity;
  double maxX = -double.infinity, maxY = -double.infinity;
  bool any = false;

  for (int i = 0; i < points.length; i++) {
    final p = points[i];
    if (!p.isFinite) continue;
    any = true;
    if (p.dx < minX) minX = p.dx;
    if (p.dy < minY) minY = p.dy;
    if (p.dx > maxX) maxX = p.dx;
    if (p.dy > maxY) maxY = p.dy;
  }

  if (!any) {
    return Rect.zero;
  }

  return Rect.fromLTRB(minX, minY, maxX, maxY);
}
