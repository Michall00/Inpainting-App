import 'dart:math' as math;

import 'package:flutter/material.dart';

class MaskPainter extends CustomPainter {
  final List<Offset> points;
  final double strokeWidth;
  MaskPainter(this.points, {required this.strokeWidth});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red.withAlpha(128)
      ..strokeCap = StrokeCap.round
      ..strokeWidth = strokeWidth;

    for (int i = 0; i < points.length - 1; i++) {
      if (points[i] != Offset.infinite && points[i + 1] != Offset.infinite) {
        canvas.drawLine(points[i], points[i + 1], paint);
      }
    }
  }

  @override
  bool shouldRepaint(MaskPainter oldDelegate) =>
      oldDelegate.points != points ||
      oldDelegate.strokeWidth != strokeWidth;
}

class BBoxPainter extends CustomPainter {
  final Rect box;
  final Color color;

  BBoxPainter(this.box, {this.color = Colors.red});

  @override
  void paint(Canvas canvas, Size size) {
    if (box.isEmpty) return;
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = color;
    canvas.drawRect(box, paint);
  }

  @override
  bool shouldRepaint(covariant BBoxPainter oldDelegate) =>
      oldDelegate.box != box || oldDelegate.color != color;
}

class SquarePointPainter extends CustomPainter {
  final Offset point;
  final Size imageSize;
  final double size;
  final Color color;

  const SquarePointPainter({
    required this.point,
    required this.imageSize,
    this.size = 10.0,
    this.color = Colors.amber,
  });

  @override
  void paint(Canvas canvas, Size canvasSize) {
    if (!point.isFinite ||
        imageSize.width == 0 ||
        imageSize.height == 0 ||
        canvasSize.width == 0 ||
        canvasSize.height == 0) {
      return;
    }

    final scaleX = canvasSize.width / imageSize.width;
    final scaleY = canvasSize.height / imageSize.height;
    final scaledPoint = Offset(point.dx * scaleX, point.dy * scaleY);
    final scale = (scaleX + scaleY) / 2;
    final visualSize = math.max(4.0, size * scale);
    final rect = Rect.fromCenter(
      center: scaledPoint,
      width: visualSize,
      height: visualSize,
    );
    final paintFill = Paint()
      ..color = color.withValues(alpha: 0.9)
      ..style = PaintingStyle.fill;
    final paintBorder = Paint()
      ..color = const Color.fromARGB(255, 255, 0, 0).withValues(alpha: 0.7)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    canvas.drawRect(rect, paintFill);
    canvas.drawRect(rect, paintBorder);
  }

  @override
  bool shouldRepaint(SquarePointPainter oldDelegate) =>
      oldDelegate.point != point ||
      oldDelegate.size != size ||
      oldDelegate.color != color ||
      oldDelegate.imageSize != imageSize;
}
