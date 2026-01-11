import 'dart:math' as math;

import 'package:flutter/material.dart';

class MaskPainter extends CustomPainter {
  final List<Offset> points;
  final double strokeWidth;
  final Animation<double> pulse;

  MaskPainter(
    this.points, {
    required this.strokeWidth,
    required this.pulse,
  }) : super(repaint: pulse);

  @override
  void paint(Canvas canvas, Size size) {
    final t = pulse.value;
    final pulseAlpha =
        (0.45 + 0.25 * math.sin(t * math.pi * 2)).clamp(0.15, 0.7).toDouble();
    final glowAlpha =
        (0.18 + 0.12 * math.cos(t * math.pi * 2)).clamp(0.08, 0.35).toDouble();

    final glowPaint = Paint()
      ..color = const Color.fromARGB(255, 72, 167, 255).withOpacity(glowAlpha)
      ..strokeCap = StrokeCap.round
      ..strokeWidth = strokeWidth * 1.8;

    final corePaint = Paint()
      ..color = const Color.fromARGB(255, 72, 167, 255).withOpacity(pulseAlpha)
      ..strokeCap = StrokeCap.round
      ..strokeWidth = strokeWidth * 0.9;

    for (int i = 0; i < points.length - 1; i++) {
      if (points[i] != Offset.infinite && points[i + 1] != Offset.infinite) {
        canvas.drawLine(points[i], points[i + 1], glowPaint);
        canvas.drawLine(points[i], points[i + 1], corePaint);
      }
    }
  }

  @override
  bool shouldRepaint(MaskPainter oldDelegate) =>
      oldDelegate.points != points ||
      oldDelegate.strokeWidth != strokeWidth ||
      oldDelegate.pulse != pulse;
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

class SegmentationHintsPainter extends CustomPainter {
  final List<Offset> positives;
  final List<Offset> negatives;
  final Size imageSize;

  const SegmentationHintsPainter({
    required this.positives,
    required this.negatives,
    required this.imageSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (imageSize.width == 0 ||
        imageSize.height == 0 ||
        size.width == 0 ||
        size.height == 0) {
      return;
    }

    final scaleX = size.width / imageSize.width;
    final scaleY = size.height / imageSize.height;
    final scale = (scaleX + scaleY) / 2;
    final radius = math.max(6.0, 8.0 * scale);
    final symbolStroke = math.max(2.0, radius * 0.35);

    final positiveFill = Paint()
      ..color = Colors.greenAccent.withOpacity(0.9)
      ..style = PaintingStyle.fill;
    final positiveBorder = Paint()
      ..color = Colors.green.shade900
      ..style = PaintingStyle.stroke
      ..strokeWidth = math.max(2.0, radius * 0.2);
    final negativeFill = Paint()
      ..color = Colors.redAccent.withOpacity(0.9)
      ..style = PaintingStyle.fill;
    final negativeBorder = Paint()
      ..color = Colors.red.shade900
      ..style = PaintingStyle.stroke
      ..strokeWidth = math.max(2.0, radius * 0.2);
    final symbolPaint = Paint()
      ..color = Colors.white
      ..strokeWidth = symbolStroke
      ..strokeCap = StrokeCap.round;

    for (final point in positives) {
      if (!point.isFinite) continue;
      final translated = Offset(point.dx * scaleX, point.dy * scaleY);
      canvas.drawCircle(translated, radius, positiveFill);
      canvas.drawCircle(translated, radius, positiveBorder);
      canvas.drawLine(
        Offset(translated.dx - radius / 2, translated.dy),
        Offset(translated.dx + radius / 2, translated.dy),
        symbolPaint,
      );
      canvas.drawLine(
        Offset(translated.dx, translated.dy - radius / 2),
        Offset(translated.dx, translated.dy + radius / 2),
        symbolPaint,
      );
    }

    for (final point in negatives) {
      if (!point.isFinite) continue;
      final translated = Offset(point.dx * scaleX, point.dy * scaleY);
      canvas.drawCircle(translated, radius, negativeFill);
      canvas.drawCircle(translated, radius, negativeBorder);
      canvas.drawLine(
        Offset(translated.dx - radius / 2, translated.dy),
        Offset(translated.dx + radius / 2, translated.dy),
        symbolPaint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant SegmentationHintsPainter oldDelegate) {
    return !_listEquals(oldDelegate.positives, positives) ||
        !_listEquals(oldDelegate.negatives, negatives) ||
        oldDelegate.imageSize != imageSize;
  }

  bool _listEquals(List<Offset> a, List<Offset> b) {
    if (identical(a, b)) return true;
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}
