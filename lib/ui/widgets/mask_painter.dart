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
