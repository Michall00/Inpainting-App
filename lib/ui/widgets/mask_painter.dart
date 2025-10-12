import 'package:flutter/material.dart';

class MaskPainter extends CustomPainter {
  final List<Offset> points;
  MaskPainter(this.points);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red.withAlpha(128)
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 20;

    for (int i = 0; i < points.length - 1; i++) {
      if (points[i] != Offset.infinite && points[i + 1] != Offset.infinite) {
        canvas.drawLine(points[i], points[i + 1], paint);
      }
    }
  }

  @override
  bool shouldRepaint(MaskPainter oldDelegate) => true;
}

class BBoxPainter extends CustomPainter {
  final Rect box;
  BBoxPainter(this.box);

  @override
  void paint(Canvas canvas, Size size) {
    if (box.isEmpty) return;
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Colors.red;
    canvas.drawRect(box, paint);
  }

  @override
  bool shouldRepaint(covariant BBoxPainter oldDelegate) =>
      oldDelegate.box != box;
}
