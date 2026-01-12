import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:vector_math/vector_math_64.dart' show Matrix4;

import 'mask_painter.dart';

class InpaintingImageStack extends StatelessWidget {
  final File? imageFile;
  final Uint8List? previewMaskBytes;
  final int? imageWidth;
  final int? imageHeight;
  final List<Offset> points;
  final Animation<double> maskPulse;
  final GlobalKey imageKey;
  final GlobalKey interactiveViewerKey;
  final TransformationController transformationController;
  final ValueChanged<Size> onCanvasSize;
  final double Function(double drawW) brushWidthForDraw;
  final ValueChanged<TapDownDetails> onTapDown;
  final VoidCallback onTapCancel;
  final Future<void> Function(TapUpDetails details, double drawW, double drawH)
      onTapUp;
  final void Function(
    DragStartDetails details,
    double drawW,
    double drawH,
    double brushSceneWidth,
  ) onPanStart;
  final void Function(
    DragUpdateDetails details,
    double drawW,
    double drawH,
    double brushSceneWidth,
  ) onPanUpdate;
  final Future<void> Function() onPanEnd;

  const InpaintingImageStack({
    super.key,
    required this.imageFile,
    required this.previewMaskBytes,
    required this.imageWidth,
    required this.imageHeight,
    required this.points,
    required this.maskPulse,
    required this.imageKey,
    required this.interactiveViewerKey,
    required this.transformationController,
    required this.onCanvasSize,
    required this.brushWidthForDraw,
    required this.onTapDown,
    required this.onTapCancel,
    required this.onTapUp,
    required this.onPanStart,
    required this.onPanUpdate,
    required this.onPanEnd,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final maxW = constraints.maxWidth;
        final maxH = constraints.maxHeight;

        final imgW = imageWidth?.toDouble() ?? 256;
        final imgH = imageHeight?.toDouble() ?? 256;

        final scale = (maxW / imgW < maxH / imgH) ? maxW / imgW : maxH / imgH;
        final drawW = imgW * scale;
        final drawH = imgH * scale;
        onCanvasSize(Size(drawW, drawH));

        return Center(
          child: SizedBox(
            width: drawW,
            height: drawH,
            child: InteractiveViewer(
              key: interactiveViewerKey,
              transformationController: transformationController,
              minScale: 1.0,
              maxScale: 5.0,
              panEnabled: false,
              clipBehavior: Clip.none,
              child: ValueListenableBuilder<Matrix4>(
                valueListenable: transformationController,
                builder: (context, value, _) {
                  final brushSceneWidth = brushWidthForDraw(drawW);
                  return Stack(
                    children: [
                      Positioned.fill(
                        child: Image(
                          key: imageKey,
                          image: FileImage(imageFile!),
                          fit: BoxFit.contain,
                        ),
                      ),
                      if (previewMaskBytes != null)
                        Positioned.fill(
                          child: AnimatedBuilder(
                            animation: maskPulse,
                            builder: (context, _) {
                              final overlayOpacity =
                                  0.75 + 0.25 * maskPulse.value;
                              return Opacity(
                                opacity: overlayOpacity,
                                child: Image(
                                  image: MemoryImage(previewMaskBytes!),
                                  fit: BoxFit.contain,
                                ),
                              );
                            },
                          ),
                        ),
                      Positioned.fill(
                        child: GestureDetector(
                          behavior: HitTestBehavior.opaque,
                          onTapDown: onTapDown,
                          onTapCancel: onTapCancel,
                          onTapUp: (details) => onTapUp(details, drawW, drawH),
                          onPanStart: (details) =>
                              onPanStart(details, drawW, drawH, brushSceneWidth),
                          onPanUpdate: (details) =>
                              onPanUpdate(details, drawW, drawH, brushSceneWidth),
                          onPanEnd: (_) => onPanEnd(),
                          child: CustomPaint(
                            painter: MaskPainter(
                              points,
                              strokeWidth: brushSceneWidth,
                              pulse: maskPulse,
                            ),
                            size: Size(drawW, drawH),
                          ),
                        ),
                      ),
                    ],
                  );
                },
              ),
            ),
          ),
        );
      },
    );
  }
}
