import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

import 'inpainting_types.dart';

class InpaintingSessionState {
  final ImageState image = ImageState();
  final MaskState mask = MaskState();
  final InteractionState interaction = InteractionState();
  final SegmentationState segmentation = SegmentationState();
  final CanvasState canvas = CanvasState();
  final OutputState output = OutputState();

  bool get hasManualDrawing =>
      interaction.points.any((offset) => offset.isFinite);

  void resetForNewImage({
    required File tempFile,
    required int width,
    required int height,
    required img.Image blankMask,
  }) {
    image.reset(
      file: tempFile,
      width: width,
      height: height,
    );
    mask.reset(blankMask);
    interaction.reset();
    segmentation.reset();
    canvas.reset();
    output.reset();
  }
}

class ImageState {
  File? file;
  int? width;
  int? height;

  void reset({
    required File file,
    required int width,
    required int height,
  }) {
    this.file = file;
    this.width = width;
    this.height = height;
  }
}

class MaskState {
  img.Image? image;
  Uint8List? previewBytes;
  Uint8List? segmentationBytes;

  void reset(img.Image blankMask) {
    image = blankMask;
    previewBytes = null;
    segmentationBytes = null;
  }
}

class InteractionState {
  final List<Offset> points = [];
  InteractionMode mode = InteractionMode.point;
  Offset? lastTapImagePoint;
  final List<Offset> positivePoints = [];
  final List<Offset> negativePoints = [];
  SegmentationPointMode pointMode = SegmentationPointMode.positive;

  void reset() {
    points.clear();
    lastTapImagePoint = null;
    mode = InteractionMode.point;
    positivePoints.clear();
    negativePoints.clear();
    pointMode = SegmentationPointMode.positive;
  }
}

class SegmentationState {
  Float32List? lowResMaskInput;
  Rect? segmentationImageRect;

  void reset() {
    lowResMaskInput = null;
    segmentationImageRect = null;
  }
}

class CanvasState {
  Size? size;

  void reset() {
    size = null;
  }
}

class OutputState {
  Uint8List? bytes;

  void reset() {
    bytes = null;
  }
}
