import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

import 'inpainting_types.dart';

class InpaintingSessionState {
  File? imageFile;
  int? imageWidth;
  int? imageHeight;

  img.Image? maskImage;
  Uint8List? previewMaskBytes;
  Uint8List? segmentationMask;

  final List<Offset> points = [];
  InteractionMode mode = InteractionMode.point;
  Offset? lastTapImagePoint;
  Rect? bbox;
  Size? canvasSize;

  Float32List? lowResMaskInput;
  Rect? segmentationImageRect;
  final List<Offset> positivePoints = [];
  final List<Offset> negativePoints = [];
  SegmentationPointMode pointMode = SegmentationPointMode.positive;

  Uint8List? outputBytes;
  bool get hasManualDrawing => points.any((offset) => offset.isFinite);

  void resetForNewImage({
    required File tempFile,
    required int width,
    required int height,
    required img.Image blankMask,
  }) {
    imageFile = tempFile;
    imageWidth = width;
    imageHeight = height;
    outputBytes = null;
    previewMaskBytes = null;
    segmentationMask = null;
    maskImage = blankMask;
    points.clear();
    lastTapImagePoint = null;
    bbox = null;
    canvasSize = null;
    mode = InteractionMode.point;
    lowResMaskInput = null;
    segmentationImageRect = null;
    positivePoints.clear();
    negativePoints.clear();
    pointMode = SegmentationPointMode.positive;
  }
}
