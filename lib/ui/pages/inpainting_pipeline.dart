import 'dart:io';
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

import '../../services/execution_provider.dart';
import '../../services/inpainting_service.dart';
import '../../services/segmentation_service.dart';

class SegmentationVisuals {
  final Uint8List maskBytes;
  final img.Image maskImage;
  final Uint8List overlayBytes;
  final Float32List lowResMask;

  const SegmentationVisuals({
    required this.maskBytes,
    required this.maskImage,
    required this.overlayBytes,
    required this.lowResMask,
  });
}

class SegmentationRefinementUpdate {
  final List<Offset> positivePoints;
  final List<Offset> negativePoints;
  final bool added;

  const SegmentationRefinementUpdate({
    required this.positivePoints,
    required this.negativePoints,
    required this.added,
  });
}

class RefinementStatus {
  final SegmentationRefinementUpdate? update;
  final String? message;

  const RefinementStatus({
    this.update,
    this.message,
  });
}

class InpaintingPipeline {
  const InpaintingPipeline();

  static String get lastSegmentationExecutionProvider =>
      SegmentationService.lastExecutionProvider;

  static String get lastInpaintingExecutionProvider =>
      InpaintingService.lastExecutionProvider;

  static void setPreferredExecutionProvider(ExecutionProvider provider) {
    SegmentationService.preferredExecutionProvider = provider;
    InpaintingService.preferredExecutionProvider = provider;
  }

  static SegmentationRefinementUpdate updateRefinementPoints({
    required bool isPositive,
    required Offset point,
    required List<Offset> positivePoints,
    required List<Offset> negativePoints,
  }) {
    final updatedPositive = List<Offset>.from(positivePoints);
    final updatedNegative = List<Offset>.from(negativePoints);

    bool added = false;
    if (isPositive) {
      if (!updatedPositive.contains(point)) {
        updatedPositive.add(point);
        added = true;
      }
    } else {
      if (!updatedNegative.contains(point)) {
        updatedNegative.add(point);
        added = true;
      }
    }

    return SegmentationRefinementUpdate(
      positivePoints: updatedPositive,
      negativePoints: updatedNegative,
      added: added,
    );
  }

  static RefinementStatus validateRefinement({
    required Float32List? lowResMask,
    required bool isPositive,
    required Offset point,
    required List<Offset> positivePoints,
    required List<Offset> negativePoints,
  }) {
    if (lowResMask == null) {
      return const RefinementStatus(
        message:
            'Refinement skipped because low-res mask is unavailable for this point.',
      );
    }

    final update = updateRefinementPoints(
      isPositive: isPositive,
      point: point,
      positivePoints: positivePoints,
      negativePoints: negativePoints,
    );

    if (!update.added) {
      return RefinementStatus(
        message: 'This ${isPositive ? 'positive' : 'negative'} point is already added.',
      );
    }

    return RefinementStatus(update: update);
  }

  static Future<SegmentationResult> refineSegmentation({
    required File imageFile,
    required Rect? segmentationImageRect,
    required List<Offset> positivePoints,
    required List<Offset> negativePoints,
    required Float32List lowResMask,
    required String encoderAsset,
    required String decoderAsset,
  }) async {
    return callSegmentation(
      imageFile: imageFile,
      bbox: segmentationImageRect,
      positivePoints: positivePoints,
      negativePoints: negativePoints,
      lowResMask: lowResMask,
      encoderAsset: encoderAsset,
      decoderAsset: decoderAsset,
    );
  }

  static Future<InpaintingResult> runInpainting({
    required img.Image original,
    required img.Image mask,
    required String modelAsset,
  }) async {
    final modelData = await rootBundle.load(modelAsset);
    return InpaintingService.runInpainting(
      original: original,
      mask: mask,
      modelData: modelData,
    );
  }

  static Future<SegmentationResult> callSegmentation({
    required File imageFile,
    required Rect? bbox,
    required List<Offset> positivePoints,
    required List<Offset> negativePoints,
    required Float32List? lowResMask,
    required String encoderAsset,
    required String decoderAsset,
  }) async {
    final encoderData = await rootBundle.load(encoderAsset);
    final decoderData = await rootBundle.load(decoderAsset);

    return SegmentationService.segmentWithPoints(
      imageFile: imageFile,
      bboxPx: bbox,
      positivePoints: List<Offset>.from(positivePoints),
      negativePoints: List<Offset>.from(negativePoints),
      lowResMaskInput: lowResMask,
      encoderData: encoderData,
      decoderData: decoderData,
    );
  }

  static Future<SegmentationVisuals> applySegmentationResult({
    required File imageFile,
    required SegmentationResult result,
    required void Function(SegmentationVisuals visuals) onApply,
  }) async {
    final visuals = await prepareSegmentationVisuals(
      imageFile: imageFile,
      result: result,
    );
    onApply(visuals);
    return visuals;
  }

  static Future<SegmentationVisuals> prepareSegmentationVisuals({
    required File imageFile,
    required SegmentationResult result,
  }) async {
    final imageBytes = await imageFile.readAsBytes();
    final baseImage = img.decodeImage(imageBytes)!;
    final decodedMask = img.decodeImage(result.maskBytes)!;
    final overlay = _composeOverlay(baseImage, decodedMask);
    return SegmentationVisuals(
      maskBytes: result.maskBytes,
      maskImage: decodedMask,
      overlayBytes: Uint8List.fromList(img.encodePng(overlay)),
      lowResMask: result.lowResMask,
    );
  }

  static img.Image _composeOverlay(img.Image baseImage, img.Image mask) {
    final overlay = img.Image.from(baseImage);
    final width = overlay.width;
    final height = overlay.height;
    const light = 235;
    const dark = 215;
    const cellSize = 8;
    const tintAlpha = 0.18;
    const glowR = 235;
    const glowG = 80;
    const glowB = 230;
    const edgeAlpha = 230;
    const midAlpha = 150;
    const fillAlpha = 70;

    bool hasOutsideNeighbor(int x, int y, int radius) {
      for (int dy = -radius; dy <= radius; dy++) {
        final ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (int dx = -radius; dx <= radius; dx++) {
          final nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          if (mask.getPixel(nx, ny).r != 0) {
            return true;
          }
        }
      }
      return false;
    }

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (mask.getPixel(x, y).r != 0) continue;
        final isEdge = hasOutsideNeighbor(x, y, 3);
        final alpha = isEdge
            ? edgeAlpha
            : (hasOutsideNeighbor(x, y, 4) ? midAlpha : fillAlpha);
        final isLight = ((x ~/ cellSize) + (y ~/ cellSize)) % 2 == 0;
        final baseShade = isLight ? light : dark;
        final tintR = (baseShade * (1 - tintAlpha) + glowR * tintAlpha).round();
        final tintG = (baseShade * (1 - tintAlpha) + glowG * tintAlpha).round();
        final tintB = (baseShade * (1 - tintAlpha) + glowB * tintAlpha).round();
        final basePixel = baseImage.getPixel(x, y);
        final baseR = basePixel.r;
        final baseG = basePixel.g;
        final baseB = basePixel.b;
        final overlayR = isEdge ? glowR : tintR;
        final overlayG = isEdge ? glowG : tintG;
        final overlayB = isEdge ? glowB : tintB;
        final t = alpha / 255.0;
        final outR = (baseR * (1 - t) + overlayR * t).round();
        final outG = (baseG * (1 - t) + overlayG * t).round();
        final outB = (baseB * (1 - t) + overlayB * t).round();
        overlay.setPixelRgba(x, y, outR, outG, outB, 255);
      }
    }
    return overlay;
  }
}
