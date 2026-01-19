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
        message:
            'This ${isPositive ? 'positive' : 'negative'} point is already added.',
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
    final overlay = img.Image(
      width: baseImage.width,
      height: baseImage.height,
      numChannels: 4,
    );
    final width = overlay.width;
    final height = overlay.height;
    const light = 236;
    const dark = 210;
    const cellSize = 6;
    const tintAlpha = 0.28;
    const glowR = 255;
    const glowG = 70;
    const glowB = 210;
    const edgeStrongAlpha = 235;
    const edgeMidAlpha = 185;
    const edgeSoftAlpha = 135;
    const fillAlpha = 90;

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
        final isEdgeStrong = hasOutsideNeighbor(x, y, 2);
        final isEdgeMid = !isEdgeStrong && hasOutsideNeighbor(x, y, 5);
        final isEdgeSoft =
            !isEdgeStrong && !isEdgeMid && hasOutsideNeighbor(x, y, 8);
        final alpha = isEdgeStrong
            ? edgeStrongAlpha
            : (isEdgeMid
                ? edgeMidAlpha
                : (isEdgeSoft ? edgeSoftAlpha : fillAlpha));
        final isLight = ((x ~/ cellSize) + (y ~/ cellSize)) % 2 == 0;
        final baseShade = isLight ? light : dark;
        final localTintAlpha = isEdgeStrong
            ? 0.55
            : (isEdgeMid ? 0.45 : (isEdgeSoft ? 0.36 : tintAlpha));
        final tintR =
            (baseShade * (1 - localTintAlpha) + glowR * localTintAlpha).round();
        final tintG =
            (baseShade * (1 - localTintAlpha) + glowG * localTintAlpha).round();
        final tintB =
            (baseShade * (1 - localTintAlpha) + glowB * localTintAlpha).round();
        final basePixel = baseImage.getPixel(x, y);
        final baseR = basePixel.r;
        final baseG = basePixel.g;
        final baseB = basePixel.b;
        final overlayR = isEdgeStrong ? glowR : tintR;
        final overlayG = isEdgeStrong ? glowG : tintG;
        final overlayB = isEdgeStrong ? glowB : tintB;
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
