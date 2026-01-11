import 'dart:io';
import 'dart:typed_data';
import 'dart:ui';

import 'package:image/image.dart' as img;

import '../../services/execution_provider.dart';
import '../../services/inpainting_service.dart';
import '../../services/segmentation_service.dart';
import 'inpainting_pipeline.dart';
import 'inpainting_session_state.dart';
import 'inpainting_types.dart';
import 'model_info.dart';

class InpaintingController {
  final InpaintingSessionState session = InpaintingSessionState();

  SegmentationPrecision segmentationPrecision = SegmentationPrecision.fp32;
  InpaintingModel inpaintingModel = InpaintingModel.fp32;
  ExecutionProvider executionProvider = ExecutionProvider.auto;

  bool isSegmentationInProgress = false;
  bool isInpaintingInProgress = false;

  void resetSession({
    required File tempFile,
    required img.Image resized,
    required img.Image blankMask,
  }) {
    session.resetForNewImage(
      tempFile: tempFile,
      width: resized.width,
      height: resized.height,
      blankMask: blankMask,
    );
  }

  void setExecutionProvider(ExecutionProvider provider) {
    executionProvider = provider;
    InpaintingPipeline.setPreferredExecutionProvider(provider);
  }

  String get lastSegmentationExecutionProvider =>
      InpaintingPipeline.lastSegmentationExecutionProvider;

  String get lastInpaintingExecutionProvider =>
      InpaintingPipeline.lastInpaintingExecutionProvider;

  RefinementStatus validateRefinement({
    required Float32List? lowResMask,
    required bool isPositive,
    required Offset point,
    required List<Offset> positivePoints,
    required List<Offset> negativePoints,
  }) {
    return InpaintingPipeline.validateRefinement(
      lowResMask: lowResMask,
      isPositive: isPositive,
      point: point,
      positivePoints: positivePoints,
      negativePoints: negativePoints,
    );
  }

  Future<SegmentationResult> callSegmentation({
    required Rect? bbox,
    required List<Offset> positivePoints,
    required List<Offset> negativePoints,
    required Float32List? lowResMask,
  }) async {
    return InpaintingPipeline.callSegmentation(
      imageFile: session.image.file!,
      bbox: bbox,
      positivePoints: positivePoints,
      negativePoints: negativePoints,
      lowResMask: lowResMask,
      encoderAsset: segmentationPrecision.encoderAsset,
      decoderAsset: segmentationPrecision.decoderAsset,
    );
  }

  Future<void> applySegmentationResult(SegmentationResult result) async {
    if (session.image.file == null) return;
    await InpaintingPipeline.applySegmentationResult(
      imageFile: session.image.file!,
      result: result,
      onApply: (visuals) {
        session.mask.segmentationBytes = visuals.maskBytes;
        session.mask.image = visuals.maskImage;
        session.mask.previewBytes = visuals.overlayBytes;
        session.segmentation.lowResMaskInput = visuals.lowResMask;
      },
    );
  }

  Future<SegmentationResult> refineSegmentation({
    required Rect? segmentationImageRect,
    required List<Offset> positivePoints,
    required List<Offset> negativePoints,
    required Float32List lowResMask,
  }) async {
    return InpaintingPipeline.refineSegmentation(
      imageFile: session.image.file!,
      segmentationImageRect: segmentationImageRect,
      positivePoints: positivePoints,
      negativePoints: negativePoints,
      lowResMask: lowResMask,
      encoderAsset: segmentationPrecision.encoderAsset,
      decoderAsset: segmentationPrecision.decoderAsset,
    );
  }

  Future<InpaintingResult> runInpainting({
    required img.Image original,
    required img.Image mask,
  }) async {
    return InpaintingPipeline.runInpainting(
      original: original,
      mask: mask,
      modelAsset: inpaintingModel.asset,
    );
  }
}
