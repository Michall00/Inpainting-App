import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import '../widgets/inpainting_action_bar.dart';
import '../widgets/inpainting_background.dart';
import '../widgets/inpainting_empty_state.dart';
import '../widgets/inpainting_header.dart';
import '../widgets/inpainting_image_stack.dart';
import '../../services/image_service.dart';
import '../../services/execution_provider.dart';
import '../../utils/app_logger.dart';
import '../../utils/image_utils.dart';
import 'package:image_gallery_saver/image_gallery_saver.dart';
import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:vector_math/vector_math_64.dart' show Matrix4;
import 'inpainting_session_state.dart';
import 'inpainting_controller.dart';
import 'model_info.dart';
import 'inpainting_types.dart';

class InpaintingPage extends StatefulWidget {
  const InpaintingPage({super.key});

  @override
  State<InpaintingPage> createState() => _InpaintingPageState();
}

class _InpaintingPageState extends State<InpaintingPage>
    with SingleTickerProviderStateMixin {
  final InpaintingController _controller = InpaintingController();
  final GlobalKey _imageKey = GlobalKey();
  final GlobalKey _interactiveViewerKey = GlobalKey();
  final TransformationController _transformationController =
      TransformationController();
  InpaintingSessionState get _session => _controller.session;
  late final AnimationController _maskPulseController;
  late final Animation<double> _maskPulse;
  Offset? _tapDownGlobal;
  DateTime? _tapDownTime;

  static const double _baseBrushSceneWidth = 20.0;
  bool get _hasManualDrawing => _session.hasManualDrawing;

  String get _executionProviderLabel {
    return executionProviderLabel(_controller.executionProvider);
  }

  String get _executionProviderValue {
    return executionProviderValue(_controller.executionProvider);
  }

  @override
  void initState() {
    super.initState();
    _maskPulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1800),
    );
    _maskPulse = CurvedAnimation(
      parent: _maskPulseController,
      curve: Curves.easeInOut,
    );
    _maskPulseController.repeat(reverse: true);
  }

  void _updateMaskPulse() {
    final shouldPulse = !_controller.isSegmentationInProgress && !_controller.isInpaintingInProgress;
    if (shouldPulse) {
      if (!_maskPulseController.isAnimating) {
        _maskPulseController.repeat(reverse: true);
      }
    } else {
      if (_maskPulseController.isAnimating) {
        _maskPulseController.stop();
      }
    }
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    final file = File(picked.path);
    final resized = await ImageService.decodeAndResize(file, 1024);
    if (resized == null) return;

    final resultBytes = Uint8List.fromList(img.encodePng(resized));
    final tempFile = await ImageService.saveTempImage(resultBytes, 'input.png');

    _startNewEditingSession(resized: resized, tempFile: tempFile);
  }

  Future<void> _saveImageToGallery(Uint8List imageBytes) async {
    try {
      final result = await ImageGallerySaver.saveImage(
        imageBytes,
        quality: 100,
        name: "inpainted_${DateTime.now().millisecondsSinceEpoch}.png",
      );
      if (result['isSuccess'] != true) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Error saving image to gallery')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error saving image: $e')),
      );
    }
  }

  Future<void> _pickImageFromCamera() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.camera);

    if (pickedFile == null) return;

    final file = File(pickedFile.path);
    final resized = await ImageService.decodeAndResize(file, 1024);
    if (resized == null) return;

    final resultBytes = Uint8List.fromList(img.encodePng(resized));
    final tempFile = await ImageService.saveTempImage(resultBytes, 'input.png');

    _startNewEditingSession(resized: resized, tempFile: tempFile);
  }

  void _startNewEditingSession({
    required img.Image resized,
    required File tempFile,
  }) {
    final blank = _createBlankMaskImage(resized.width, resized.height);

    _transformationController.value = Matrix4.identity();
    _controller.resetSessionWithFlags(
      tempFile: tempFile,
      resized: resized,
      blankMask: blank,
    );
    _updateMaskPulse();
  }

  Future<void> _runSegmentationFromClick(Offset point) async {
    if (_session.image.file == null || _controller.isSegmentationInProgress) {
      return;
    }
    AppLogger.log('Segmentation from point requested: $point');

    _controller.setSegmentationInProgress(true);
    _updateMaskPulse();
    try {
      final segmentationStart = DateTime.now();
      FirebaseAnalytics.instance.logEvent(
        name: 'segmentation_started',
        parameters: {
          'x': point.dx,
          'y': point.dy,
          'device': AppLogger.deviceInfo,
          'os_version': AppLogger.osVersion,
        },
      );

      final newPositive = [point];
      final newNegative = <Offset>[];

      final result = await _callSegmentation(
        bbox: null,
        positivePoints: newPositive,
        negativePoints: newNegative,
        lowResMask: null,
      );

      final segmentationEnd = DateTime.now();
      final durationMs =
          segmentationEnd.difference(segmentationStart).inMilliseconds;

      FirebaseAnalytics.instance.logEvent(
        name: 'segmentation_completed',
        parameters: {
          'segmentation_duration_ms': durationMs,
          'encoder_inference_ms': result.encoderInferenceMs,
          'decoder_inference_ms': result.decoderInferenceMs,
          'model': _controller.segmentationPrecision.modelName,
          'quantization': _controller.segmentationPrecision.quantizationType,
          'environment': _controller.lastSegmentationExecutionProvider,
          'device': AppLogger.deviceInfo,
          'os_version': AppLogger.osVersion,
        },
      );
      AppLogger.log(
        'Segmentation from bbox completed in ${durationMs}ms '
        'encoder=${result.encoderInferenceMs}ms '
        'decoder=${result.decoderInferenceMs}ms '
        'model=${_controller.segmentationPrecision.modelName} '
        'quantization=${_controller.segmentationPrecision.quantizationType} '
        'env=${_controller.lastSegmentationExecutionProvider}',
      );

      await _applySegmentationResult(result);
      if (!mounted) return;
      _controller.update(() {
        _session.interaction.positivePoints
          ..clear()
          ..addAll(newPositive);
        _session.interaction.negativePoints.clear();
        _session.segmentation.segmentationImageRect = null;
        _session.interaction.points.clear();
        _session.interaction.lastTapImagePoint = null;
        _session.interaction.pointMode = SegmentationPointMode.positive;
      });
    } catch (error, stackTrace) {
      AppLogger.error(
        'Segmentation from point failed',
        error,
        stackTrace,
      );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Segmentation failed: $error')),
      );
    } finally {
      if (mounted) _controller.setSegmentationInProgress(false);
      _updateMaskPulse();
    }
  }

  img.Image _createBlankMaskImage(int width, int height) {
    final blank = img.Image(width: width, height: height, numChannels: 1);
    blank.getBytes().fillRange(0, width * height, 255);
    return blank;
  }

  void _clearManualMask() {
    if (_session.image.width == null || _session.image.height == null) return;
    _controller.update(() {
      _session.interaction.points.clear();
      if (_session.mask.segmentationBytes != null) {
        _session.mask.image = img.decodeImage(_session.mask.segmentationBytes!)!;
      } else {
        _session.mask.image =
            _createBlankMaskImage(_session.image.width!, _session.image.height!);
      }
    });
  }

  Future<void> _runSegmentationFromBbox(Rect bbox) async {
    if (_session.image.file == null || _controller.isSegmentationInProgress) return;
    AppLogger.log('Segmentation from bbox requested: $bbox');

    _controller.setSegmentationInProgress(true);
    try {
      final segmentationStart = DateTime.now();
      FirebaseAnalytics.instance.logEvent(
        name: 'segmentation_started',
        parameters: {
          'x1': bbox.left,
          'y1': bbox.top,
          'x2': bbox.right,
          'y2': bbox.bottom,
          'device': AppLogger.deviceInfo,
          'os_version': AppLogger.osVersion,
        },
      );

      final result = await _callSegmentation(
        bbox: bbox,
        positivePoints: const [],
        negativePoints: const [],
        lowResMask: null,
      );

      final segmentationEnd = DateTime.now();
      final durationMs =
          segmentationEnd.difference(segmentationStart).inMilliseconds;

      FirebaseAnalytics.instance.logEvent(
        name: 'segmentation_completed',
        parameters: {
          'segmentation_duration_ms': durationMs,
          'encoder_inference_ms': result.encoderInferenceMs,
          'decoder_inference_ms': result.decoderInferenceMs,
          'model': _controller.segmentationPrecision.modelName,
          'quantization': _controller.segmentationPrecision.quantizationType,
          'environment': _controller.lastSegmentationExecutionProvider,
          'device': AppLogger.deviceInfo,
          'os_version': AppLogger.osVersion,
        },
      );
      AppLogger.log(
        'Segmentation from bbox completed in ${durationMs}ms',
      );

      await _applySegmentationResult(result);
      if (!mounted) return;
      _controller.update(() {
        _session.interaction.positivePoints.clear();
        _session.interaction.negativePoints.clear();
        _session.segmentation.segmentationImageRect = bbox;
        _session.interaction.points.clear();
        _session.interaction.pointMode = SegmentationPointMode.positive;
      });
    } catch (error, stackTrace) {
      AppLogger.error(
        'Segmentation from bbox failed',
        error,
        stackTrace,
      );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Segmentation failed: $error')),
      );
    } finally {
      if (mounted) _controller.setSegmentationInProgress(false);
      _updateMaskPulse();
    }
  }

  Future<dynamic> _callSegmentation({
    required Rect? bbox,
    required List<Offset> positivePoints,
    required List<Offset> negativePoints,
    required Float32List? lowResMask,
  }) async {
    return _controller.callSegmentation(
      bbox: bbox,
      positivePoints: positivePoints,
      negativePoints: negativePoints,
      lowResMask: lowResMask,
    );
  }

  Future<void> _applySegmentationResult(dynamic result) async {
    await _controller.applySegmentationResult(result);
  }

  Future<void> _refineSegmentation(Offset point) async {
    if (_session.image.file == null ||
        _session.mask.segmentationBytes == null ||
        _controller.isSegmentationInProgress) {
      return;
    }
    final isPositive =
        _session.interaction.pointMode == SegmentationPointMode.positive;
    AppLogger.log(
        'Refining segmentation with ${isPositive ? 'positive' : 'negative'} point: $point');

    final previousPositive =
        List<Offset>.from(_session.interaction.positivePoints);
    final previousNegative =
        List<Offset>.from(_session.interaction.negativePoints);
    final refinement = _controller.validateRefinement(
      lowResMask: _session.segmentation.lowResMaskInput,
      isPositive: isPositive,
      point: point,
      positivePoints: _session.interaction.positivePoints,
      negativePoints: _session.interaction.negativePoints,
    );

    if (refinement.message != null) {
      AppLogger.log(refinement.message!);
      if (!mounted) return;
      return;
    }
    final update = refinement.update!;

    if (mounted) {
      _controller.update(() {
        _controller.isSegmentationInProgress = true;
        _session.interaction.positivePoints
          ..clear()
          ..addAll(update.positivePoints);
        _session.interaction.negativePoints
          ..clear()
          ..addAll(update.negativePoints);
      });
      _updateMaskPulse();
    }

    try {
      final result = await _controller.refineSegmentation(
        segmentationImageRect: _session.segmentation.segmentationImageRect,
        positivePoints: update.positivePoints,
        negativePoints: update.negativePoints,
        lowResMask: _session.segmentation.lowResMaskInput!,
      );
      await _applySegmentationResult(result);
      if (!mounted) return;
      _controller.update(() {
        _session.interaction.lastTapImagePoint = null;
        _controller.isSegmentationInProgress = false;
      });
      _updateMaskPulse();
    } catch (error, stackTrace) {
      AppLogger.error(
        'Segmentation refinement failed',
        error,
        stackTrace,
      );
      if (!mounted) return;
      _controller.update(() {
        _session.interaction.positivePoints
          ..clear()
          ..addAll(previousPositive);
        _session.interaction.negativePoints
          ..clear()
          ..addAll(previousNegative);
        _controller.isSegmentationInProgress = false;
      });
      _updateMaskPulse();
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Refinement failed: $error')),
      );
    }
  }

  Offset? _globalToScene(Offset globalPosition) {
    final renderBox =
        _interactiveViewerKey.currentContext?.findRenderObject() as RenderBox?;
    final canvasSize = _session.canvas.size;
    if (renderBox == null || canvasSize == null) {
      return null;
    }
    final local = renderBox.globalToLocal(globalPosition);
    final scenePoint = _transformationController.toScene(local);
    return Offset(
      scenePoint.dx.clamp(0.0, canvasSize.width),
      scenePoint.dy.clamp(0.0, canvasSize.height),
    );
  }

  double get _currentViewerScale =>
      _transformationController.value.getMaxScaleOnAxis();

  double _currentBrushSceneWidth(double drawW) {
    final scale = _currentViewerScale;
    final width = _baseBrushSceneWidth / (scale <= 0 ? 1.0 : scale);
    return width.clamp(2.0, _baseBrushSceneWidth * 2);
  }

  void _addStrokePoint(
    Offset scenePoint,
    double drawW,
    double drawH,
    double brushSceneWidth,
  ) {
    if (_session.mask.image == null) return;
    final clamped = Offset(
      scenePoint.dx.clamp(0.0, drawW),
      scenePoint.dy.clamp(0.0, drawH),
    );
    final widthScale = _session.mask.image!.width / drawW;
    final heightScale = _session.mask.image!.height / drawH;
    final centerX = (clamped.dx * widthScale)
        .round()
        .clamp(0, _session.mask.image!.width - 1);
    final centerY = (clamped.dy * heightScale)
        .round()
        .clamp(0, _session.mask.image!.height - 1);

    final brushRadiusScene = math.max(1.0, brushSceneWidth / 2.0);
    final radiusX = math.max(1, (brushRadiusScene * widthScale).round());
    final radiusY = math.max(1, (brushRadiusScene * heightScale).round());

    for (int dy = -radiusY; dy <= radiusY; dy++) {
      for (int dx = -radiusX; dx <= radiusX; dx++) {
        final normX = dx / radiusX;
        final normY = dy / radiusY;
        if ((normX * normX + normY * normY) > 1.0) continue;
        final nx = (centerX + dx).clamp(0, _session.mask.image!.width - 1);
        final ny = (centerY + dy).clamp(0, _session.mask.image!.height - 1);
        _session.mask.image!.setPixelRgba(nx, ny, 0, 0, 0, 255);
      }
    }

    _session.interaction.points.add(clamped);
  }

  Future<void> _onSegmentPressed() async {
    if (_controller.isSegmentationInProgress) {
      return;
    }
    if (_session.image.file == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("No image selected.")),
      );
      return;
    }

    AppLogger.log(
      'Segment button pressed. interaction.mode=$_session.interaction.mode, lastPoint=$_session.interaction.lastTapImagePoint',
    );

    if (_session.interaction.mode == InteractionMode.point) {
      if (_session.interaction.lastTapImagePoint == null) {
        return;
      }
      final point = _session.interaction.lastTapImagePoint!;
      await _runSegmentationFromClick(point);
      return;
    }

    if (_session.interaction.points.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("First draw the mask")),
      );
      return;
    }
    final box = bboxFromPoints(_session.interaction.points);
    if (box.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Failed to determine bbox")),
      );
      return;
    }

    final canvasSize = _session.canvas.size;
    if (canvasSize == null ||
        canvasSize.width == 0 ||
        canvasSize.height == 0 ||
        _session.image.width == null ||
        _session.image.height == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Canvas size unavailable for bbox.")),
      );
      return;
    }

    if (box.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Failed to determine bbox")),
      );
      return;
    }

    final scaleX = _session.image.width! / canvasSize.width;
    final scaleY = _session.image.height! / canvasSize.height;
    final imageRect = Rect.fromLTRB(
      ((box.left.clamp(0.0, canvasSize.width)) * scaleX)
          .clamp(0.0, _session.image.width!.toDouble())
          .toDouble(),
      ((box.top.clamp(0.0, canvasSize.height)) * scaleY)
          .clamp(0.0, _session.image.height!.toDouble())
          .toDouble(),
      ((box.right.clamp(0.0, canvasSize.width)) * scaleX)
          .clamp(0.0, _session.image.width!.toDouble())
          .toDouble(),
      ((box.bottom.clamp(0.0, canvasSize.height)) * scaleY)
          .clamp(0.0, _session.image.height!.toDouble())
          .toDouble(),
    );

    await _runSegmentationFromBbox(imageRect);
  }

  Future<void> _runInpainting() async {
    if (_session.image.file == null || _session.mask.image == null) return;

    _controller.setInpaintingInProgress(true);
    _updateMaskPulse();

    try {
      final inpaintingStart = DateTime.now();
      FirebaseAnalytics.instance.logEvent(
        name: 'inpainting_started',
        parameters: {
          'width': _session.image.width!,
          'height': _session.image.height!,
          'model': _controller.inpaintingModel.modelName,
          'quantization': _controller.inpaintingModel.quantizationType,
          'environment': _controller.lastInpaintingExecutionProvider,
          'device': AppLogger.deviceInfo,
          'os_version': AppLogger.osVersion,
        },
      );

      final bytes = await _session.image.file!.readAsBytes();
      final originalImage = img.decodeImage(bytes)!;

      if (_session.mask.segmentationBytes != null) {
        _session.mask.image = img.decodeImage(_session.mask.segmentationBytes!)!;
      }

      final dilated = dilateMask(_session.mask.image!, radius: 20);

      final output = await _controller.runInpainting(
        original: originalImage,
        mask: dilated,
      );

      FirebaseAnalytics.instance.logEvent(
        name: 'inpainting_inference',
        parameters: {
          'model': _controller.inpaintingModel.modelName,
          'quantization': _controller.inpaintingModel.quantizationType,
          'inference_ms': output.inferenceDurationMs,
          'environment': _controller.lastInpaintingExecutionProvider,
          'device': AppLogger.deviceInfo,
          'os_version': AppLogger.osVersion,
        },
      );

      final inpaintingEnd = DateTime.now();
      final durationMs =
          inpaintingEnd.difference(inpaintingStart).inMilliseconds;

      final decoded = img.decodeImage(output.bytes)!;
      final newTemp =
          await ImageService.saveTempImage(output.bytes, 'input.png');

      FirebaseAnalytics.instance.logEvent(
        name: 'inpainting_completed',
        parameters: {
          'inpainting_duration_ms': durationMs,
          'inpainting_inference_ms': output.inferenceDurationMs,
          'model': _controller.inpaintingModel.modelName,
          'quantization': _controller.inpaintingModel.quantizationType,
          'environment': _controller.lastInpaintingExecutionProvider,
          'device': AppLogger.deviceInfo,
          'os_version': AppLogger.osVersion,
        },
      );

      _startNewEditingSession(resized: decoded, tempFile: newTemp);
    } finally {
      _controller.setInpaintingInProgress(false);
      _updateMaskPulse();
    }
  }

  void _handleTapDown(TapDownDetails details) {
    _tapDownGlobal = details.globalPosition;
    _tapDownTime = DateTime.now();
  }

  void _handleTapCancel() {
    _tapDownGlobal = null;
    _tapDownTime = null;
  }

  Future<void> _handleTapUp(
    TapUpDetails details,
    double drawW,
    double drawH,
  ) async {
    const tapMsThreshold = 180;
    const tapMoveThreshold = 6.0;
    final downPos = _tapDownGlobal;
    final downTime = _tapDownTime;
    _tapDownGlobal = null;
    _tapDownTime = null;
    if (downPos == null || downTime == null) return;
    final elapsedMs = DateTime.now().difference(downTime).inMilliseconds;
    final moveDistance = (details.globalPosition - downPos).distance;
    if (moveDistance > tapMoveThreshold) return;
    if (_session.image.width == null || _session.image.height == null) {
      return;
    }
    if (elapsedMs > tapMsThreshold) {
      AppLogger.log('Long press detected: ${elapsedMs}ms');
    }

    final scenePoint = _globalToScene(details.globalPosition);
    final size = _session.canvas.size;
    if (scenePoint == null || size == null) return;
    final px = (scenePoint.dx * (_session.image.width! / size.width))
        .clamp(0.0, _session.image.width!.toDouble());
    final py = (scenePoint.dy * (_session.image.height! / size.height))
        .clamp(0.0, _session.image.height!.toDouble());
    final imagePoint = Offset(px, py);

    AppLogger.log(
        'Tap on image (scene=$scenePoint → image=$imagePoint) drawSize=($drawW,$drawH)');

    if (_session.mask.segmentationBytes != null) {
      _refineSegmentation(imagePoint);
    } else {
    _controller.update(() {
      _session.interaction.mode = InteractionMode.point;
      _session.interaction.lastTapImagePoint = imagePoint;
    });
      if (!_controller.isSegmentationInProgress) {
        await _onSegmentPressed();
      }
    }
  }

  void _handlePanStart(
    DragStartDetails details,
    double drawW,
    double drawH,
    double brushSceneWidth,
  ) {
    final scenePoint = _globalToScene(details.globalPosition);
    if (scenePoint == null) return;
    _controller.update(() {
      _session.interaction.mode = InteractionMode.draw;
      _session.interaction.lastTapImagePoint = null;
      _addStrokePoint(
        scenePoint,
        drawW,
        drawH,
        brushSceneWidth,
      );
    });
  }

  void _handlePanUpdate(
    DragUpdateDetails details,
    double drawW,
    double drawH,
    double brushSceneWidth,
  ) {
    final scenePoint = _globalToScene(details.globalPosition);
    if (scenePoint == null) return;
    _controller.update(() {
      _addStrokePoint(
        scenePoint,
        drawW,
        drawH,
        brushSceneWidth,
      );
    });
  }

  Future<void> _handlePanEnd() async {
    _controller.update(() => _session.interaction.points.add(Offset.infinite));
    final hasStroke = _session.interaction.points.any((point) => point.isFinite);
    if (hasStroke && !_controller.isSegmentationInProgress) {
      await _onSegmentPressed();
    }
  }

  @override
  void dispose() {
    _maskPulseController.dispose();
    _transformationController.dispose();
    _controller.dispose();
    super.dispose();
  }

  Future<void> _saveCurrentImage() async {
    if (_session.image.file == null) return;
    final bytes = await _session.image.file!.readAsBytes();
    await _saveImageToGallery(bytes);
  }

  Future<void> _selectSegmentationPrecision() async {
    final precision = await showModalBottomSheet<SegmentationPrecision>(
      context: context,
      builder: (context) {
        return SafeArea(
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const ListTile(
                  title: Text('MobileSAM (full)'),
                ),
                ListTile(
                  leading: const Icon(Icons.memory),
                  title: const Text('MobileSAM FP32'),
                  subtitle: const Text('Highest precision, largest model'),
                  onTap: () => Navigator.pop(
                    context,
                    SegmentationPrecision.fp32,
                  ),
                ),
                ListTile(
                  leading: const Icon(Icons.blur_on),
                  title: const Text('MobileSAM FP16'),
                  subtitle: const Text('Half precision, balance speed/quality'),
                  onTap: () => Navigator.pop(
                    context,
                    SegmentationPrecision.fp16,
                  ),
                ),
                ListTile(
                  leading: const Icon(Icons.speed),
                  title: const Text('MobileSAM INT8 (dynamic quant)'),
                  subtitle: const Text('Smaller model, dynamic calibration'),
                  onTap: () => Navigator.pop(
                    context,
                    SegmentationPrecision.int8Dynamic,
                  ),
                ),
                ListTile(
                  leading: const Icon(Icons.flash_on),
                  title: const Text('MobileSAM INT8 (static quant)'),
                  subtitle: const Text('Static calibration, fastest option'),
                  onTap: () => Navigator.pop(
                    context,
                    SegmentationPrecision.int8Static,
                  ),
                ),
                const ListTile(
                  title: Text('MobileSAM (pruned encoder)'),
                ),
                ListTile(
                  leading: const Icon(Icons.crop),
                  title: const Text('MobileSAM FP32 pruned 12'),
                  subtitle: const Text('Smallest encoder (pruned 12%)'),
                  onTap: () => Navigator.pop(
                    context,
                    SegmentationPrecision.pruned012,
                  ),
                ),
                ListTile(
                  leading: const Icon(Icons.crop),
                  title: const Text('MobileSAM FP32 pruned 25'),
                  subtitle: const Text('Pruned encoder (25%)'),
                  onTap: () => Navigator.pop(
                    context,
                    SegmentationPrecision.pruned025,
                  ),
                ),
                ListTile(
                  leading: const Icon(Icons.crop),
                  title: const Text('MobileSAM FP32 pruned 40'),
                  subtitle: const Text('Pruned encoder (40%)'),
                  onTap: () => Navigator.pop(
                    context,
                    SegmentationPrecision.pruned040,
                  ),
                ),
                ListTile(
                  leading: const Icon(Icons.crop),
                  title: const Text('MobileSAM FP32 pruned 54'),
                  subtitle: const Text('Pruned encoder (54%)'),
                  onTap: () => Navigator.pop(
                    context,
                    SegmentationPrecision.pruned054,
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
    if (precision != null) {
      _controller.setSegmentationPrecision(precision);
      FirebaseAnalytics.instance.logEvent(
        name: 'segmentation_precision_selected',
        parameters: {
          'model': _controller.segmentationPrecision.modelName,
          'quantization': _controller.segmentationPrecision.quantizationType,
        },
      );
      AppLogger.log('Segmentation precision changed to $precision');
    }
  }

  Future<void> _selectExecutionProvider() async {
    final provider = await showModalBottomSheet<ExecutionProvider>(
      context: context,
      builder: (context) {
        return SafeArea(
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const ListTile(
                  title: Text('Execution environment'),
                ),
                ListTile(
                  leading: const Icon(Icons.auto_mode),
                  title: const Text('Auto (CoreML > CPU)'),
                  subtitle: const Text('Try CoreML first, fallback to CPU'),
                  onTap: () => Navigator.pop(
                    context,
                    ExecutionProvider.auto,
                  ),
                ),
                ListTile(
                  leading: const Icon(Icons.memory),
                  title: const Text('CPU only'),
                  subtitle: const Text('Disable hardware acceleration'),
                  onTap: () => Navigator.pop(
                    context,
                    ExecutionProvider.cpu,
                  ),
                ),
                ListTile(
                  leading: const Icon(Icons.developer_board),
                  title: const Text('CoreML only'),
                  subtitle:
                      const Text('Force CoreML (fallback if unsupported)'),
                  onTap: () => Navigator.pop(
                    context,
                    ExecutionProvider.coreml,
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );

    if (provider != null) {
      _controller.setExecutionProvider(provider);
      FirebaseAnalytics.instance.logEvent(
        name: 'execution_provider_selected',
        parameters: {
          'provider': _executionProviderValue,
        },
      );
      AppLogger.log('Execution provider changed to $_executionProviderValue');
    }
  }

  Future<void> _selectInpaintingModel() async {
    final model = await showModalBottomSheet<InpaintingModel>(
      context: context,
      builder: (context) {
        return SafeArea(
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                ListTile(
                  leading: const Icon(Icons.memory),
                  title: const Text('MI-GAN FP32'),
                  subtitle: const Text('Higher precision, larger model'),
                  onTap: () => Navigator.pop(context, InpaintingModel.fp32),
                ),
                ListTile(
                  leading: const Icon(Icons.blur_on),
                  title: const Text('MI-GAN FP16'),
                  subtitle: const Text(
                      'Half precision mixed model, balance speed/quality'),
                  onTap: () => Navigator.pop(context, InpaintingModel.fp16),
                ),
                ListTile(
                  leading: const Icon(Icons.speed),
                  title: const Text('MI-GAN INT8 (dynamic quant)'),
                  subtitle: const Text(
                      'Smaller quantized model, dynamic calibration (may be slower)'),
                  onTap: () =>
                      Navigator.pop(context, InpaintingModel.int8Dynamic),
                ),
                ListTile(
                  leading: const Icon(Icons.flash_on),
                  title: const Text('MI-GAN INT8 (static quant)'),
                  subtitle: const Text(
                      'Static calibration, fastest quantized option'),
                  onTap: () =>
                      Navigator.pop(context, InpaintingModel.int8Static),
                ),
              ],
            ),
          ),
        );
      },
    );
    if (model != null) {
      _controller.setInpaintingModel(model);
      FirebaseAnalytics.instance.logEvent(
        name: 'inpainting_model_selected',
        parameters: {
          'model': _controller.inpaintingModel.modelName,
          'quantization': _controller.inpaintingModel.quantizationType,
        },
      );
      AppLogger.log('Inpainting model changed to $model');
    }
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) {
        final colorScheme = Theme.of(context).colorScheme;
        final Widget canvasChild;
        if (_session.output.bytes != null) {
          canvasChild = Center(
            child: Image.memory(
              _session.output.bytes!,
              width: _session.image.width?.toDouble(),
              height: _session.image.height?.toDouble(),
              fit: BoxFit.contain,
            ),
          );
        } else if (_session.image.file == null) {
          canvasChild = const InpaintingEmptyState();
        } else {
          canvasChild = InpaintingImageStack(
            imageFile: _session.image.file,
            previewMaskBytes: _session.mask.previewBytes,
            imageWidth: _session.image.width,
            imageHeight: _session.image.height,
            points: _session.interaction.points,
            maskPulse: _maskPulse,
            imageKey: _imageKey,
            interactiveViewerKey: _interactiveViewerKey,
            transformationController: _transformationController,
            onCanvasSize: (size) => _session.canvas.size = size,
            brushWidthForDraw: _currentBrushSceneWidth,
            onTapDown: _handleTapDown,
            onTapCancel: _handleTapCancel,
            onTapUp: _handleTapUp,
            onPanStart: _handlePanStart,
            onPanUpdate: _handlePanUpdate,
            onPanEnd: _handlePanEnd,
          );
        }

        return Scaffold(
          body: Stack(
            children: [
              const InpaintingBackground(),
              Positioned(
                top: -60,
                right: -40,
                child: Container(
                  width: 180,
                  height: 180,
                  decoration: BoxDecoration(
                    color: colorScheme.primary.withOpacity(0.12),
                    shape: BoxShape.circle,
                  ),
                ),
              ),
              Positioned(
                bottom: 120,
                left: -30,
                child: Container(
                  width: 140,
                  height: 140,
                  decoration: BoxDecoration(
                    color: colorScheme.secondary.withOpacity(0.12),
                    shape: BoxShape.circle,
                  ),
                ),
              ),
              SafeArea(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    InpaintingHeader(
                      segmentationPrecisionLabel:
                          _controller.segmentationPrecision.label,
                      inpaintingModelLabel: _controller.inpaintingModel.label,
                      executionProviderLabel: _executionProviderLabel,
                      isSegmentationInProgress:
                          _controller.isSegmentationInProgress,
                    ),
                    Expanded(
                      child: Padding(
                        padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
                        child: Center(
                          child: AspectRatio(
                            aspectRatio: 1,
                            child: Container(
                              decoration: BoxDecoration(
                                color: colorScheme.surface.withOpacity(0.95),
                                borderRadius: BorderRadius.circular(28),
                                border: Border.all(
                                  color: colorScheme.outline.withOpacity(0.08),
                                ),
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.black.withOpacity(0.08),
                                    blurRadius: 24,
                                    offset: const Offset(0, 12),
                                  ),
                                ],
                              ),
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(24),
                                child: canvasChild,
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          bottomNavigationBar: InpaintingActionBar(
            onPick: _pickImage,
            onCamera: _pickImageFromCamera,
            onInpaint: _runInpainting,
            onSave: _session.image.file == null ? null : _saveCurrentImage,
            onSelectSegmentationModel: _selectSegmentationPrecision,
            onSelectExecutionProvider: _selectExecutionProvider,
            onSelectInpaintingModel: _selectInpaintingModel,
            showHintControls:
                _session.interaction.mode == InteractionMode.point &&
                    _session.mask.segmentationBytes != null,
            isPositiveSelected:
                _session.interaction.pointMode == SegmentationPointMode.positive,
            isNegativeSelected:
                _session.interaction.pointMode == SegmentationPointMode.negative,
            isSegmentationInProgress: _controller.isSegmentationInProgress,
            onSelectPositive: () {
              _controller.update(() => _session.interaction.pointMode =
                  SegmentationPointMode.positive);
            },
            onSelectNegative: () {
              _controller.update(() => _session.interaction.pointMode =
                  SegmentationPointMode.negative);
            },
            showClear: _session.interaction.mode == InteractionMode.draw &&
                _hasManualDrawing,
            onClear: _clearManualMask,
          ),
        );
      },
    );
  }
}
