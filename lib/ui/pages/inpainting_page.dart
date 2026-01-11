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
      if (result['isSuccess'] == true) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Image saved to gallery')),
        );
      } else {
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
    setState(() {
      _controller.resetSession(
        tempFile: tempFile,
        resized: resized,
        blankMask: blank,
      );
      _controller.isSegmentationInProgress = false;
      _controller.isInpaintingInProgress = false;
    });
    _updateMaskPulse();
  }

  Future<void> _runSegmentationFromClick(Offset point) async {
    if (_session.imageFile == null || _controller.isSegmentationInProgress) {
      if (_session.imageFile == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text(".")),
        );
      }
      return;
    }

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text("Starting segmentation from point...")),
    );
    AppLogger.log('Segmentation from point requested: $point');

    _controller.isSegmentationInProgress = true;
    _updateMaskPulse();
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
      setState(() {
        _session.positivePoints
          ..clear()
          ..addAll(newPositive);
        _session.negativePoints.clear();
        _session.segmentationImageRect = null;
        _session.points.clear();
        _session.lastTapImagePoint = null;
        _session.pointMode = SegmentationPointMode.positive;
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
      if (mounted) setState(() => _controller.isSegmentationInProgress = false);
      _updateMaskPulse();
    }
  }

  img.Image _createBlankMaskImage(int width, int height) {
    final blank = img.Image(width: width, height: height, numChannels: 1);
    blank.getBytes().fillRange(0, width * height, 255);
    return blank;
  }

  void _clearManualMask() {
    if (_session.imageWidth == null || _session.imageHeight == null) return;
    setState(() {
      _session.points.clear();
      if (_session.segmentationMask != null) {
        _session.maskImage = img.decodeImage(_session.segmentationMask!)!;
      } else {
        _session.maskImage =
            _createBlankMaskImage(_session.imageWidth!, _session.imageHeight!);
      }
    });
  }

  Future<void> _runSegmentationFromBbox(Rect bbox) async {
    if (_session.imageFile == null || _controller.isSegmentationInProgress) return;
    AppLogger.log('Segmentation from bbox requested: $bbox');

    _controller.isSegmentationInProgress = true;
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
      setState(() {
        _session.positivePoints.clear();
        _session.negativePoints.clear();
        _session.segmentationImageRect = bbox;
        _session.points.clear();
        _session.pointMode = SegmentationPointMode.positive;
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
      if (mounted) setState(() => _controller.isSegmentationInProgress = false);
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
    if (!mounted) return;
    setState(() {});
  }

  Future<void> _refineSegmentation(Offset point) async {
    if (_session.imageFile == null ||
        _session.segmentationMask == null ||
        _controller.isSegmentationInProgress) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
              'Refinement skipped because segmentation mask is unavailable.'),
        ),
      );
      return;
    }
    final isPositive = _session.pointMode == SegmentationPointMode.positive;
    AppLogger.log(
        'Refining segmentation with ${isPositive ? 'positive' : 'negative'} point: $point');

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
            'Refining segmentation with ${isPositive ? 'positive' : 'negative'} point: $point'),
      ),
    );

    final previousPositive = List<Offset>.from(_session.positivePoints);
    final previousNegative = List<Offset>.from(_session.negativePoints);
    final refinement = _controller.validateRefinement(
      lowResMask: _session.lowResMaskInput,
      isPositive: isPositive,
      point: point,
      positivePoints: _session.positivePoints,
      negativePoints: _session.negativePoints,
    );

    if (refinement.message != null) {
      AppLogger.log(refinement.message!);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(refinement.message!),
        ),
      );
      return;
    }
    final update = refinement.update!;

    if (mounted) {
      setState(() {
        _controller.isSegmentationInProgress = true;
        _session.positivePoints
          ..clear()
          ..addAll(update.positivePoints);
        _session.negativePoints
          ..clear()
          ..addAll(update.negativePoints);
      });
      _updateMaskPulse();
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Added ${isPositive ? 'positive' : 'negative'} point at '
            '(${point.dx.toStringAsFixed(1)}, ${point.dy.toStringAsFixed(1)})',
          ),
        ),
      );
    }

    try {
      final result = await _controller.refineSegmentation(
        segmentationImageRect: _session.segmentationImageRect,
        positivePoints: update.positivePoints,
        negativePoints: update.negativePoints,
        lowResMask: _session.lowResMaskInput!,
      );
      await _applySegmentationResult(result);
      if (!mounted) return;
      setState(() {
        _session.lastTapImagePoint = null;
        _controller.isSegmentationInProgress = false;
      });
      _updateMaskPulse();
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            isPositive
                ? 'Positive point applied to mask'
                : 'Negative point applied to mask',
          ),
        ),
      );
    } catch (error, stackTrace) {
      AppLogger.error(
        'Segmentation refinement failed',
        error,
        stackTrace,
      );
      if (!mounted) return;
      setState(() {
        _session.positivePoints
          ..clear()
          ..addAll(previousPositive);
        _session.negativePoints
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
    final canvasSize = _session.canvasSize;
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
    if (_session.maskImage == null) return;
    final clamped = Offset(
      scenePoint.dx.clamp(0.0, drawW),
      scenePoint.dy.clamp(0.0, drawH),
    );
    final widthScale = _session.maskImage!.width / drawW;
    final heightScale = _session.maskImage!.height / drawH;
    final centerX = (clamped.dx * widthScale)
        .round()
        .clamp(0, _session.maskImage!.width - 1);
    final centerY = (clamped.dy * heightScale)
        .round()
        .clamp(0, _session.maskImage!.height - 1);

    final brushRadiusScene = math.max(1.0, brushSceneWidth / 2.0);
    final radiusX = math.max(1, (brushRadiusScene * widthScale).round());
    final radiusY = math.max(1, (brushRadiusScene * heightScale).round());

    for (int dy = -radiusY; dy <= radiusY; dy++) {
      for (int dx = -radiusX; dx <= radiusX; dx++) {
        final normX = dx / radiusX;
        final normY = dy / radiusY;
        if ((normX * normX + normY * normY) > 1.0) continue;
        final nx = (centerX + dx).clamp(0, _session.maskImage!.width - 1);
        final ny = (centerY + dy).clamp(0, _session.maskImage!.height - 1);
        _session.maskImage!.setPixelRgba(nx, ny, 0, 0, 0, 255);
      }
    }

    _session.points.add(clamped);
  }

  Future<void> _onSegmentPressed() async {
    if (_controller.isSegmentationInProgress) {
      return;
    }
    if (_session.imageFile == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("No image selected.")),
      );
      return;
    }

    AppLogger.log(
      'Segment button pressed. mode=$_session.mode, lastPoint=$_session.lastTapImagePoint',
    );

    if (_session.mode == InteractionMode.point) {
      if (_session.lastTapImagePoint == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
              content: Text(
                  "Click on the object in the image to start segmentation")),
        );
        return;
      }
      final point = _session.lastTapImagePoint!;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Segmentation point: $point")),
      );
      await _runSegmentationFromClick(point);
      return;
    }

    if (_session.points.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("First draw the mask")),
      );
      return;
    }
    final box = bboxFromPoints(_session.points);
    if (box.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Failed to determine bbox")),
      );
      return;
    }

    final canvasSize = _session.canvasSize;
    if (canvasSize == null ||
        canvasSize.width == 0 ||
        canvasSize.height == 0 ||
        _session.imageWidth == null ||
        _session.imageHeight == null) {
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

    final scaleX = _session.imageWidth! / canvasSize.width;
    final scaleY = _session.imageHeight! / canvasSize.height;
    final imageRect = Rect.fromLTRB(
      ((box.left.clamp(0.0, canvasSize.width)) * scaleX)
          .clamp(0.0, _session.imageWidth!.toDouble())
          .toDouble(),
      ((box.top.clamp(0.0, canvasSize.height)) * scaleY)
          .clamp(0.0, _session.imageHeight!.toDouble())
          .toDouble(),
      ((box.right.clamp(0.0, canvasSize.width)) * scaleX)
          .clamp(0.0, _session.imageWidth!.toDouble())
          .toDouble(),
      ((box.bottom.clamp(0.0, canvasSize.height)) * scaleY)
          .clamp(0.0, _session.imageHeight!.toDouble())
          .toDouble(),
    );

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text("Segmentation from bbox: $box")),
    );
    await _runSegmentationFromBbox(imageRect);
  }

  Future<void> _runInpainting() async {
    if (_session.imageFile == null || _session.maskImage == null) return;

    _controller.isInpaintingInProgress = true;
    _updateMaskPulse();

    try {
      final inpaintingStart = DateTime.now();
      FirebaseAnalytics.instance.logEvent(
        name: 'inpainting_started',
        parameters: {
          'width': _session.imageWidth!,
          'height': _session.imageHeight!,
          'model': _controller.inpaintingModel.modelName,
          'quantization': _controller.inpaintingModel.quantizationType,
        'environment': _controller.lastInpaintingExecutionProvider,
          'device': AppLogger.deviceInfo,
          'os_version': AppLogger.osVersion,
        },
      );

      final bytes = await _session.imageFile!.readAsBytes();
      final originalImage = img.decodeImage(bytes)!;

      if (_session.segmentationMask != null) {
        _session.maskImage = img.decodeImage(_session.segmentationMask!)!;
      }

      final dilated = dilateMask(_session.maskImage!, radius: 20);

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
      _controller.isInpaintingInProgress = false;
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
    if (_session.imageWidth == null || _session.imageHeight == null) {
      return;
    }
    if (elapsedMs > tapMsThreshold) {
      AppLogger.log('Long press detected: ${elapsedMs}ms');
    }

    final scenePoint = _globalToScene(details.globalPosition);
    final size = _session.canvasSize;
    if (scenePoint == null || size == null) return;
    final px = (scenePoint.dx * (_session.imageWidth! / size.width))
        .clamp(0.0, _session.imageWidth!.toDouble());
    final py = (scenePoint.dy * (_session.imageHeight! / size.height))
        .clamp(0.0, _session.imageHeight!.toDouble());
    final imagePoint = Offset(px, py);

    AppLogger.log(
        'Tap on image (scene=$scenePoint → image=$imagePoint) drawSize=($drawW,$drawH)');

    if (_session.segmentationMask != null) {
      _refineSegmentation(imagePoint);
    } else {
      setState(() {
        _session.mode = InteractionMode.point;
        _session.lastTapImagePoint = imagePoint;
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
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text("Clicked on image (draw)."),
      ),
    );
    final scenePoint = _globalToScene(details.globalPosition);
    if (scenePoint == null) return;
    setState(() {
      _session.mode = InteractionMode.draw;
      _session.lastTapImagePoint = null;
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
    setState(() {
      _addStrokePoint(
        scenePoint,
        drawW,
        drawH,
        brushSceneWidth,
      );
    });
  }

  Future<void> _handlePanEnd() async {
    setState(() => _session.points.add(Offset.infinite));
    final hasStroke = _session.points.any((point) => point.isFinite);
    if (hasStroke && !_controller.isSegmentationInProgress) {
      await _onSegmentPressed();
    }
  }

  @override
  void dispose() {
    _maskPulseController.dispose();
    _transformationController.dispose();
    super.dispose();
  }

  Future<void> _saveCurrentImage() async {
    if (_session.imageFile == null) return;
    final bytes = await _session.imageFile!.readAsBytes();
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
      setState(() {
        _controller.segmentationPrecision = precision;
      });
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
      setState(() {
        _controller.setExecutionProvider(provider);
      });
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
      setState(() {
        _controller.inpaintingModel = model;
      });
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
    final colorScheme = Theme.of(context).colorScheme;
    final Widget canvasChild;
    if (_session.outputBytes != null) {
      canvasChild = Center(
        child: Image.memory(
          _session.outputBytes!,
          width: _session.imageWidth?.toDouble(),
          height: _session.imageHeight?.toDouble(),
          fit: BoxFit.contain,
        ),
      );
    } else if (_session.imageFile == null) {
      canvasChild = const InpaintingEmptyState();
    } else {
      canvasChild = InpaintingImageStack(
        imageFile: _session.imageFile,
        previewMaskBytes: _session.previewMaskBytes,
        imageWidth: _session.imageWidth,
        imageHeight: _session.imageHeight,
        points: _session.points,
        maskPulse: _maskPulse,
        imageKey: _imageKey,
        interactiveViewerKey: _interactiveViewerKey,
        transformationController: _transformationController,
        onCanvasSize: (size) => _session.canvasSize = size,
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
                  segmentationPrecisionLabel: _controller.segmentationPrecision.label,
                  inpaintingModelLabel: _controller.inpaintingModel.label,
                  executionProviderLabel: _executionProviderLabel,
                  isSegmentationInProgress: _controller.isSegmentationInProgress,
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
        onSave: _session.imageFile == null ? null : _saveCurrentImage,
        onSelectSegmentationModel: _selectSegmentationPrecision,
        onSelectExecutionProvider: _selectExecutionProvider,
        onSelectInpaintingModel: _selectInpaintingModel,
        showHintControls: _session.mode == InteractionMode.point &&
            _session.segmentationMask != null,
        isPositiveSelected:
            _session.pointMode == SegmentationPointMode.positive,
        isNegativeSelected:
            _session.pointMode == SegmentationPointMode.negative,
        isSegmentationInProgress: _controller.isSegmentationInProgress,
        onSelectPositive: () {
          setState(() => _session.pointMode = SegmentationPointMode.positive);
        },
        onSelectNegative: () {
          setState(() => _session.pointMode = SegmentationPointMode.negative);
        },
        showClear: _session.mode == InteractionMode.draw && _hasManualDrawing,
        onClear: _clearManualMask,
      ),
    );
  }
}
