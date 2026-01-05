import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import '../widgets/mask_painter.dart';
import '../../services/image_service.dart';
import '../../services/inpainting_service.dart';
import '../../services/segmentation_service.dart';
import '../../services/execution_provider.dart';
import '../../utils/app_logger.dart';
import '../../utils/image_utils.dart';
import 'package:image_gallery_saver/image_gallery_saver.dart';
import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:vector_math/vector_math_64.dart' show Matrix4;

class InpaintingPage extends StatefulWidget {
  const InpaintingPage({super.key});

  @override
  State<InpaintingPage> createState() => _InpaintingPageState();
}

class _InpaintingPageState extends State<InpaintingPage> {
  File? _imageFile;
  img.Image? _maskImage;
  int? _imageWidth;
  int? _imageHeight;
  Uint8List? _previewMaskBytes;
  Uint8List? _segmentationMask;
  Uint8List? _outputBytes;
  final List<Offset> _points = [];
  final GlobalKey _imageKey = GlobalKey();
  final GlobalKey _interactiveViewerKey = GlobalKey();
  final TransformationController _transformationController =
      TransformationController();
  InteractionMode _mode = InteractionMode.point;
  Offset? _lastTapImagePoint;
  Rect? _bbox;
  Size? _canvasSize;
  Float32List? _lowResMaskInput;
  Rect? _segmentationImageRect;
  final List<Offset> _positivePoints = [];
  final List<Offset> _negativePoints = [];
  SegmentationPointMode _pointMode = SegmentationPointMode.positive;
  bool _isSegmentationInProgress = false;
  SegmentationPrecision _segmentationPrecision = SegmentationPrecision.fp32;
  InpaintingModel _inpaintingModel = InpaintingModel.fp32;
  ExecutionProvider _executionProvider = ExecutionProvider.auto;

  static const double _baseBrushSceneWidth = 20.0;
  bool get _hasManualDrawing => _points.any((offset) => offset.isFinite);

  String get _segmentationModelName {
    switch (_segmentationPrecision) {
      case SegmentationPrecision.fp32:
        return 'mobileSAM_fp32';
      case SegmentationPrecision.fp16:
        return 'mobileSAM_fp16';
      case SegmentationPrecision.int8Dynamic:
        return 'mobileSAM_int8_dynamic';
      case SegmentationPrecision.int8Static:
        return 'mobileSAM_int8_static';
      case SegmentationPrecision.pruned012:
        return 'mobileSAM_pruned_012';
      case SegmentationPrecision.pruned025:
        return 'mobileSAM_pruned_025';
      case SegmentationPrecision.pruned040:
        return 'mobileSAM_pruned_040';
      case SegmentationPrecision.pruned054:
        return 'mobileSAM_pruned_054';
    }
  }

  String get _segmentationQuantizationType {
    switch (_segmentationPrecision) {
      case SegmentationPrecision.fp32:
        return 'fp32';
      case SegmentationPrecision.fp16:
        return 'fp16';
      case SegmentationPrecision.int8Dynamic:
        return 'int8_dynamic';
      case SegmentationPrecision.int8Static:
        return 'int8_static';
      case SegmentationPrecision.pruned012:
        return 'pruned_012';
      case SegmentationPrecision.pruned025:
        return 'pruned_025';
      case SegmentationPrecision.pruned040:
        return 'pruned_040';
      case SegmentationPrecision.pruned054:
        return 'pruned_054';
    }
  }

  String get _segmentationEncoderAsset {
    switch (_segmentationPrecision) {
      case SegmentationPrecision.fp32:
        return 'assets/encoder_best.onnx';
      case SegmentationPrecision.fp16:
        return 'assets/encoder_best_fp16.onnx';
      case SegmentationPrecision.int8Dynamic:
        return 'assets/encoder_best_int8_dynamic.onnx';
      case SegmentationPrecision.int8Static:
        return 'assets/encoder_best_int8_static.onnx';
      case SegmentationPrecision.pruned012:
        return 'assets/encoder_best_pruned_012.onnx';
      case SegmentationPrecision.pruned025:
        return 'assets/encoder_best_pruned_025.onnx';
      case SegmentationPrecision.pruned040:
        return 'assets/encoder_best_pruned_040.onnx';
      case SegmentationPrecision.pruned054:
        return 'assets/encoder_best_pruned_054.onnx';
    }
  }

  String get _segmentationDecoderAsset {
    switch (_segmentationPrecision) {
      case SegmentationPrecision.fp32:
        return 'assets/decoder_best.onnx';
      case SegmentationPrecision.fp16:
        return 'assets/decoder_best_fp16.onnx';
      case SegmentationPrecision.int8Dynamic:
        return 'assets/decoder_best_int8_dynamic.onnx';
      case SegmentationPrecision.int8Static:
        return 'assets/decoder_best_int8_static.onnx';
      case SegmentationPrecision.pruned012:
      case SegmentationPrecision.pruned025:
      case SegmentationPrecision.pruned040:
      case SegmentationPrecision.pruned054:
        return 'assets/decoder_best.onnx';
    }
  }

  String get _inpaintingModelAsset {
    switch (_inpaintingModel) {
      case InpaintingModel.fp32:
        return 'assets/migan.onnx';
      case InpaintingModel.fp16:
        return 'assets/migan_mixed_fp16.onnx';
      case InpaintingModel.int8Dynamic:
        return 'assets/migan_int8_quant.onnx';
      case InpaintingModel.int8Static:
        return 'assets/migan_int8_quant_static.onnx';
    }
  }

  String get _inpaintingModelName {
    switch (_inpaintingModel) {
      case InpaintingModel.fp32:
        return 'migan_fp32';
      case InpaintingModel.fp16:
        return 'migan_fp16';
      case InpaintingModel.int8Dynamic:
        return 'migan_int8_dynamic';
      case InpaintingModel.int8Static:
        return 'migan_int8_static';
    }
  }

  String get _inpaintingQuantizationType {
    switch (_inpaintingModel) {
      case InpaintingModel.fp32:
        return 'fp32';
      case InpaintingModel.fp16:
        return 'fp16';
      case InpaintingModel.int8Dynamic:
        return 'int8_dynamic';
      case InpaintingModel.int8Static:
        return 'int8_static';
    }
  }

  String get _executionProviderLabel {
    return executionProviderLabel(_executionProvider);
  }

  String get _executionProviderValue {
    return executionProviderValue(_executionProvider);
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
      _imageFile = tempFile;
      _imageWidth = resized.width;
      _imageHeight = resized.height;

      _outputBytes = null;
      _previewMaskBytes = null;
      _segmentationMask = null;
      _maskImage = blank;
      _points.clear();
      _lastTapImagePoint = null;
      _bbox = null;
      _canvasSize = null;
      _mode = InteractionMode.point;
      _lowResMaskInput = null;
      _segmentationImageRect = null;
      _positivePoints.clear();
      _negativePoints.clear();
      _pointMode = SegmentationPointMode.positive;
      _isSegmentationInProgress = false;
    });
  }

  Future<void> _runSegmentationFromClick(Offset point) async {
    if (_imageFile == null || _isSegmentationInProgress) {
      if (_imageFile == null) {
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

    _isSegmentationInProgress = true;
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
          'model': _segmentationModelName,
          'quantization': _segmentationQuantizationType,
          'environment': SegmentationService.lastExecutionProvider,
          'device': AppLogger.deviceInfo,
          'os_version': AppLogger.osVersion,
        },
      );
      AppLogger.log(
        'Segmentation from bbox completed in ${durationMs}ms '
        'encoder=${result.encoderInferenceMs}ms '
        'decoder=${result.decoderInferenceMs}ms '
        'model=$_segmentationModelName '
        'quantization=$_segmentationQuantizationType '
        'env=${SegmentationService.lastExecutionProvider}',
      );

      await _applySegmentationResult(result);
      if (!mounted) return;
      setState(() {
        _positivePoints
          ..clear()
          ..addAll(newPositive);
        _negativePoints.clear();
        _segmentationImageRect = null;
        _points.clear();
        _lastTapImagePoint = null;
        _bbox = null;
        _pointMode = SegmentationPointMode.positive;
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
      if (mounted) setState(() => _isSegmentationInProgress = false);
    }
  }

  img.Image _createBlankMaskImage(int width, int height) {
    final blank = img.Image(width: width, height: height, numChannels: 1);
    blank.getBytes().fillRange(0, width * height, 255);
    return blank;
  }

  void _clearManualMask() {
    if (_imageWidth == null || _imageHeight == null) return;
    setState(() {
      _points.clear();
      _bbox = null;
      if (_segmentationMask != null) {
        _maskImage = img.decodeImage(_segmentationMask!)!;
      } else {
        _maskImage = _createBlankMaskImage(_imageWidth!, _imageHeight!);
      }
    });
  }

  Future<void> _runSegmentationFromBbox(Rect bbox) async {
    if (_imageFile == null || _isSegmentationInProgress) return;
    AppLogger.log('Segmentation from bbox requested: $bbox');

    _isSegmentationInProgress = true;
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
          'model': _segmentationModelName,
          'quantization': _segmentationQuantizationType,
          'environment': SegmentationService.lastExecutionProvider,
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
        _positivePoints.clear();
        _negativePoints.clear();
        _segmentationImageRect = bbox;
        _points.clear();
        _bbox = null;
        _pointMode = SegmentationPointMode.positive;
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
      if (mounted) setState(() => _isSegmentationInProgress = false);
    }
  }

  Future<SegmentationResult> _callSegmentation({
    required Rect? bbox,
    required List<Offset> positivePoints,
    required List<Offset> negativePoints,
    required Float32List? lowResMask,
  }) async {
    final encoderData = await rootBundle.load(_segmentationEncoderAsset);
    final decoderData = await rootBundle.load(_segmentationDecoderAsset);

    return SegmentationService.segmentWithPoints(
      imageFile: _imageFile!,
      bboxPx: bbox,
      positivePoints: List<Offset>.from(positivePoints),
      negativePoints: List<Offset>.from(negativePoints),
      lowResMaskInput: lowResMask,
      encoderData: encoderData,
      decoderData: decoderData,
    );
  }

  Future<_SegmentationVisuals> _prepareSegmentationVisuals(
      SegmentationResult result) async {
    final imageBytes = await _imageFile!.readAsBytes();
    final baseImage = img.decodeImage(imageBytes)!;
    final decodedMask = img.decodeImage(result.maskBytes)!;
    final overlay = _composeOverlay(baseImage, decodedMask);
    return _SegmentationVisuals(
      maskBytes: result.maskBytes,
      maskImage: decodedMask,
      overlayBytes: Uint8List.fromList(img.encodePng(overlay)),
      lowResMask: result.lowResMask,
    );
  }

  Future<void> _applySegmentationResult(SegmentationResult result) async {
    final visuals = await _prepareSegmentationVisuals(result);
    if (!mounted) return;
    setState(() {
      _segmentationMask = visuals.maskBytes;
      _maskImage = visuals.maskImage;
      _previewMaskBytes = visuals.overlayBytes;
      _lowResMaskInput = visuals.lowResMask;
    });
  }

  img.Image _composeOverlay(img.Image baseImage, img.Image mask) {
    final overlay = img.Image.from(baseImage);
    for (int y = 0; y < overlay.height; y++) {
      for (int x = 0; x < overlay.width; x++) {
        final value = mask.getPixel(x, y).r;
        if (value == 0) {
          overlay.setPixelRgba(x, y, 0, 255, 0, 120);
        }
      }
    }
    return overlay;
  }

  Future<void> _refineSegmentation(Offset point) async {
    if (_imageFile == null ||
        _segmentationMask == null ||
        _isSegmentationInProgress) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
              'Refinement skipped because segmentation mask is unavailable.'),
        ),
      );
      return;
    }
    final lowRes = _lowResMaskInput;
    if (lowRes == null) {
      AppLogger.log(
          'Refinement skipped because low-res mask is unavailable for point $point');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
              'Refinement skipped because low-res mask is unavailable for point $point'),
        ),
      );
      return;
    }

    final isPositive = _pointMode == SegmentationPointMode.positive;
    AppLogger.log(
        'Refining segmentation with ${isPositive ? 'positive' : 'negative'} point: $point');

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
            'Refining segmentation with ${isPositive ? 'positive' : 'negative'} point: $point'),
      ),
    );

    final previousPositive = List<Offset>.from(_positivePoints);
    final previousNegative = List<Offset>.from(_negativePoints);
    final updatedPositive = List<Offset>.from(_positivePoints);
    final updatedNegative = List<Offset>.from(_negativePoints);

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

    if (!added) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'This ${isPositive ? 'positive' : 'negative'} point is already added.',
          ),
        ),
      );
      return;
    }

    if (mounted) {
      setState(() {
        _isSegmentationInProgress = true;
        _positivePoints
          ..clear()
          ..addAll(updatedPositive);
        _negativePoints
          ..clear()
          ..addAll(updatedNegative);
      });
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
      final result = await _callSegmentation(
        bbox: _segmentationImageRect,
        positivePoints: updatedPositive,
        negativePoints: updatedNegative,
        lowResMask: lowRes,
      );
      await _applySegmentationResult(result);
      if (!mounted) return;
      setState(() {
        _lastTapImagePoint = null;
        _isSegmentationInProgress = false;
      });
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
        _positivePoints
          ..clear()
          ..addAll(previousPositive);
        _negativePoints
          ..clear()
          ..addAll(previousNegative);
        _isSegmentationInProgress = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Refinement failed: $error')),
      );
    }
  }

  Offset? _globalToScene(Offset globalPosition) {
    final renderBox =
        _interactiveViewerKey.currentContext?.findRenderObject() as RenderBox?;
    final canvasSize = _canvasSize;
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
    if (_maskImage == null) return;
    final clamped = Offset(
      scenePoint.dx.clamp(0.0, drawW),
      scenePoint.dy.clamp(0.0, drawH),
    );
    final widthScale = _maskImage!.width / drawW;
    final heightScale = _maskImage!.height / drawH;
    final centerX =
        (clamped.dx * widthScale).round().clamp(0, _maskImage!.width - 1);
    final centerY =
        (clamped.dy * heightScale).round().clamp(0, _maskImage!.height - 1);

    final brushRadiusScene = math.max(1.0, brushSceneWidth / 2.0);
    final radiusX = math.max(1, (brushRadiusScene * widthScale).round());
    final radiusY = math.max(1, (brushRadiusScene * heightScale).round());

    for (int dy = -radiusY; dy <= radiusY; dy++) {
      for (int dx = -radiusX; dx <= radiusX; dx++) {
        final normX = dx / radiusX;
        final normY = dy / radiusY;
        if ((normX * normX + normY * normY) > 1.0) continue;
        final nx = (centerX + dx).clamp(0, _maskImage!.width - 1);
        final ny = (centerY + dy).clamp(0, _maskImage!.height - 1);
        _maskImage!.setPixelRgba(nx, ny, 0, 0, 0, 255);
      }
    }

    _points.add(clamped);
    final newBox = bboxFromPoints(_points);
    _bbox = newBox.isEmpty ? null : newBox;
  }

  Future<void> _onSegmentPressed() async {
    if (_isSegmentationInProgress) {
      return;
    }
    if (_imageFile == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("No image selected.")),
      );
      return;
    }

    AppLogger.log(
      'Segment button pressed. mode=$_mode, lastPoint=$_lastTapImagePoint, bbox=$_bbox',
    );

    if (_mode == InteractionMode.point) {
      if (_lastTapImagePoint == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
              content: Text(
                  "Click on the object in the image to start segmentation")),
        );
        return;
      }
      final point = _lastTapImagePoint!;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Segmentation point: $point")),
      );
      await _runSegmentationFromClick(point);
      return;
    }

    Rect? box = _bbox;
    if (box == null) {
      if (_points.isEmpty) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("First draw the mask")),
        );
        return;
      }
      box = bboxFromPoints(_points);
      if (box.isEmpty) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Failed to determine bbox")),
        );
        return;
      }
    }

    final canvasSize = _canvasSize;
    if (canvasSize == null ||
        canvasSize.width == 0 ||
        canvasSize.height == 0 ||
        _imageWidth == null ||
        _imageHeight == null) {
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

    final scaleX = _imageWidth! / canvasSize.width;
    final scaleY = _imageHeight! / canvasSize.height;
    final imageRect = Rect.fromLTRB(
      ((box.left.clamp(0.0, canvasSize.width)) * scaleX)
          .clamp(0.0, _imageWidth!.toDouble())
          .toDouble(),
      ((box.top.clamp(0.0, canvasSize.height)) * scaleY)
          .clamp(0.0, _imageHeight!.toDouble())
          .toDouble(),
      ((box.right.clamp(0.0, canvasSize.width)) * scaleX)
          .clamp(0.0, _imageWidth!.toDouble())
          .toDouble(),
      ((box.bottom.clamp(0.0, canvasSize.height)) * scaleY)
          .clamp(0.0, _imageHeight!.toDouble())
          .toDouble(),
    );

    setState(() {
      _bbox = box;
    });

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text("Segmentation from bbox: $box")),
    );
    await _runSegmentationFromBbox(imageRect);
  }

  Future<void> _runInpainting() async {
    if (_imageFile == null || _maskImage == null) return;

    final inpaintingStart = DateTime.now();
    FirebaseAnalytics.instance.logEvent(
      name: 'inpainting_started',
      parameters: {
        'width': _imageWidth!,
        'height': _imageHeight!,
        'model': _inpaintingModelName,
        'quantization': _inpaintingQuantizationType,
        'environment': InpaintingService.lastExecutionProvider,
        'device': AppLogger.deviceInfo,
        'os_version': AppLogger.osVersion,
      },
    );

    final bytes = await _imageFile!.readAsBytes();
    final originalImage = img.decodeImage(bytes)!;

    if (_segmentationMask != null) {
      _maskImage = img.decodeImage(_segmentationMask!)!;
    }

    final modelData = await rootBundle.load(_inpaintingModelAsset);

    final dilated = dilateMask(_maskImage!, radius: 20);

    final output = await InpaintingService.runInpainting(
      original: originalImage,
      mask: dilated,
      modelData: modelData,
    );

    FirebaseAnalytics.instance.logEvent(
      name: 'inpainting_inference',
      parameters: {
        'model': _inpaintingModelName,
        'quantization': _inpaintingQuantizationType,
        'inference_ms': output.inferenceDurationMs,
        'environment': InpaintingService.lastExecutionProvider,
        'device': AppLogger.deviceInfo,
        'os_version': AppLogger.osVersion,
      },
    );

    final inpaintingEnd = DateTime.now();
    final durationMs = inpaintingEnd.difference(inpaintingStart).inMilliseconds;

    final decoded = img.decodeImage(output.bytes)!;
    final newTemp = await ImageService.saveTempImage(output.bytes, 'input.png');

    FirebaseAnalytics.instance.logEvent(
      name: 'inpainting_completed',
      parameters: {
        'inpainting_duration_ms': durationMs,
        'inpainting_inference_ms': output.inferenceDurationMs,
        'model': _inpaintingModelName,
        'quantization': _inpaintingQuantizationType,
        'environment': InpaintingService.lastExecutionProvider,
        'device': AppLogger.deviceInfo,
        'os_version': AppLogger.osVersion,
      },
    );

    _startNewEditingSession(resized: decoded, tempFile: newTemp);
  }

  Widget _buildImageStack() {
    if (_imageFile == null && _previewMaskBytes == null) {
      return const Center(child: Text("No image selected"));
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        final maxW = constraints.maxWidth;
        final maxH = constraints.maxHeight;

        final imgW = _imageWidth?.toDouble() ?? 256;
        final imgH = _imageHeight?.toDouble() ?? 256;

        final scale = (maxW / imgW < maxH / imgH) ? maxW / imgW : maxH / imgH;
        final drawW = imgW * scale;
        final drawH = imgH * scale;
        _canvasSize = Size(drawW, drawH);

        return Center(
          child: SizedBox(
            width: drawW,
            height: drawH,
            child: InteractiveViewer(
              key: _interactiveViewerKey,
              transformationController: _transformationController,
              minScale: 1.0,
              maxScale: 5.0,
              panEnabled: false,
              clipBehavior: Clip.none,
              child: ValueListenableBuilder<Matrix4>(
                valueListenable: _transformationController,
                builder: (context, value, _) {
                  final brushSceneWidth = _currentBrushSceneWidth(drawW);
                  return Stack(
                    children: [
                      Positioned.fill(
                        child: Image(
                          key: _imageKey,
                          image: _previewMaskBytes != null
                              ? MemoryImage(_previewMaskBytes!)
                              : FileImage(_imageFile!) as ImageProvider,
                          fit: BoxFit.contain,
                        ),
                      ),
                      Positioned.fill(
                        child: GestureDetector(
                          behavior: HitTestBehavior.opaque,
                          onTapDown: _mode == InteractionMode.point
                              ? (details) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                      content:
                                          Text("Clicked on image (point)."),
                                    ),
                                  );
                                  if (_imageWidth == null ||
                                      _imageHeight == null) return;
                                  final scenePoint =
                                      _globalToScene(details.globalPosition);
                                  final size = _canvasSize;
                                  if (scenePoint == null || size == null)
                                    return;
                                  final px = (scenePoint.dx *
                                          (_imageWidth! / size.width))
                                      .clamp(0.0, _imageWidth!.toDouble());
                                  final py = (scenePoint.dy *
                                          (_imageHeight! / size.height))
                                      .clamp(0.0, _imageHeight!.toDouble());
                                  final imagePoint = Offset(px, py);

                                  AppLogger.log(
                                      'Tap on image (scene=$scenePoint → image=$imagePoint) drawSize=($drawW,$drawH)');

                                  ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(
                                      content: Text(
                                          "Clicked on image at point: ${imagePoint.dx.toStringAsFixed(1)}, ${imagePoint.dy.toStringAsFixed(1)}"),
                                    ),
                                  );

                                  if (_segmentationMask != null) {
                                    _refineSegmentation(imagePoint);
                                  } else {
                                    setState(() {
                                      _lastTapImagePoint = imagePoint;
                                      _bbox = null;
                                    });
                                  }
                                }
                              : null,
                          onPanStart: _mode == InteractionMode.draw
                              ? (details) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                      content: Text("Clicked on image (draw)."),
                                    ),
                                  );
                                  final scenePoint =
                                      _globalToScene(details.globalPosition);
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
                              : null,
                          onPanUpdate: _mode == InteractionMode.draw
                              ? (details) {
                                  final scenePoint =
                                      _globalToScene(details.globalPosition);
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
                              : null,
                          onPanEnd: _mode == InteractionMode.draw
                              ? (_) {
                                  setState(() => _points.add(Offset.infinite));
                                }
                              : null,
                          child: CustomPaint(
                            painter: MaskPainter(
                              _points,
                              strokeWidth: brushSceneWidth,
                            ),
                            size: Size(drawW, drawH),
                          ),
                        ),
                      ),
                      if (_bbox != null)
                        Positioned.fill(
                          child: IgnorePointer(
                            ignoring: true,
                            child: CustomPaint(painter: BBoxPainter(_bbox!)),
                          ),
                        ),
                      if ((_positivePoints.isNotEmpty ||
                              _negativePoints.isNotEmpty) &&
                          _imageWidth != null &&
                          _imageHeight != null)
                        Positioned.fill(
                          child: IgnorePointer(
                            ignoring: true,
                            child: CustomPaint(
                              painter: SegmentationHintsPainter(
                                positives: _positivePoints,
                                negatives: _negativePoints,
                                imageSize: Size(_imageWidth!.toDouble(),
                                    _imageHeight!.toDouble()),
                              ),
                            ),
                          ),
                        ),
                      if (_lastTapImagePoint != null)
                        Positioned.fill(
                          child: IgnorePointer(
                            ignoring: true,
                            child: CustomPaint(
                              painter: SquarePointPainter(
                                point: _lastTapImagePoint!,
                                imageSize: Size(_imageWidth!.toDouble(),
                                    _imageHeight!.toDouble()),
                                size: 16.0,
                                color: Colors.blueAccent,
                              ),
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

  @override
  void dispose() {
    _transformationController.dispose();
    super.dispose();
  }

  Widget _buildFloatingButtons() {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.symmetric(horizontal: 12),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          FloatingActionButton(
            onPressed: _pickImage,
            heroTag: 'pick',
            child: const Icon(Icons.photo_library),
          ),
          const SizedBox(width: 12),
          FloatingActionButton(
            onPressed: _pickImageFromCamera,
            heroTag: 'camera',
            tooltip: 'Take a photo',
            child: const Icon(Icons.camera_alt),
          ),
          const SizedBox(width: 12),
          FloatingActionButton(
            onPressed: _runInpainting,
            heroTag: 'inpaint',
            child: const Icon(Icons.auto_fix_high),
          ),
          const SizedBox(width: 12),
          FloatingActionButton(
            onPressed: _imageFile == null
                ? null
                : () async {
                    final bytes = await _imageFile!.readAsBytes();
                    await _saveImageToGallery(bytes);
                  },
            heroTag: 'save',
            tooltip: 'Save to gallery',
            child: const Icon(Icons.save_alt),
          ),
          const SizedBox(width: 12),
          FloatingActionButton(
            onPressed: () {
              setState(() {
                _mode = _mode == InteractionMode.draw
                    ? InteractionMode.point
                    : InteractionMode.draw;
                if (_mode == InteractionMode.draw) {
                  _lastTapImagePoint = null;
                }
              });
            },
            heroTag: 'mode',
            child: Icon(
                _mode == InteractionMode.draw ? Icons.brush : Icons.touch_app),
          ),
          const SizedBox(width: 12),
          FloatingActionButton(
            onPressed: _isSegmentationInProgress ? null : _onSegmentPressed,
            heroTag: 'segment',
            child: const Icon(Icons.crop_square),
          ),
          const SizedBox(width: 12),
          FloatingActionButton(
            onPressed: () async {
              final precision =
                  await showModalBottomSheet<SegmentationPrecision>(
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
                            subtitle: const Text(
                                'Highest precision, largest model'),
                            onTap: () => Navigator.pop(
                              context,
                              SegmentationPrecision.fp32,
                            ),
                          ),
                          ListTile(
                            leading: const Icon(Icons.blur_on),
                            title: const Text('MobileSAM FP16'),
                            subtitle: const Text(
                                'Half precision, balance speed/quality'),
                            onTap: () => Navigator.pop(
                              context,
                              SegmentationPrecision.fp16,
                            ),
                          ),
                          ListTile(
                            leading: const Icon(Icons.speed),
                            title:
                                const Text('MobileSAM INT8 (dynamic quant)'),
                            subtitle: const Text(
                                'Smaller model, dynamic calibration'),
                            onTap: () => Navigator.pop(
                              context,
                              SegmentationPrecision.int8Dynamic,
                            ),
                          ),
                          ListTile(
                            leading: const Icon(Icons.flash_on),
                            title:
                                const Text('MobileSAM INT8 (static quant)'),
                            subtitle: const Text(
                                'Static calibration, fastest option'),
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
                            subtitle:
                                const Text('Smallest encoder (pruned 12%)'),
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
                  _segmentationPrecision = precision;
                });
                FirebaseAnalytics.instance.logEvent(
                  name: 'segmentation_precision_selected',
                  parameters: {
                    'model': _segmentationModelName,
                    'quantization': _segmentationQuantizationType,
                  },
                );
                AppLogger.log(
                  'Segmentation precision changed to $precision',
                );
              }
            },
            heroTag: 'backend',
            tooltip: 'Choose MobileSAM model',
            child: const Icon(Icons.tune),
          ),
          const SizedBox(width: 12),
          FloatingActionButton(
            onPressed: () async {
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
                            subtitle: const Text(
                                'Try CoreML first, fallback to CPU'),
                            onTap: () => Navigator.pop(
                              context,
                              ExecutionProvider.auto,
                            ),
                          ),
                          ListTile(
                            leading: const Icon(Icons.memory),
                            title: const Text('CPU only'),
                            subtitle:
                                const Text('Disable hardware acceleration'),
                            onTap: () => Navigator.pop(
                              context,
                              ExecutionProvider.cpu,
                            ),
                          ),
                          ListTile(
                            leading: const Icon(Icons.developer_board),
                            title: const Text('CoreML only'),
                            subtitle: const Text(
                                'Force CoreML (fallback if unsupported)'),
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
                  _executionProvider = provider;
                });
                SegmentationService.preferredExecutionProvider = provider;
                InpaintingService.preferredExecutionProvider = provider;
                FirebaseAnalytics.instance.logEvent(
                  name: 'execution_provider_selected',
                  parameters: {
                    'provider': _executionProviderValue,
                  },
                );
                AppLogger.log(
                    'Execution provider changed to $_executionProviderValue');
              }
            },
            heroTag: 'executionProvider',
            tooltip: 'Execution environment: $_executionProviderLabel',
            child: const Icon(Icons.computer),
          ),
          const SizedBox(width: 12),
          FloatingActionButton(
            onPressed: () async {
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
                            subtitle:
                                const Text('Higher precision, larger model'),
                            onTap: () =>
                                Navigator.pop(context, InpaintingModel.fp32),
                          ),
                          ListTile(
                            leading: const Icon(Icons.blur_on),
                            title: const Text('MI-GAN FP16'),
                            subtitle: const Text(
                                'Half precision mixed model, balance speed/quality'),
                            onTap: () =>
                                Navigator.pop(context, InpaintingModel.fp16),
                          ),
                          ListTile(
                            leading: const Icon(Icons.speed),
                            title: const Text('MI-GAN INT8 (dynamic quant)'),
                            subtitle: const Text(
                                'Smaller quantized model, dynamic calibration (may be slower)'),
                            onTap: () => Navigator.pop(
                                context, InpaintingModel.int8Dynamic),
                          ),
                          ListTile(
                            leading: const Icon(Icons.flash_on),
                            title: const Text('MI-GAN INT8 (static quant)'),
                            subtitle: const Text(
                                'Static calibration, fastest quantized option'),
                            onTap: () => Navigator.pop(
                                context, InpaintingModel.int8Static),
                          ),
                        ],
                      ),
                    ),
                  );
                },
              );
              if (model != null) {
                setState(() {
                  _inpaintingModel = model;
                });
                FirebaseAnalytics.instance.logEvent(
                  name: 'inpainting_model_selected',
                  parameters: {
                    'model': _inpaintingModelName,
                    'quantization': _inpaintingQuantizationType,
                  },
                );
                AppLogger.log('Inpainting model changed to $model');
              }
            },
            heroTag: 'inpaintingModel',
            tooltip: 'Choose MI-GAN model',
            child: const Icon(Icons.swap_vert),
          ),
          if (_mode == InteractionMode.point && _segmentationMask != null) ...[
            const SizedBox(width: 12),
            FloatingActionButton.small(
              onPressed: _isSegmentationInProgress
                  ? null
                  : () {
                      setState(
                        () => _pointMode = SegmentationPointMode.positive,
                      );
                    },
              heroTag: 'positiveHint',
              backgroundColor: _pointMode == SegmentationPointMode.positive
                  ? Colors.green
                  : null,
              child: const Icon(Icons.add_circle),
            ),
            const SizedBox(width: 12),
            FloatingActionButton.small(
              onPressed: _isSegmentationInProgress
                  ? null
                  : () {
                      setState(
                        () => _pointMode = SegmentationPointMode.negative,
                      );
                    },
              heroTag: 'negativeHint',
              backgroundColor: _pointMode == SegmentationPointMode.negative
                  ? Colors.red
                  : null,
              child: const Icon(Icons.remove_circle),
            ),
          ],
          if (_mode == InteractionMode.draw && _hasManualDrawing)
            const SizedBox(width: 12),
          if (_mode == InteractionMode.draw && _hasManualDrawing)
            FloatingActionButton(
              onPressed: _clearManualMask,
              heroTag: 'clear',
              tooltip: 'Clear drawing',
              child: const Icon(Icons.clear),
            ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Inpainting")),
      body: Builder(
        builder: (_) {
          if (_outputBytes != null) {
            return Center(
              child: Image.memory(
                _outputBytes!,
                width: _imageWidth?.toDouble(),
                height: _imageHeight?.toDouble(),
                fit: BoxFit.contain,
              ),
            );
          }
          if (_imageFile == null) {
            return const Center(child: Text("No image selected"));
          }
          return _buildImageStack();
        },
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      floatingActionButton: _buildFloatingButtons(),
    );
  }
}

enum InteractionMode { draw, point }

enum SegmentationPointMode { positive, negative }

enum SegmentationPrecision {
  fp32,
  fp16,
  int8Dynamic,
  int8Static,
  pruned012,
  pruned025,
  pruned040,
  pruned054,
}

enum InpaintingModel { fp32, fp16, int8Dynamic, int8Static }

class _SegmentationVisuals {
  final Uint8List maskBytes;
  final img.Image maskImage;
  final Uint8List overlayBytes;
  final Float32List lowResMask;

  const _SegmentationVisuals({
    required this.maskBytes,
    required this.maskImage,
    required this.overlayBytes,
    required this.lowResMask,
  });
}
