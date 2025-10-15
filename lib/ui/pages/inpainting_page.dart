import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import '../widgets/mask_painter.dart';
import '../../services/image_service.dart';
import '../../services/inpainting_service.dart';
import '../../services/segmentation_service.dart';
import '../../utils/app_logger.dart';
import '../../utils/image_utils.dart';
import 'package:image_gallery_saver/image_gallery_saver.dart';
import 'package:firebase_analytics/firebase_analytics.dart';

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
  InteractionMode _mode = InteractionMode.point;
  Offset? _lastTapImagePoint;
  Rect? _bbox;
  Size? _canvasSize;

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
    final blank =
        img.Image(width: resized.width, height: resized.height, numChannels: 1)
          ..getBytes().fillRange(0, resized.width * resized.height, 255);

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
    });
  }

  Future<void> _runSegmentationFromClick(Offset point) async {
    if (_imageFile == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text(".")),
      );
      return;
    }

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text("Starting segmentation from point...")),
    );
    AppLogger.log('Segmentation from point requested: $point');

    try {
      final encoderData = await rootBundle.load('assets/encoder.onnx');
      final decoderData = await rootBundle.load('assets/decoder.onnx');

      final segmentationStart = DateTime.now();
      FirebaseAnalytics.instance.logEvent(
        name: 'segmentation_started',
        parameters: {
          'x': point.dx,
          'y': point.dy,
        },
      );

      final mask = await SegmentationService.segmentFromPoint(
        imageFile: _imageFile!,
        clickPoint: point,
        encoderData: encoderData,
        decoderData: decoderData,
      );

      final segmentationEnd = DateTime.now();
      final durationMs =
          segmentationEnd.difference(segmentationStart).inMilliseconds;

      FirebaseAnalytics.instance.logEvent(
        name: 'segmentation_completed',
        parameters: {
          'segmentation_duration_ms': durationMs,
          'model': 'mobileSAM'
        },
      );
      AppLogger.log(
        'Segmentation from point completed in ${durationMs}ms',
      );

      final imageBytes = await _imageFile!.readAsBytes();
      final baseImage = img.decodeImage(imageBytes)!;
      final decodedMask = img.decodeImage(mask)!;

      final overlay = img.Image.from(baseImage);
      for (int y = 0; y < overlay.height; y++) {
        for (int x = 0; x < overlay.width; x++) {
          final v = decodedMask.getPixel(x, y).r;
          if (v == 0) {
            overlay.setPixelRgba(x, y, 255, 0, 0, 100);
          }
        }
      }

      if (!mounted) return;
      setState(() {
        _segmentationMask = mask;
        _maskImage = decodedMask;
        _points.clear();
        _previewMaskBytes = Uint8List.fromList(img.encodePng(overlay));
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
    }
  }

  Future<void> _runSegmentationFromBbox(Rect bbox) async {
    if (_imageFile == null) return;
    AppLogger.log('Segmentation from bbox requested: $bbox');
    try {
      final encoderData = await rootBundle.load('assets/encoder.onnx');
      final decoderData = await rootBundle.load('assets/decoder.onnx');

      final segmentationStart = DateTime.now();
      FirebaseAnalytics.instance.logEvent(
        name: 'segmentation_started',
        parameters: {
          'x1': bbox.left,
          'y1': bbox.top,
          'x2': bbox.right,
          'y2': bbox.bottom,
        },
      );

      final mask = await SegmentationService.segmentFromBbox(
        imageFile: _imageFile!,
        bboxPx: bbox,
        encoderData: encoderData,
        decoderData: decoderData,
      );

      final segmentationEnd = DateTime.now();
      final durationMs =
          segmentationEnd.difference(segmentationStart).inMilliseconds;

      FirebaseAnalytics.instance.logEvent(
        name: 'segmentation_completed',
        parameters: {
          'segmentation_duration_ms': durationMs,
          'model': 'mobileSAM'
        },
      );
      AppLogger.log(
        'Segmentation from bbox completed in ${durationMs}ms',
      );

      final imageBytes = await _imageFile!.readAsBytes();
      final baseImage = img.decodeImage(imageBytes)!;
      final decodedMask = img.decodeImage(mask)!;

      final overlay = img.Image.from(baseImage);
      for (int y = 0; y < overlay.height; y++) {
        for (int x = 0; x < overlay.width; x++) {
          final v = decodedMask.getPixel(x, y).r;
          if (v == 0) {
            overlay.setPixelRgba(x, y, 255, 0, 0, 100);
          }
        }
      }

      if (!mounted) return;
      setState(() {
        _segmentationMask = mask;
        _maskImage = decodedMask;
        _points.clear();
        _previewMaskBytes = Uint8List.fromList(img.encodePng(overlay));
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
    }
  }

  Future<void> _onSegmentPressed() async {
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
      },
    );

    final bytes = await _imageFile!.readAsBytes();
    final originalImage = img.decodeImage(bytes)!;

    if (_segmentationMask != null) {
      _maskImage = img.decodeImage(_segmentationMask!)!;
    }

    final modelData = await rootBundle.load('assets/migan.onnx');

    final dilated = dilateMask(_maskImage!, radius: 20);

    final output = await InpaintingService.runInpainting(
      original: originalImage,
      mask: dilated,
      modelData: modelData,
    );

    final inpaintingEnd = DateTime.now();
    final durationMs = inpaintingEnd.difference(inpaintingStart).inMilliseconds;

    final decoded = img.decodeImage(output)!;
    final newTemp = await ImageService.saveTempImage(output, 'input.png');

    FirebaseAnalytics.instance.logEvent(
      name: 'inpainting_completed',
      parameters: {
        'inpainting_duration_ms': durationMs,
        'model': 'migan',
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
            child: Stack(
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
                GestureDetector(
                  behavior: HitTestBehavior.deferToChild,
                  onTapDown: (details) {
                    if (_mode != InteractionMode.point) return;

                    final box = _imageKey.currentContext?.findRenderObject()
                        as RenderBox?;
                    if (box == null ||
                        _imageWidth == null ||
                        _imageHeight == null) return;

                    final local = details.localPosition;

                    final px = (local.dx * (_imageWidth! / drawW))
                        .clamp(0.0, _imageWidth!.toDouble());
                    final py = (local.dy * (_imageHeight! / drawH))
                        .clamp(0.0, _imageHeight!.toDouble());
                    final imagePoint = Offset(px, py);

                    AppLogger.log(
                        'Tap on image (local=$local → image=$imagePoint) drawSize=($drawW,$drawH)');

                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text(
                            "Clicked on image at point: ${imagePoint.dx.toStringAsFixed(1)}, ${imagePoint.dy.toStringAsFixed(1)}"),
                      ),
                    );

                    setState(() {
                      _lastTapImagePoint = imagePoint;
                      _bbox = null;
                    });
                  },
                  onPanUpdate: (details) {
                    if (_mode == InteractionMode.draw && _maskImage != null) {
                      final local = details.localPosition;
                      final clamped = Offset(
                        local.dx.clamp(0.0, drawW).toDouble(),
                        local.dy.clamp(0.0, drawH).toDouble(),
                      );
                      final x = (clamped.dx * (_maskImage!.width / drawW))
                          .toInt()
                          .clamp(0, _maskImage!.width - 1);
                      final y = (clamped.dy * (_maskImage!.height / drawH))
                          .toInt()
                          .clamp(0, _maskImage!.height - 1);
                      _maskImage!.setPixelRgba(x, y, 0, 0, 0, 255);
                      setState(() {
                        _points.add(clamped);
                        final newBox = bboxFromPoints(_points);
                        _bbox = newBox.isEmpty ? null : newBox;
                      });
                    }
                  },
                  onPanEnd: (_) {
                    if (_mode == InteractionMode.draw) {
                      setState(() => _points.add(Offset.infinite));
                    }
                  },
                  child: CustomPaint(
                    painter:
                        _segmentationMask == null ? MaskPainter(_points) : null,
                    size: Size(drawW, drawH),
                  ),
                ),
                if (_bbox != null)
                  Positioned.fill(
                    child: CustomPaint(
                      painter: BBoxPainter(_bbox!),
                    ),
                  ),
                if (_lastTapImagePoint != null)
                  Positioned.fill(
                    child: CustomPaint(
                      painter: SquarePointPainter(
                        point: _lastTapImagePoint!,
                        imageSize: Size(
                            _imageWidth!.toDouble(), _imageHeight!.toDouble()),
                        size: 16.0,
                        color: Colors.blueAccent,
                      ),
                    ),
                  )
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildFloatingButtons() {
    return Row(
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
          onPressed: _onSegmentPressed,
          heroTag: 'segment',
          child: const Icon(Icons.crop_square),
        ),
      ],
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
