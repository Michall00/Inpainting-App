import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

const _pixelMean = [123.675, 116.28, 103.53];
const _pixelStd = [58.395, 57.12, 57.375];

OrtValueTensor convertImageToUint8NCHW(img.Image image) {
  final w = image.width, h = image.height;
  final pixels = image.getBytes(order: img.ChannelOrder.rgb);
  final uint8s = <int>[];
  for (int c = 0; c < 3; c++) {
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final i = (y * w + x) * 3 + c;
        uint8s.add(pixels[i]);
      }
    }
  }
  return OrtValueTensor.createTensorWithDataList(
      Uint8List.fromList(uint8s), [1, 3, h, w]);
}

OrtValueTensor convertImageToFloatNCHW(img.Image image) {
  final w = image.width, h = image.height;
  final pixels = image.getBytes(order: img.ChannelOrder.rgb);
  final floats = Float32List(3 * h * w);
  for (int c = 0; c < 3; c++) {
    final channelOffset = c * h * w;
    final mean = _pixelMean[c];
    final std = _pixelStd[c];
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final srcIndex = (y * w + x) * 3 + c;
        final dstIndex = channelOffset + y * w + x;
        final pixel = pixels[srcIndex].toDouble();
        floats[dstIndex] = (pixel - mean) / std;
      }
    }
  }
  return OrtValueTensor.createTensorWithDataList(
    floats,
    [1, 3, h, w],
  );
}

OrtValueTensor convertMaskToUint8NCHW(img.Image mask) {
  final w = mask.width, h = mask.height;
  final pixels = mask.getBytes();
  final uint8s = <int>[];
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final i = y * w + x;
      uint8s.add(pixels[i]);
    }
  }
  return OrtValueTensor.createTensorWithDataList(
      Uint8List.fromList(uint8s), [1, 1, h, w]);
}

img.Image convertNCHWtoImage(List data) {
  final channels = data[0];
  final h = channels[0].length;
  final w = channels[0][0].length;
  final image = img.Image(width: w, height: h);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      final r = channels[0][y][x];
      final g = channels[1][y][x];
      final b = channels[2][y][x];
      image.setPixelRgb(x, y, r, g, b);
    }
  }
  return image;
}
