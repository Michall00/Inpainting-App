import 'inpainting_types.dart';

extension SegmentationPrecisionInfo on SegmentationPrecision {
  String get modelName {
    switch (this) {
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

  String get quantizationType {
    switch (this) {
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

  String get encoderAsset {
    switch (this) {
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

  String get decoderAsset {
    switch (this) {
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

  String get label {
    switch (this) {
      case SegmentationPrecision.fp32:
        return 'SAM FP32';
      case SegmentationPrecision.fp16:
        return 'SAM FP16';
      case SegmentationPrecision.int8Dynamic:
        return 'SAM INT8 Dyn';
      case SegmentationPrecision.int8Static:
        return 'SAM INT8 Static';
      case SegmentationPrecision.pruned012:
        return 'SAM Pruned 12';
      case SegmentationPrecision.pruned025:
        return 'SAM Pruned 25';
      case SegmentationPrecision.pruned040:
        return 'SAM Pruned 40';
      case SegmentationPrecision.pruned054:
        return 'SAM Pruned 54';
    }
  }
}

extension InpaintingModelInfo on InpaintingModel {
  String get asset {
    switch (this) {
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

  String get modelName {
    switch (this) {
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

  String get quantizationType {
    switch (this) {
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

  String get label {
    switch (this) {
      case InpaintingModel.fp32:
        return 'MI-GAN FP32';
      case InpaintingModel.fp16:
        return 'MI-GAN FP16';
      case InpaintingModel.int8Dynamic:
        return 'MI-GAN INT8 Dyn';
      case InpaintingModel.int8Static:
        return 'MI-GAN INT8 Static';
    }
  }
}
