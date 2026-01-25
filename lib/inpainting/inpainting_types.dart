enum InteractionMode { draw, point }

enum SegmentationPointMode { positive, negative }

enum SegmentationPrecision {
  fp32,
  fp16,
  int8Dynamic,
  pruned012,
  pruned025,
  pruned040,
  pruned054,
}

enum InpaintingModel { fp32, fp16, int8Dynamic }
