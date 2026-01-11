import onnx

print(onnx.load("assets/encoder_best_pruned_012_single_ir9.onnx").ir_version)
