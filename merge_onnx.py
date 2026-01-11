import onnx
from onnx import external_data_helper

src = "assets/encoder_best_pruned_054.onnx"
dst = "assets/encoder_best_pruned_054_single_ir9.onnx"

model = onnx.load(src)
external_data_helper.convert_model_from_external_data(model)

model.ir_version = 9

onnx.save(model, dst)
print("Saved:", dst, "IR:", model.ir_version)
