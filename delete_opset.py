import onnx

src = "assets/migan_int8_quant_static.onnx"
dst = "assets/migan_int8_quant_static_no_onnx_ml.onnx"

m = onnx.load(src)

kept = [op for op in m.opset_import if op.domain != "ai.onnx.ml"]

m.ClearField("opset_import")
m.opset_import.extend(kept)

onnx.save(m, dst)
print("Saved:", dst)
print("Opsets:", [(op.domain, op.version) for op in m.opset_import])
