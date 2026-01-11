import onnx
from onnx import version_converter

m = onnx.load("assets/coordfill.onnx")
m = version_converter.convert_version(m, 12)
onnx.save(m, "assets/coordfill_opset12.onnx")
