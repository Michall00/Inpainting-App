import onnx

def print_opsets(path: str) -> None:
    print("Opset for", path)
    m = onnx.load(path)
    for o in m.opset_import:
        domain = o.domain if o.domain else "ai.onnx"
        print(domain, o.version)


print_opsets("assets/coordfill.onnx")
print_opsets("assets/migan.onnx")
print_opsets("assets/migan_int8_quant.onnx")
print_opsets("assets/migan_mixed_fp16.onnx")
# print_opsets("assets/coordfill.onnx")
