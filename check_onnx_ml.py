import onnx

m = onnx.load("assets/migan_int8_quant_static_no_onnx_ml.onnx")
domains = sorted({n.domain for n in m.graph.node})
print("Node domains:", domains)

ml_nodes = [n for n in m.graph.node if n.domain == "ai.onnx.ml"]
print("ai.onnx.ml nodes:", len(ml_nodes))
if ml_nodes:
    print("Example:", ml_nodes[0].op_type)
