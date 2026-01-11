import argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort


def run_model(model_path: str, batch: int, h: int, w: int, reps: int) -> None:
    print(f"\n=== {model_path} ===")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1

    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    print("Inputs:")
    for i in inputs:
        print(f"  - name={i.name}, type={i.type}, shape={i.shape}")

    print("Outputs:")
    for o in outputs:
        print(f"  - name={o.name}, type={o.type}, shape={o.shape}")

    inp_name = "input_image"
    model_input_names = [i.name for i in inputs]
    if inp_name not in model_input_names:
        if len(inputs) == 1:
            inp_name = inputs[0].name
            print(f"Using input name from model: {inp_name}")
        else:
            raise ValueError(f"Model expects inputs {model_input_names}, but 'input_image' not found.")

    x = np.random.randn(batch, 3, h, w).astype(np.float32)

    sess.run(None, {inp_name: x})

    for _ in range(reps - 1):
        sess.run(None, {inp_name: x})

    y = sess.run(None, {inp_name: x})
    print("Run OK. Output shapes:")
    for idx, arr in enumerate(y):
        if isinstance(arr, np.ndarray):
            print(f"  - out[{idx}] shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"  - out[{idx}] type={type(arr)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="Paths to .onnx models")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--h", type=int, default=1024)
    p.add_argument("--w", type=int, default=1024)
    p.add_argument("--reps", type=int, default=3, help="How many runs per model (warmup-ish)")
    args = p.parse_args()

    for mp in args.models:
        path = Path(mp)
        if not path.exists():
            print(f"Missing: {mp}")
            continue
        run_model(str(path), args.batch, args.h, args.w, args.reps)


if __name__ == "__main__":
    main()
