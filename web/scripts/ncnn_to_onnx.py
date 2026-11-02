"""Convert the ncnn realesr-animevideov3 (SRVGGNetCompact) model to ONNX.

Reads the .param/.bin pair already used by the CLI, so no PyTorch install is
needed. Only the layer types that SRVGGNetCompact uses are supported.

Usage: python ncnn_to_onnx.py MODEL.param MODEL.bin OUT.onnx [OUT-fp16.onnx]
Deps: numpy, onnx (+ onnxconverter-common for the fp16 copy).
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FP16_TAG = 0x01306B47
FP32_TAG = 0


class BinReader:
    def __init__(self, path):
        self.data = open(path, "rb").read()
        self.pos = 0

    def raw_f32(self, n):
        arr = np.frombuffer(self.data, np.float32, n, self.pos).copy()
        self.pos += 4 * n
        return arr

    def tagged(self, n):
        # ncnn prefixes conv weights with a 4-byte storage tag.
        tag = int(np.frombuffer(self.data, np.uint32, 1, self.pos)[0])
        self.pos += 4
        if tag == FP16_TAG:
            arr = np.frombuffer(self.data, np.float16, n, self.pos).astype(np.float32)
            self.pos += (2 * n + 3) & ~3
            return arr
        if tag == FP32_TAG:
            return self.raw_f32(n)
        raise ValueError(f"unsupported weight tag {tag:#x}")


def parse_param(path):
    lines = open(path).read().split("\n")
    assert lines[0].strip() == "7767517", "not an ncnn param file"
    layers = []
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split()
        ltype, name, n_in, n_out = parts[0], parts[1], int(parts[2]), int(parts[3])
        ins = parts[4 : 4 + n_in]
        outs = parts[4 + n_in : 4 + n_in + n_out]
        params = {}
        for kv in parts[4 + n_in + n_out :]:
            k, v = kv.split("=")
            params[int(k)] = v
        layers.append((ltype, name, ins, outs, params))
    return layers


def convert(param_path, bin_path, out_path, fp16_path=None):
    reader = BinReader(bin_path)
    nodes, inits = [], []
    alias = {}  # ncnn Split outputs -> original blob
    channels = {}
    graph_in = None

    def blob(b):
        return alias.get(b, b)

    for ltype, name, ins, outs, p in parse_param(param_path):
        ins = [blob(b) for b in ins]
        if ltype == "Input":
            graph_in = outs[0]
            channels[graph_in] = 3
        elif ltype == "Split":
            for o in outs:
                alias[o] = ins[0]
        elif ltype == "Convolution":
            c_out, k = int(p[0]), int(p.get(1, 1))
            w_size = int(p[6])
            c_in = w_size // (c_out * k * k)
            assert int(p.get(5, 0)) == 1, "conv without bias not supported"
            w = reader.tagged(w_size).reshape(c_out, c_in, k, k)
            b = reader.raw_f32(c_out)
            pad = int(p.get(4, 0))
            inits += [
                numpy_helper.from_array(w, f"{name}.w"),
                numpy_helper.from_array(b, f"{name}.b"),
            ]
            nodes.append(
                helper.make_node(
                    "Conv",
                    [ins[0], f"{name}.w", f"{name}.b"],
                    outs,
                    name=name,
                    kernel_shape=[k, k],
                    pads=[pad] * 4,
                )
            )
        elif ltype == "PReLU":
            n = int(p.get(0, 1))
            # ONNX PReLU broadcasts slope from the right, so shape it [C,1,1].
            slope = reader.raw_f32(n).reshape(n, 1, 1)
            inits.append(numpy_helper.from_array(slope, f"{name}.slope"))
            nodes.append(helper.make_node("PRelu", [ins[0], f"{name}.slope"], outs, name=name))
        elif ltype == "PixelShuffle":
            # ncnn mode 0 == torch.pixel_shuffle == ONNX DepthToSpace CRD.
            assert int(p.get(1, 0)) == 0
            nodes.append(
                helper.make_node(
                    "DepthToSpace", ins, outs, name=name, blocksize=int(p[0]), mode="CRD"
                )
            )
        elif ltype == "Interp":
            assert int(p[0]) == 1, "only nearest interp supported"
            scales = np.array([1, 1, float(p[1]), float(p[2])], np.float32)
            inits.append(numpy_helper.from_array(scales, f"{name}.scales"))
            nodes.append(
                helper.make_node(
                    "Resize",
                    [ins[0], "", f"{name}.scales"],
                    outs,
                    name=name,
                    mode="nearest",
                    coordinate_transformation_mode="asymmetric",
                    nearest_mode="floor",
                )
            )
        elif ltype == "BinaryOp":
            assert int(p.get(0, 0)) == 0, "only add supported"
            nodes.append(helper.make_node("Add", ins, outs, name=name))
        else:
            raise ValueError(f"unsupported layer {ltype}")

    assert reader.pos == len(reader.data), f"{len(reader.data) - reader.pos} unread weight bytes"

    # Rename graph endpoints to stable names the web page relies on.
    last_out = nodes[-1].output[0]
    for node in nodes:
        node.input[:] = ["input" if i == graph_in else i for i in node.input]
        node.output[:] = ["output" if o == last_out else o for o in node.output]

    graph = helper.make_graph(
        nodes,
        "realesr-animevideov3",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "H", "W"])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out_path)

    if fp16_path:
        from onnxconverter_common import float16

        # Keep float32 input/output so callers don't care which variant they load.
        onnx.save(float16.convert_float_to_float16(model, keep_io_types=True), fp16_path)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    convert(*sys.argv[1:])
