#!/usr/bin/env python3
"""
Generates a tiny synthetic dataset in the SAME format the real Colab export
produces, so we can test the C++ simulator's plumbing (JSON parsing, weight
loading, im2col, tiling, layer chaining) before the real VGG16/CIFAR-10
export finishes. Uses random weights - NOT for accuracy validation, just
for making sure the C++ code runs end-to-end without crashing/hanging.
"""
import numpy as np
import json
import os
import shutil

OUT = "test_export"
if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(f"{OUT}/weights")
os.makedirs(f"{OUT}/golden")

rng = np.random.default_rng(0)

layers = []
idx = 0
in_shape = [3, 32, 32]  # small spatial size so this runs fast for a smoke test

def add_conv(name, in_c, out_c, k, stride, pad, in_shape):
    global idx
    h, w = in_shape[1], in_shape[2]
    h_out = (h + 2*pad - k)//stride + 1
    w_out = (w + 2*pad - k)//stride + 1
    meta = {
        "index": idx, "name": name, "type": "conv",
        "in_channels": in_c, "out_channels": out_c,
        "kernel_size": k, "stride": stride, "padding": pad,
        "in_shape": [in_c, h, w], "out_shape": [out_c, h_out, w_out],
        "weight_file": f"weights/{name}_weight.bin", "bias_file": f"weights/{name}_bias.bin",
        "weight_shape": [out_c, in_c, k, k], "bias_shape": [out_c],
    }
    idx += 1
    w_arr = (rng.standard_normal((out_c, in_c, k, k)) * 0.05).astype(np.float32)
    b_arr = (rng.standard_normal((out_c,)) * 0.01).astype(np.float32)
    w_arr.tofile(f"{OUT}/{meta['weight_file']}")
    b_arr.tofile(f"{OUT}/{meta['bias_file']}")
    layers.append(meta)
    return [out_c, h_out, w_out]

def add_relu(name, in_shape):
    global idx
    layers.append({"index": idx, "name": name, "type": "relu", "in_shape": in_shape, "out_shape": in_shape})
    idx += 1
    return in_shape

def add_pool(name, k, stride, in_shape):
    global idx
    c, h, w = in_shape
    h_out = (h-k)//stride+1
    w_out = (w-k)//stride+1
    layers.append({"index": idx, "name": name, "type": "maxpool", "kernel_size": k, "stride": stride,
                    "in_shape": [c,h,w], "out_shape": [c,h_out,w_out]})
    idx += 1
    return [c, h_out, w_out]

def add_fc(name, in_f, out_f):
    global idx
    meta = {
        "index": idx, "name": name, "type": "fc",
        "in_features": in_f, "out_features": out_f,
        "in_shape": [in_f], "out_shape": [out_f],
        "weight_file": f"weights/{name}_weight.bin", "bias_file": f"weights/{name}_bias.bin",
        "weight_shape": [out_f, in_f], "bias_shape": [out_f],
    }
    idx += 1
    w_arr = (rng.standard_normal((out_f, in_f)) * 0.02).astype(np.float32)
    b_arr = (rng.standard_normal((out_f,)) * 0.01).astype(np.float32)
    w_arr.tofile(f"{OUT}/{meta['weight_file']}")
    b_arr.tofile(f"{OUT}/{meta['bias_file']}")
    layers.append(meta)
    return out_f

# tiny 3-conv-block network (mimics VGG16 structure but small, for smoke test)
s = in_shape
s = add_conv("conv1_1", 3, 8, 3, 1, 1, s); s = add_relu("relu1_1", s)
s = add_conv("conv1_2", 8, 8, 3, 1, 1, s); s = add_relu("relu1_2", s)
s = add_pool("pool1", 2, 2, s)

s = add_conv("conv2_1", 8, 16, 3, 1, 1, s); s = add_relu("relu2_1", s)
s = add_pool("pool2", 2, 2, s)

flat = s[0]*s[1]*s[2]
f = add_fc("fc1", flat, 32)
_ = add_relu("relu_fc1", [32])
f = add_fc("fc2", 32, 10)

with open(f"{OUT}/layers.json", "w") as fp:
    json.dump(layers, fp, indent=2)

# 3 dummy test images
classes = ["a","b","c","d","e","f","g","h","i","j"]
for i in range(3):
    img = (rng.standard_normal((3,32,32)) * 0.5).astype(np.float32)
    img.tofile(f"{OUT}/golden/image_{i:02d}_input.bin")
    label = int(rng.integers(0,10))
    with open(f"{OUT}/golden/image_{i:02d}_label.txt","w") as fp:
        fp.write(str(label))
    with open(f"{OUT}/golden/image_{i:02d}_pred.txt","w") as fp:
        fp.write(str(label))  # fake "pytorch pred" = label for smoke test

print(f"Generated {len(layers)} layers, 3 dummy images in {OUT}/")
