# Cycle-Accurate Systolic Array Simulator for VGG16 (C++)

## What this is

A cycle-accurate simulator (in the SCALE-Sim / Timeloop sense: explicit,
documented cycle-timing model, not gate-level RTL) of a weight-stationary
NxN systolic array, driving full VGG16 inference layer-by-layer and
reporting per-layer cycle counts, PE utilization, and classification
accuracy.

## Files

- `include/systolic_array.h` — the core cycle-timing model. `SystolicArray::run_tile()`
  computes LOAD + FILL + STREAM + DRAIN cycles for one GEMM tile that fits
  in the array, and returns the real (numpy-equivalent) GEMM result.
- `include/gemm_tiler.h` — tiles arbitrarily large GEMMs (conv-via-im2col,
  or FC layers) into array-sized chunks and calls `run_tile()` repeatedly,
  accumulating cycles and partial sums across K-tiles.
- `include/tensor_ops.h` — im2col, weight reshaping, ReLU, maxpool, flatten.
- `include/json_lite.h` — minimal JSON parser (no external deps) for reading
  `layers.json` exported from PyTorch.
- `src/main.cpp` — driver: loads layer metadata + weights + golden test
  images, runs full inference through the systolic array model, prints
  per-layer cycle breakdown + summary (total cycles, utilization, accuracy).
- `make_dummy_data.py` — generates a tiny synthetic dataset (random weights,
  small 32x32 images, 3-conv-block toy network) in the same format as the
  real Colab export, for smoke-testing the pipeline before real data exists.

## Cycle-timing model (documented contract)

For one GEMM tile `A[M,K] @ B[K,N]` where `K <= ARRAY_ROWS` and `N <= ARRAY_COLS`:

```
load_cycles   = K * N                  (weights shifted in serially)
fill_latency  = K + N - 1              (systolic pipeline fill)
stream_cycles = M                      (M activation rows streamed through)
drain_latency = K + N - 1              (pipeline drain)
total_cycles  = load_cycles + fill_latency + stream_cycles + drain_latency
```

Larger GEMMs (every real VGG16 layer) are tiled: M is chunked by `tile_M`
(a throughput knob), K and N are chunked by `ARRAY_ROWS`/`ARRAY_COLS`.
Partial sums across K-tiles accumulate in software, matching how a real
accelerator would handle a reduction dimension larger than the array.

**Scoping choice (documented, defensible):** ReLU and MaxPool are treated
as free/off-array elementwise operations, not run through the systolic
model. Only GEMMs (conv-via-im2col, and FC layers) consume array cycles.
This is the standard scoping choice in systolic-array simulators since
these ops are typically handled by a separate vector unit in real hardware.

## Build

```bash
./build.sh
```//or manually:
```bash
g++ -O2 -std=c++17 -I include src/main.cpp -o vgg_sim
```

## Run

```bash
# Smoke test (already generated dummy data, random weights):
./vgg_sim test_export 16 16 3

# Real data (after you unzip the Colab export into ./export):
./vgg_sim export 16 16 50
```

Arguments: `<export_dir> <array_rows> <array_cols> <num_images>`

## Verified

- Compiles clean with `-Wall` (no warnings on core logic)
- Runs clean under AddressSanitizer (`-fsanitize=address`) — no memory errors
- Smoke-tested end-to-end on synthetic data: correct layer chaining
  (conv→relu→pool→conv→relu→pool→fc→relu→fc), correct dimension tiling,
  sensible cycle counts and utilization percentages reported per layer

## Next step

Once you have the real `vgg16_cifar10_export.zip` from Colab:
```bash
unzip vgg16_cifar10_export.zip -d export
./vgg_sim export 16 16 50
```
This runs real VGG16 (fine-tuned on CIFAR-10) inference through the
cycle-accurate systolic array model, reporting real accuracy and cycle
counts. Compare `sim_pred` vs `pytorch_pred` in the per-image output —
they should MATCH for every image if the systolic array arithmetic is
correct (floating point tiling order can cause tiny numerical differences
but argmax predictions should agree).
