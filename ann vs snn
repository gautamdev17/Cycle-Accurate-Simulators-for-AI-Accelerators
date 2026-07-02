"""
VGG16-CIFAR10 Systolic Array Sweep: ANN (MAC) vs SNN (T=8 parallel spike-accumulate)

Uses the SAME weight-stationary cycle-timing model as the C++ cycle-accurate
simulators already built for this project:

  LOAD phase:   K*N cycles (weights shifted into the K x N sub-grid of PEs)
  FILL latency: (K + N - 1) cycles (systolic pipeline fill)
  STREAM phase: M cycles (M activation rows streamed through)
  DRAIN phase:  (K + N - 1) cycles (pipeline drain)
  total_cycles = LOAD + FILL + STREAM + DRAIN

ANN PE:  multiply-accumulate (1 MAC per PE per cycle)
SNN PE:  T=8 PARALLEL spike-gated accumulate per PE per cycle, matching:

    for (int t=0; t<T; t++)
        acc[t] <= spike_in[t] ? acc[t] + $signed(wgt_in) : acc[t];

  i.e. each PE holds T independent accumulators and updates ALL T of them
  in the SAME cycle (T parallelized in hardware width, not serialized in
  time) - so SNN cycle count for a given tile is IDENTICAL to the ANN
  cycle count for that tile (same LOAD+FILL+STREAM+DRAIN formula). What
  differs between ANN and SNN is:
    - useful-work (utilization): ANN does 1 MAC/PE/cycle; SNN does up to
      T spike-gated accumulates/PE/cycle, but only where spike_in[t]==1
      (so SNN's ACTUAL useful work depends on spike sparsity)
    - energy: SNN's accumulate is cheaper than ANN's multiply-accumulate
      per operation, and spike sparsity means many of the T*K*N possible
      accumulates in a tile are skipped (never toggle silicon), so real
      SNN accelerators trade a T-wide datapath for a large energy
      reduction versus running T full ANN passes

Both ANN and SNN use synthetic (random) weights/spikes for now - this
script is structured so that swapping in real trained VGG16-CIFAR10
weights (conv/fc layer shapes, real activations, real calibrated spike
trains) tomorrow requires touching only the DATA GENERATION functions,
not the cycle/energy/EDP model.

Array is WEIGHT-STATIONARY throughout (weights loaded once per tile,
activations/spikes streamed through) - matching the C++ simulators.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
ARRAY_ROWS = 16
ARRAY_COLS = 16
SNN_TIMESTEPS = [4, 8, 16]   # sweep timesteps; T=8 is the primary/default config
DEFAULT_T = 8
CLOCK_FREQ_MHZ = 560
CLOCK_PERIOD_NS = 1.0 / CLOCK_FREQ_MHZ

# Per-PE energy costs (illustrative, consistent with the literature's framing
# that a spike-gated accumulate is cheaper per-op than a multiply-accumulate;
# tune these once you have real synthesis/power numbers)
ANN_MAC_ENERGY_PJ = 0.85     # energy per MAC operation (pJ)
SNN_ACCUM_ENERGY_PJ = 0.20   # energy per spike-gated accumulate operation (pJ)
IDLE_PE_ENERGY_PJ = 0.02     # leakage/idle energy per PE per cycle regardless of activity

# ============================================================
# VGG16 ARCHITECTURE (CIFAR-10 style: 224x224 input scaled down internally
# by 5 pool stages to 7x7, 13 conv + 3 FC = 16 learnable layers, 10-class
# output) - EXACT layer shapes matching your trained Kaggle model.
# ============================================================
def get_vgg16_cifar10_layers():
    """
    Returns list of layer dicts with full shape info needed for cycle-
    accurate GEMM tiling (im2col dimensions for conv, direct dims for fc).
    Matches the layers.json structure from your Kaggle ANN export.
    """
    layers = []

    def conv(name, in_c, out_c, in_hw, k=3, stride=1, pad=1):
        h_out = (in_hw + 2*pad - k)//stride + 1
        return {
            "type": "conv", "name": name,
            "in_c": in_c, "out_c": out_c, "k": k, "stride": stride, "pad": pad,
            "in_hw": in_hw, "out_hw": h_out,
            "M": h_out*h_out, "K": in_c*k*k, "N": out_c,
        }, h_out

    def pool(name, c, in_hw, k=2, stride=2):
        h_out = (in_hw - k)//stride + 1
        return {"type": "pool", "name": name, "c": c, "in_hw": in_hw, "out_hw": h_out}, h_out

    def fc(name, in_f, out_f):
        return {"type": "fc", "name": name, "M": 1, "K": in_f, "N": out_f}

    hw = 224
    l, hw = conv("conv1_1", 3, 64, hw);   layers.append(l)
    l, hw = conv("conv1_2", 64, 64, hw);  layers.append(l)
    l, hw = pool("pool1", 64, hw);        layers.append(l)

    l, hw = conv("conv2_1", 64, 128, hw); layers.append(l)
    l, hw = conv("conv2_2", 128, 128, hw);layers.append(l)
    l, hw = pool("pool2", 128, hw);       layers.append(l)

    l, hw = conv("conv3_1", 128, 256, hw);layers.append(l)
    l, hw = conv("conv3_2", 256, 256, hw);layers.append(l)
    l, hw = conv("conv3_3", 256, 256, hw);layers.append(l)
    l, hw = pool("pool3", 256, hw);       layers.append(l)

    l, hw = conv("conv4_1", 256, 512, hw);layers.append(l)
    l, hw = conv("conv4_2", 512, 512, hw);layers.append(l)
    l, hw = conv("conv4_3", 512, 512, hw);layers.append(l)
    l, hw = pool("pool4", 512, hw);       layers.append(l)

    l, hw = conv("conv5_1", 512, 512, hw);layers.append(l)
    l, hw = conv("conv5_2", 512, 512, hw);layers.append(l)
    l, hw = conv("conv5_3", 512, 512, hw);layers.append(l)
    l, hw = pool("pool5", 512, hw);       layers.append(l)

    flat = 512 * hw * hw  # 512*7*7 = 25088
    layers.append(fc("fc1", flat, 4096))
    layers.append(fc("fc2", 4096, 4096))
    layers.append(fc("fc3", 4096, 10))   # readout layer, 10 CIFAR-10 classes

    return layers


# ============================================================
# WEIGHT-STATIONARY CYCLE MODEL (shared by ANN and SNN - this is the
# core contract, identical to systolic_array.h / snn_systolic_array.h)
# ============================================================
def tile_cycles(K_tile, N_tile, M_tile):
    """One GEMM tile's LOAD+FILL+STREAM+DRAIN cycle cost."""
    load = K_tile * N_tile
    fill = K_tile + N_tile - 1
    stream = M_tile
    drain = K_tile + N_tile - 1
    return load + fill + stream + drain


def tile_gemm_cycles(M, K, N, rows=ARRAY_ROWS, cols=ARRAY_COLS, tile_M=64):
    """
    Tiles a full [M,K]x[K,N] GEMM across a rows x cols array, returns
    (total_cycles, num_tiles, macs_or_accum_capacity_slots).
    Same tiling strategy as gemm_tiler.h / snn_gemm_tiler.h.
    """
    total_cycles = 0
    num_tiles = 0
    capacity_pe_cycles = 0  # sum over tiles of (rows*cols*stream_cycles) - "slots available"

    for m0 in range(0, M, tile_M):
        mtile = min(tile_M, M - m0)
        for n0 in range(0, N, cols):
            ntile = min(cols, N - n0)
            for k0 in range(0, K, rows):
                ktile = min(rows, K - k0)
                c = tile_cycles(ktile, ntile, mtile)
                total_cycles += c
                num_tiles += 1
                capacity_pe_cycles += rows * cols * mtile

    return total_cycles, num_tiles, capacity_pe_cycles


# ============================================================
# SYNTHETIC DATA GENERATION
#   - ANN: activation sparsity via ReLU-like masking (post-ReLU tensors
#     are naturally sparse - typically 40-60% zeros in trained VGG nets,
#     we use a representative synthetic sparsity level here)
#   - SNN: T-wide spike trains per activation, generated with realistic
#     temporal sparsity (not every active neuron spikes every timestep)
#
#   NOTE: swap these two functions for real data tomorrow; everything
#   downstream (cycle/energy/EDP calculation) stays the same.
# ============================================================
def generate_ann_activation_sparsity(K, seed=0, sparsity=0.55):
    """Fraction of the K (reduction-dim) activations that are nonzero
    after ReLU, for one representative im2col row. Real VGG16 activations
    average roughly 40-60% sparsity post-ReLU; using 0.55 as a
    representative synthetic value until real activations are available."""
    rng = np.random.default_rng(seed)
    mask = rng.random(K) > sparsity
    return mask  # boolean: True = nonzero/active


def generate_snn_spike_train(K, T, seed=0, spatial_sparsity=0.55, per_timestep_spike_prob=0.35):
    """
    [K, T] boolean spike matrix. A synapse is only ever able to spike if
    it's spatially active (matches the ANN's active/nonzero mask, since
    SNN conversion preserves which weights matter) - then, given spatial
    activity, each timestep independently spikes with per_timestep_spike_prob.
    This matches standard rate-coding assumptions used in ANN->SNN literature
    and in the reference energy/EDP sweep's temporal sparsity model.
    """
    rng = np.random.default_rng(seed)
    spatial_mask = rng.random(K) > spatial_sparsity  # same active/inactive pattern as ANN
    spikes = np.zeros((K, T), dtype=bool)
    active_idx = np.where(spatial_mask)[0]
    for k in active_idx:
        spikes[k, :] = rng.random(T) < per_timestep_spike_prob
    return spikes, spatial_mask


# ============================================================
# ANN MEASUREMENT: cycles, useful MACs, utilization, energy
# ============================================================
def measure_ann_layer(layer, seed):
    M, K, N = layer["M"], layer["K"], layer["N"]
    cycles, num_tiles, capacity_pe_cycles = tile_gemm_cycles(M, K, N)

    # useful work: for a representative row, how many of K reduction-dim
    # entries are actually nonzero (post-ReLU sparsity) - scaled to the
    # whole tile's MAC count for an energy estimate
    mask = generate_ann_activation_sparsity(K, seed=seed)
    active_frac = mask.mean()
    useful_macs = int(M * K * N * active_frac)  # PEs still compute for zero
                                                    # inputs in a dense systolic
                                                    # array (no skip logic) -
                                                    # this fraction instead
                                                    # informs an OPTIONAL sparse-
                                                    # aware energy estimate below
    capacity_macs = M * K * N  # dense array does every MAC regardless of sparsity

    return {
        "cycles": cycles,
        "num_tiles": num_tiles,
        "capacity_macs": capacity_macs,
        "useful_macs": useful_macs,          # informational (dense array still pays full energy)
        "active_frac": active_frac,
    }


def measure_ann_network(layers, num_images=5, seed0=0):
    per_layer = {l["name"]: {"cycles": 0, "capacity_macs": 0, "useful_macs": 0, "tiles": 0}
                 for l in layers if l["type"] in ("conv", "fc")}
    total_cycles = 0
    for img in range(num_images):
        for li, layer in enumerate(layers):
            if layer["type"] not in ("conv", "fc"):
                continue
            res = measure_ann_layer(layer, seed=seed0 + img*1000 + li)
            per_layer[layer["name"]]["cycles"] += res["cycles"]
            per_layer[layer["name"]]["capacity_macs"] += res["capacity_macs"]
            per_layer[layer["name"]]["useful_macs"] += res["useful_macs"]
            per_layer[layer["name"]]["tiles"] += res["num_tiles"]
            total_cycles += res["cycles"]
    avg_cycles = total_cycles / num_images
    return avg_cycles, per_layer


# ============================================================
# SNN MEASUREMENT: cycles (same formula as ANN), spike-gated accumulate
# utilization, energy (much lower per-op, gated by real spike sparsity)
# ============================================================
def measure_snn_layer(layer, T, seed):
    M, K, N = layer["M"], layer["K"], layer["N"]
    # SAME cycle formula as ANN - T is parallel in PE width, not serialized
    cycles, num_tiles, capacity_pe_cycles = tile_gemm_cycles(M, K, N)

    spikes, spatial_mask = generate_snn_spike_train(K, T, seed=seed)
    # useful accumulates: for each active K-row, count how many of the T
    # timesteps actually spiked, times N (each spike fans out to N output cols)
    spike_count = spikes.sum()  # total spikes across all K rows and T timesteps
    useful_accumulates = int(spike_count * N)
    # capacity: every PE could do T accumulates every cycle it's active (M streaming cycles)
    capacity_accumulates = ARRAY_ROWS * ARRAY_COLS * T * M

    return {
        "cycles": cycles,               # identical formula to ANN (T parallel in width)
        "num_tiles": num_tiles,
        "capacity_accumulates": capacity_accumulates,
        "useful_accumulates": useful_accumulates,
        "spike_rate": spike_count / (K * T) if K > 0 else 0.0,
    }


def measure_snn_network(layers, T, num_images=5, seed0=0):
    per_layer = {l["name"]: {"cycles": 0, "capacity_accumulates": 0, "useful_accumulates": 0, "tiles": 0}
                 for l in layers if l["type"] in ("conv", "fc")}
    total_cycles = 0
    total_useful = 0
    total_capacity = 0
    for img in range(num_images):
        for li, layer in enumerate(layers):
            if layer["type"] not in ("conv", "fc"):
                continue
            res = measure_snn_layer(layer, T, seed=seed0 + img*1000 + li)
            per_layer[layer["name"]]["cycles"] += res["cycles"]
            per_layer[layer["name"]]["capacity_accumulates"] += res["capacity_accumulates"]
            per_layer[layer["name"]]["useful_accumulates"] += res["useful_accumulates"]
            per_layer[layer["name"]]["tiles"] += res["num_tiles"]
            total_cycles += res["cycles"]
            total_useful += res["useful_accumulates"]
            total_capacity += res["capacity_accumulates"]
    avg_cycles = total_cycles / num_images
    avg_useful = total_useful / num_images
    utilization = total_useful / total_capacity if total_capacity > 0 else 0.0
    return avg_cycles, avg_useful, utilization, per_layer


# ============================================================
# ENERGY / EDP
# ============================================================
def ann_energy_nj(capacity_macs_total):
    """Dense systolic array: every PE does a MAC every active cycle
    regardless of activation sparsity (no skip logic in a plain
    weight-stationary array), so energy scales with CAPACITY, not
    useful work. This matches real dense-systolic-array behavior."""
    return capacity_macs_total * ANN_MAC_ENERGY_PJ * 1e-3  # pJ -> nJ


def snn_energy_nj(useful_accumulates_total, capacity_accumulates_total):
    """
    Spike-gated accumulate: PEs only toggle (consume dynamic energy) when
    spike_in[t]==1. Non-spiking slots still draw idle/leakage energy but
    at a much lower rate. This is the key mechanism behind SNN energy
    efficiency claims in the literature.
    """
    active_energy = useful_accumulates_total * SNN_ACCUM_ENERGY_PJ * 1e-3
    idle_slots = max(capacity_accumulates_total - useful_accumulates_total, 0)
    idle_energy = idle_slots * IDLE_PE_ENERGY_PJ * 1e-3
    return active_energy + idle_energy


def calculate_metrics(cycles, energy_nj):
    latency_s = cycles / (CLOCK_FREQ_MHZ * 1e6)
    edp_js = (energy_nj * 1e-9) * latency_s
    return latency_s, edp_js


# ============================================================
# MAIN SWEEP
# ============================================================
def run_sweep(num_images=5):
    layers = get_vgg16_cifar10_layers()
    conv_fc_layers = [l for l in layers if l["type"] in ("conv", "fc")]
    print("="*80)
    print("VGG16-CIFAR10 SYSTOLIC ARRAY SWEEP: ANN (MAC) vs SNN (T-parallel accumulate)")
    print("="*80)
    print(f"Array size: {ARRAY_ROWS} x {ARRAY_COLS} PEs, weight-stationary")
    print(f"Clock: {CLOCK_FREQ_MHZ} MHz")
    print(f"Layers: {len(conv_fc_layers)} conv/fc "
          f"({sum(1 for l in conv_fc_layers if l['type']=='conv')} conv, "
          f"{sum(1 for l in conv_fc_layers if l['type']=='fc')} fc)")
    print(f"Images (synthetic): {num_images}")
    print("NOTE: using SYNTHETIC weights/activations/spikes. Swap in real Kaggle-trained")
    print("      weights and real calibrated SNN thresholds/spike trains once available.\n")

    # ---- ANN ----
    print("-"*80)
    print("Measuring ANN (dense weight-stationary systolic array, 1 MAC/PE/cycle)...")
    ann_cycles, ann_per_layer = measure_ann_network(conv_fc_layers, num_images=num_images)
    ann_capacity_macs_total = sum(v["capacity_macs"] for v in ann_per_layer.values()) / num_images
    ann_energy = ann_energy_nj(ann_capacity_macs_total)
    ann_latency, ann_edp = calculate_metrics(ann_cycles, ann_energy)
    print(f"  ANN avg cycles/image: {ann_cycles:,.0f}")
    print(f"  ANN avg energy/image: {ann_energy:,.1f} nJ")
    print(f"  ANN avg latency/image: {ann_latency*1e6:.2f} us")
    print(f"  ANN EDP: {ann_edp:.3e} J*s")

    # ---- SNN sweep over T ----
    snn_results = {}
    for T in SNN_TIMESTEPS:
        print("-"*80)
        print(f"Measuring SNN (T={T}, {ARRAY_ROWS}x{ARRAY_COLS} array, "
              f"each PE does {T} parallel spike-gated accumulates/cycle)...")
        snn_cycles, snn_useful, snn_util, snn_per_layer = measure_snn_network(
            conv_fc_layers, T, num_images=num_images)
        snn_capacity_total = sum(v["capacity_accumulates"] for v in snn_per_layer.values()) / num_images
        snn_useful_total = sum(v["useful_accumulates"] for v in snn_per_layer.values()) / num_images
        snn_energy = snn_energy_nj(snn_useful_total, snn_capacity_total)
        snn_latency, snn_edp = calculate_metrics(snn_cycles, snn_energy)

        print(f"  SNN avg cycles/image: {snn_cycles:,.0f}  "
              f"(same formula as ANN - T parallel in width, not time)")
        print(f"  SNN spike utilization: {snn_util*100:.2f}%")
        print(f"  SNN avg energy/image: {snn_energy:,.1f} nJ")
        print(f"  SNN avg latency/image: {snn_latency*1e6:.2f} us")
        print(f"  SNN EDP: {snn_edp:.3e} J*s")

        snn_results[T] = {
            "cycles": snn_cycles, "energy": snn_energy, "latency": snn_latency,
            "edp": snn_edp, "utilization": snn_util, "per_layer": snn_per_layer,
        }

    # ---- Comparison @ default T ----
    print("\n" + "="*80)
    print(f"ANN vs SNN COMPARISON (T={DEFAULT_T})")
    print("="*80)
    snn_default = snn_results[DEFAULT_T]
    print(f"{'Metric':<25} {'ANN':>18} {'SNN (T='+str(DEFAULT_T)+')':>18} {'SNN/ANN Ratio':>15}")
    print("-"*80)
    print(f"{'Cycles':<25} {ann_cycles:>18,.0f} {snn_default['cycles']:>18,.0f} "
          f"{snn_default['cycles']/ann_cycles:>15.3f}")
    print(f"{'Energy (nJ)':<25} {ann_energy:>18,.1f} {snn_default['energy']:>18,.1f} "
          f"{snn_default['energy']/ann_energy:>15.3f}")
    print(f"{'Latency (us)':<25} {ann_latency*1e6:>18.2f} {snn_default['latency']*1e6:>18.2f} "
          f"{snn_default['latency']/ann_latency:>15.3f}")
    print(f"{'EDP (J*s)':<25} {ann_edp:>18.3e} {snn_default['edp']:>18.3e} "
          f"{snn_default['edp']/ann_edp:>15.3f}")
    print("="*80)

    return {
        "layers": conv_fc_layers,
        "ann": {"cycles": ann_cycles, "energy": ann_energy, "latency": ann_latency,
                "edp": ann_edp, "per_layer": ann_per_layer},
        "snn": snn_results,
    }


# ============================================================
# PLOTTING
# ============================================================
def plot_comparison(results, filename="vgg16_cifar10_ann_vs_snn.png"):
    ann = results["ann"]
    snn = results["snn"]

    fig = plt.figure(figsize=(16, 10))

    # ---- Plot 1: Cycles ANN vs SNN(T) ----
    ax1 = plt.subplot(2, 3, 1)
    Ts = sorted(snn.keys())
    labels = ["ANN"] + [f"SNN T={t}" for t in Ts]
    cycles_vals = [ann["cycles"]] + [snn[t]["cycles"] for t in Ts]
    colors = ["steelblue"] + ["coral"]*len(Ts)
    ax1.bar(labels, cycles_vals, color=colors, alpha=0.85)
    ax1.set_ylabel("Cycles/image")
    ax1.set_title("Cycles: ANN vs SNN")
    ax1.grid(axis="y", alpha=0.3)
    for i, v in enumerate(cycles_vals):
        ax1.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8, rotation=0)

    # ---- Plot 2: Energy ----
    ax2 = plt.subplot(2, 3, 2)
    energy_vals = [ann["energy"]] + [snn[t]["energy"] for t in Ts]
    ax2.bar(labels, energy_vals, color=colors, alpha=0.85)
    ax2.set_ylabel("Energy/image (nJ)")
    ax2.set_title("Energy: ANN vs SNN")
    ax2.grid(axis="y", alpha=0.3)

    # ---- Plot 3: EDP ----
    ax3 = plt.subplot(2, 3, 3)
    edp_vals = [ann["edp"]] + [snn[t]["edp"] for t in Ts]
    ax3.bar(labels, edp_vals, color=colors, alpha=0.85)
    ax3.set_ylabel("EDP (J*s)")
    ax3.set_title("EDP: ANN vs SNN")
    ax3.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax3.grid(axis="y", alpha=0.3)

    # ---- Plot 4: SNN utilization vs T ----
    ax4 = plt.subplot(2, 3, 4)
    util_vals = [snn[t]["utilization"]*100 for t in Ts]
    ax4.plot(Ts, util_vals, marker="o", color="darkred")
    ax4.set_xlabel("Timesteps (T)")
    ax4.set_ylabel("Spike utilization (%)")
    ax4.set_title("SNN Array Utilization vs T")
    ax4.grid(alpha=0.3)

    # ---- Plot 5: Per-layer cycles (ANN, since cycles same for SNN) ----
    ax5 = plt.subplot(2, 3, 5)
    layer_names = list(ann["per_layer"].keys())
    layer_cycles = [ann["per_layer"][n]["cycles"] / len(Ts) if False else
                    ann["per_layer"][n]["cycles"] for n in layer_names]
    # normalize by num_images already baked into per_layer sums; show as-is (relative comparison)
    ax5.barh(layer_names, layer_cycles, color="steelblue", alpha=0.8)
    ax5.set_xlabel("Total cycles (summed over synthetic images)")
    ax5.set_title("Per-Layer Cycles (ANN = SNN, same formula)")
    ax5.tick_params(axis="y", labelsize=7)

    # ---- Plot 6: Energy ratio SNN/ANN vs T ----
    ax6 = plt.subplot(2, 3, 6)
    ratio_vals = [snn[t]["energy"]/ann["energy"] for t in Ts]
    ax6.plot(Ts, ratio_vals, marker="s", color="green")
    ax6.axhline(y=1.0, color="black", linestyle="--", alpha=0.4)
    ax6.set_xlabel("Timesteps (T)")
    ax6.set_ylabel("SNN Energy / ANN Energy")
    ax6.set_title("Energy Efficiency vs T")
    ax6.grid(alpha=0.3)

    plt.suptitle("VGG16-CIFAR10: ANN vs SNN Weight-Stationary Systolic Array\n"
                 "(SYNTHETIC DATA - swap in real trained weights when available)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"\nSaved plot: {filename}")
    plt.close()


if __name__ == "__main__":
    results = run_sweep(num_images=5)
    plot_comparison(results)
