#include "systolic_array.h"
#include "gemm_tiler.h"
#include "tensor_ops.h"
#include "json_lite.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cstdlib>

// ------------------------------------------------------------
// Utility: read a raw float32 binary file into a vector<float>
// ------------------------------------------------------------
std::vector<float> read_bin_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

int read_label_file(const std::string& path) {
    std::ifstream f(path);
    int label;
    f >> label;
    return label;
}

struct LayerInfo {
    int index;
    std::string name;
    std::string type; // conv, relu, maxpool, fc
    std::vector<int> in_shape, out_shape;
    // conv
    int in_channels=0, out_channels=0, kernel_size=0, stride=0, padding=0;
    std::string weight_file, bias_file;
    // fc
    int in_features=0, out_features=0;
};

std::vector<LayerInfo> load_layers(const std::string& json_path) {
    JsonValue root = parse_json_file(json_path);
    std::vector<LayerInfo> layers;
    for (auto& l : root.arr_val) {
        LayerInfo li;
        li.index = l["index"].asInt();
        li.name = l["name"].asString();
        li.type = l["type"].asString();
        li.in_shape = l["in_shape"].asIntArray();
        li.out_shape = l["out_shape"].asIntArray();
        if (li.type == "conv") {
            li.in_channels = l["in_channels"].asInt();
            li.out_channels = l["out_channels"].asInt();
            li.kernel_size = l["kernel_size"].asInt();
            li.stride = l["stride"].asInt();
            li.padding = l["padding"].asInt();
            li.weight_file = l["weight_file"].asString();
            li.bias_file = l["bias_file"].asString();
        } else if (li.type == "maxpool") {
            li.kernel_size = l["kernel_size"].asInt();
            li.stride = l["stride"].asInt();
        } else if (li.type == "fc") {
            li.in_features = l["in_features"].asInt();
            li.out_features = l["out_features"].asInt();
            li.weight_file = l["weight_file"].asString();
            li.bias_file = l["bias_file"].asString();
        }
        layers.push_back(li);
    }
    return layers;
}

struct LayerCycleReport {
    std::string name, type;
    uint64_t cycles = 0;
    uint64_t macs_performed = 0;
    uint64_t macs_capacity = 0;
    int tiles = 0;
};

int main(int argc, char** argv) {
    // ---------------- CONFIG ----------------
    std::string export_dir = argc > 1 ? argv[1] : "export";
    int ARRAY_ROWS = argc > 2 ? std::atoi(argv[2]) : 16;
    int ARRAY_COLS = argc > 3 ? std::atoi(argv[3]) : 16;
    int NUM_IMAGES = argc > 4 ? std::atoi(argv[4]) : 10;

    std::cout << "===========================================================\n";
    std::cout << " Cycle-Accurate Systolic Array Simulator - VGG16 Inference\n";
    std::cout << "===========================================================\n";
    std::cout << "Export dir: " << export_dir << "\n";
    std::cout << "Array size: " << ARRAY_ROWS << " x " << ARRAY_COLS << " PEs\n";
    std::cout << "Num images: " << NUM_IMAGES << "\n\n";

    // ---------------- LOAD LAYER METADATA ----------------
    std::vector<LayerInfo> layers = load_layers(export_dir + "/layers.json");
    std::cout << "Loaded " << layers.size() << " layers from layers.json\n";

    // preload all weights once
    struct WeightSet { std::vector<float> w, b; };
    std::vector<WeightSet> weights(layers.size());
    for (size_t i = 0; i < layers.size(); i++) {
        if (layers[i].type == "conv" || layers[i].type == "fc") {
            weights[i].w = read_bin_f32(export_dir + "/" + layers[i].weight_file);
            weights[i].b = read_bin_f32(export_dir + "/" + layers[i].bias_file);
        }
    }
    std::cout << "Loaded all layer weights.\n\n";

    // ---------------- SYSTOLIC ARRAY ----------------
    SystolicArray arr(ARRAY_ROWS, ARRAY_COLS);

    // per-layer aggregate stats across all images
    std::vector<LayerCycleReport> layer_reports(layers.size());
    for (size_t i = 0; i < layers.size(); i++) {
        layer_reports[i].name = layers[i].name;
        layer_reports[i].type = layers[i].type;
    }

    int correct = 0;
    uint64_t total_cycles_all_images = 0;

    auto wall_start = std::chrono::high_resolution_clock::now();

    for (int img = 0; img < NUM_IMAGES; img++) {
        char tagbuf[32];
        snprintf(tagbuf, sizeof(tagbuf), "image_%02d", img);
        std::string tag = tagbuf;

        std::string input_path = export_dir + "/golden/" + tag + "_input.bin";
        std::string label_path = export_dir + "/golden/" + tag + "_label.txt";
        std::string pred_path  = export_dir + "/golden/" + tag + "_pred.txt";

        std::ifstream check(input_path);
        if (!check.good()) {
            std::cout << "  [skip] " << tag << " not found, stopping image loop.\n";
            break;
        }
        check.close();

        std::vector<float> input_flat = read_bin_f32(input_path);
        int true_label = read_label_file(label_path);
        int pytorch_pred = read_label_file(pred_path);

        // Determine input tensor shape dynamically from the first layer's in_shape
        // (avoids hardcoding image size, so this works for both the 224x224 real
        // export and smaller smoke-test data).
        int in_C = layers[0].in_shape[0];
        int in_H = layers[0].in_shape[1];
        int in_W = layers[0].in_shape[2];
        Tensor3D activation(in_C, in_H, in_W);
        if (input_flat.size() != activation.data.size()) {
            throw std::runtime_error("Input image size (" + std::to_string(input_flat.size()) +
                ") doesn't match expected " + std::to_string(activation.data.size()) +
                " from layers.json first layer in_shape");
        }
        activation.data = input_flat; // NCHW, N=1, matches Tensor3D layout

        uint64_t image_cycles = 0;

        for (size_t li = 0; li < layers.size(); li++) {
            LayerInfo& L = layers[li];

            if (L.type == "conv") {
                int H_out, W_out;
                auto A = im2col(activation, L.kernel_size, L.stride, L.padding, H_out, W_out);
                auto B = reshape_conv_weights(weights[li].w, L.out_channels, L.in_channels, L.kernel_size);
                std::vector<double> bias(weights[li].b.begin(), weights[li].b.end());

                GemmStats stats;
                auto C = tiled_gemm(arr, A, B, bias, stats, /*tile_M=*/64);

                activation = col2im_output(C, L.out_channels, H_out, W_out);

                layer_reports[li].cycles += stats.total_cycles;
                layer_reports[li].macs_performed += stats.total_macs_performed;
                layer_reports[li].macs_capacity += stats.total_macs_capacity;
                layer_reports[li].tiles += stats.num_tiles;
                image_cycles += stats.total_cycles;

            } else if (L.type == "relu") {
                relu_inplace(activation);
                // ReLU treated as free (elementwise, not run through the systolic array) - documented scoping choice

            } else if (L.type == "maxpool") {
                activation = maxpool2d(activation, L.kernel_size, L.stride);
                // pooling likewise treated as free / off-array overhead

            } else if (L.type == "fc") {
                std::vector<std::vector<double>> A = flatten_to_row(activation);
                auto B = reshape_fc_weights(weights[li].w, L.out_features, L.in_features);
                std::vector<double> bias(weights[li].b.begin(), weights[li].b.end());

                GemmStats stats;
                auto C = tiled_gemm(arr, A, B, bias, stats, /*tile_M=*/1);

                activation = Tensor3D(1, 1, L.out_features);
                for (int j = 0; j < L.out_features; j++) activation.data[j] = (float)C[0][j];

                layer_reports[li].cycles += stats.total_cycles;
                layer_reports[li].macs_performed += stats.total_macs_performed;
                layer_reports[li].macs_capacity += stats.total_macs_capacity;
                layer_reports[li].tiles += stats.num_tiles;
                image_cycles += stats.total_cycles;
            }
        }

        // argmax over final activation (10-class logits)
        int pred = 0;
        float best = activation.data[0];
        for (size_t j = 1; j < activation.data.size(); j++) {
            if (activation.data[j] > best) { best = activation.data[j]; pred = (int)j; }
        }

        bool sim_correct = (pred == true_label);
        bool matches_pytorch = (pred == pytorch_pred);
        if (sim_correct) correct++;
        total_cycles_all_images += image_cycles;

        std::cout << "  " << tag
                  << " | true=" << true_label
                  << " sim_pred=" << pred
                  << " pytorch_pred=" << pytorch_pred
                  << (matches_pytorch ? "  [MATCH]" : "  [MISMATCH]")
                  << " | cycles=" << image_cycles << "\n";
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_secs = std::chrono::duration<double>(wall_end - wall_start).count();

    // ---------------- REPORT ----------------
    std::cout << "\n===========================================================\n";
    std::cout << " PER-LAYER CYCLE BREAKDOWN (summed across all images run)\n";
    std::cout << "===========================================================\n";
    std::cout << std::left << std::setw(12) << "Layer" << std::setw(8) << "Type"
              << std::setw(14) << "Cycles" << std::setw(10) << "Tiles" << "Utilization%\n";
    uint64_t grand_total_cycles = 0;
    for (auto& r : layer_reports) {
        if (r.cycles == 0) continue;
        double util = r.macs_capacity ? 100.0 * (double)r.macs_performed / (double)r.macs_capacity : 0.0;
        std::cout << std::left << std::setw(12) << r.name << std::setw(8) << r.type
                  << std::setw(14) << r.cycles << std::setw(10) << r.tiles
                  << std::fixed << std::setprecision(2) << util << "\n";
        grand_total_cycles += r.cycles;
    }

    std::cout << "\n===========================================================\n";
    std::cout << " SUMMARY\n";
    std::cout << "===========================================================\n";
    std::cout << "Array size:              " << ARRAY_ROWS << " x " << ARRAY_COLS << " PEs\n";
    std::cout << "Images run:              " << NUM_IMAGES << "\n";
    std::cout << "Total cycles (all imgs): " << grand_total_cycles << "\n";
    std::cout << "Avg cycles/image:        " << (NUM_IMAGES ? grand_total_cycles / NUM_IMAGES : 0) << "\n";
    std::cout << "Simulator accuracy:      " << (100.0 * correct / NUM_IMAGES) << "% (" << correct << "/" << NUM_IMAGES << ")\n";
    std::cout << "Wall-clock sim time:     " << wall_secs << " s\n";

    return 0;
}
