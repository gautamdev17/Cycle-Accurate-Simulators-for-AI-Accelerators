#pragma once
#include "systolic_array.h"
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>

// ============================================================
// Drives the SystolicArray over GEMMs larger than the array itself.
// Full GEMM: C[M][N] = A[M][K] @ B[K][N] + bias[N]
// Tiles M into chunks of TILE_M (matches array's "stream" capacity, can be
// arbitrarily large since streaming just takes more cycles), and tiles
// K, N into chunks of ROWS, COLS respectively (must fit in the array).
// Partial sums across K-tiles are accumulated in software (matches how a
// real accelerator would accumulate partial sums across multiple weight
// loads when K > ROWS, i.e. "K-reduction tiling").
// ============================================================

struct GemmStats {
    uint64_t total_cycles = 0;
    uint64_t total_macs_performed = 0;
    uint64_t total_macs_capacity = 0;
    int num_tiles = 0;
    double utilization() const {
        return total_macs_capacity ? (double)total_macs_performed / (double)total_macs_capacity : 0.0;
    }
};

inline std::vector<std::vector<double>> tiled_gemm(
    SystolicArray& arr,
    const std::vector<std::vector<double>>& A, // [M][K]
    const std::vector<std::vector<double>>& B, // [K][N]
    const std::vector<double>& bias,           // [N]
    GemmStats& stats,
    int tile_M = 64) // how many activation rows to stream per tile call (throughput knob, not an array-size limit)
{
    int M = (int)A.size();
    int K = (int)A[0].size();
    int N = (int)B[0].size();
    int K_B = (int)B.size();
    int N_bias = (int)bias.size();
    if (K != K_B) {
        throw std::runtime_error("tiled_gemm: A's K (" + std::to_string(K) +
            ") != B's K (" + std::to_string(K_B) + ")");
    }
    if (!bias.empty() && N_bias != N) {
        throw std::runtime_error("tiled_gemm: bias size (" + std::to_string(N_bias) +
            ") != N (" + std::to_string(N) + ")");
    }

    std::vector<std::vector<double>> C(M, std::vector<double>(N, 0.0));

    int ROWS = arr.rows();
    int COLS = arr.cols();

    for (int m0 = 0; m0 < M; m0 += tile_M) {
        int m1 = std::min(M, m0 + tile_M);
        int mtile = m1 - m0;

        for (int n0 = 0; n0 < N; n0 += COLS) {
            int n1 = std::min(N, n0 + COLS);
            int ntile = n1 - n0;

            for (int k0 = 0; k0 < K; k0 += ROWS) {
                int k1 = std::min(K, k0 + ROWS);
                int ktile = k1 - k0;

                // build sub-tiles
                std::vector<std::vector<double>> Asub(mtile, std::vector<double>(ktile));
                for (int i = 0; i < mtile; i++)
                    for (int j = 0; j < ktile; j++)
                        Asub[i][j] = A[m0+i][k0+j];

                std::vector<std::vector<double>> Bsub(ktile, std::vector<double>(ntile));
                for (int i = 0; i < ktile; i++)
                    for (int j = 0; j < ntile; j++)
                        Bsub[i][j] = B[k0+i][n0+j];

                // bias only added once (on the first K-tile) to avoid double-adding
                std::vector<double> bias_sub;
                if (k0 == 0 && !bias.empty()) {
                    bias_sub.assign(bias.begin()+n0, bias.begin()+n1);
                } else {
                    bias_sub.assign(ntile, 0.0);
                }

                TileResult tr = arr.run_tile(Asub, Bsub, bias_sub);

                for (int i = 0; i < mtile; i++)
                    for (int j = 0; j < ntile; j++)
                        C[m0+i][n0+j] += tr.output[i][j];

                stats.total_cycles += tr.cycles;
                stats.total_macs_performed += tr.macs_performed;
                stats.total_macs_capacity += tr.macs_capacity;
                stats.num_tiles += 1;
            }
        }
    }

    return C;
}
