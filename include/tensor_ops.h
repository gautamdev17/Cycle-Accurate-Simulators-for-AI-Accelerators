#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Simple 3D tensor: [C][H][W], stored flat with helper accessors
struct Tensor3D {
    int C, H, W;
    std::vector<float> data; // size C*H*W

    Tensor3D() : C(0), H(0), W(0) {}
    Tensor3D(int c, int h, int w) : C(c), H(h), W(w), data((size_t)c*h*w, 0.0f) {}

    inline float& at(int c, int h, int w) {
        return data[((size_t)c*H + h)*W + w];
    }
    inline float at(int c, int h, int w) const {
        return data[((size_t)c*H + h)*W + w];
    }
};

// ------------------------------------------------------------
// im2col: converts a conv layer's input into a GEMM-ready matrix.
// Input:  [C_in][H][W], kernel k x k, stride s, padding p
// Output: A matrix of shape [H_out*W_out][C_in*k*k]  (M x K for the GEMM)
// Weight matrix B should be reshaped as [C_in*k*k][C_out] (K x N)
// ------------------------------------------------------------
inline std::vector<std::vector<double>> im2col(
    const Tensor3D& input, int k, int stride, int pad, int& H_out, int& W_out)
{
    int C = input.C, H = input.H, W = input.W;
    H_out = (H + 2*pad - k) / stride + 1;
    W_out = (W + 2*pad - k) / stride + 1;

    int M = H_out * W_out;
    int K = C * k * k;
    std::vector<std::vector<double>> A(M, std::vector<double>(K));

    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            int row = oh * W_out + ow;
            int col = 0;
            for (int c = 0; c < C; c++) {
                for (int kh = 0; kh < k; kh++) {
                    for (int kw = 0; kw < k; kw++) {
                        int ih = oh*stride - pad + kh;
                        int iw = ow*stride - pad + kw;
                        double val = 0.0;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            val = input.at(c, ih, iw);
                        }
                        A[row][col++] = val;
                    }
                }
            }
        }
    }
    return A;
}

// Reshape conv weights [C_out][C_in][k][k] (flat, PyTorch layout) into
// GEMM-ready B matrix [C_in*k*k][C_out]
inline std::vector<std::vector<double>> reshape_conv_weights(
    const std::vector<float>& w_flat, int C_out, int C_in, int k)
{
    int K = C_in * k * k;
    std::vector<std::vector<double>> B(K, std::vector<double>(C_out));
    for (int oc = 0; oc < C_out; oc++) {
        int idx = 0;
        for (int ic = 0; ic < C_in; ic++) {
            for (int kh = 0; kh < k; kh++) {
                for (int kw = 0; kw < k; kw++) {
                    size_t widx = (((size_t)oc*C_in + ic)*k + kh)*k + kw;
                    B[idx][oc] = w_flat[widx];
                    idx++;
                }
            }
        }
    }
    return B;
}

// Convert GEMM output [H_out*W_out][C_out] back into Tensor3D [C_out][H_out][W_out]
inline Tensor3D col2im_output(const std::vector<std::vector<double>>& C_mat,
                               int C_out, int H_out, int W_out)
{
    Tensor3D out(C_out, H_out, W_out);
    for (int oh = 0; oh < H_out; oh++) {
        for (int ow = 0; ow < W_out; ow++) {
            int row = oh * W_out + ow;
            for (int oc = 0; oc < C_out; oc++) {
                out.at(oc, oh, ow) = (float)C_mat[row][oc];
            }
        }
    }
    return out;
}

inline void relu_inplace(Tensor3D& t) {
    for (auto& v : t.data) v = std::max(0.0f, v);
}

inline Tensor3D maxpool2d(const Tensor3D& input, int k, int stride) {
    int C = input.C, H = input.H, W = input.W;
    int H_out = (H - k) / stride + 1;
    int W_out = (W - k) / stride + 1;
    Tensor3D out(C, H_out, W_out);
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                float m = -1e30f;
                for (int kh = 0; kh < k; kh++)
                    for (int kw = 0; kw < k; kw++)
                        m = std::max(m, input.at(c, oh*stride+kh, ow*stride+kw));
                out.at(c, oh, ow) = m;
            }
        }
    }
    return out;
}

// Flatten Tensor3D to a [1][C*H*W] matrix, matching PyTorch's flatten (C,H,W order)
inline std::vector<std::vector<double>> flatten_to_row(const Tensor3D& t) {
    std::vector<std::vector<double>> row(1, std::vector<double>(t.data.size()));
    for (size_t i = 0; i < t.data.size(); i++) row[0][i] = t.data[i];
    return row;
}

inline std::vector<std::vector<double>> reshape_fc_weights(
    const std::vector<float>& w_flat, int out_f, int in_f)
{
    // PyTorch Linear weight is [out_f][in_f]; GEMM needs B = [in_f][out_f]
    std::vector<std::vector<double>> B(in_f, std::vector<double>(out_f));
    for (int o = 0; o < out_f; o++)
        for (int i = 0; i < in_f; i++)
            B[i][o] = w_flat[(size_t)o*in_f + i];
    return B;
}
