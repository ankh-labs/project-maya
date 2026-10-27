// tests/glm_lat4_test.cu - the q4_0 latent records of stage B (GLM_KV_Q4_STAGE_B; src/kernels/cuda/glm_lat4.cuh):
//  1. the rotation is ORTHONORMAL: <H512 x, H512 y> == <x, y> to 1e-6 on random vectors (fwht512_rows_cuda against
//     a host FWHT reference).  An unscaled transform multiplies every score by 256^k and shifts the logits
//     systematically - which reads as a precision problem, not a bug, and would be blamed on q4_0 forever.
//  2. H is its own inverse: H(H x) == x to 1e-4 (the context mix is rotated back with the same kernel).
//  3. the records round-trip: lat4_store_warp -> lat4_f == (q - 8) * d bit for bit, and the vectorized decodes
//     lat4_h4 / lat4_h8 agree with lat4_f element for element (needs a GPU).
#include "../src/kernels/cuda/glm_lat4.cuh"
#include "strata/kernels/glm_lat_types.hpp"   // lat4_rec_bytes
#include "strata/kernels/kv_q4.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::kernels;

namespace {

constexpr int KV = 512;
constexpr int REC = strata::kernels::glmf::lat4_rec_bytes(KV);   // 288 at kv_lora 512

// 16 warps store one row: warp w writes block w (lane t stores element 32 w + t), glm_lat4.cuh's contract
__global__ void store_kernel(uint8_t* rec, const float* row, int kv) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (32 * warp + lane < kv) lat4_store_warp(row[32 * warp + lane], rec, kv, 32 * warp + lane, lane);
}

// the host reference: unnormalized FWHT within each 256 half, then 1/16 a half (orthonormal H512)
void fwht512_host(const float* in, float* out) {
    std::vector<float> t(in, in + KV);
    for (int half = 0; half < 2; ++half) {
        float* h = t.data() + half * 256;
        for (int stride = 1; stride < 256; stride <<= 1)
            for (int i = 0; i < 256; i += 2 * stride)
                for (int j = 0; j < stride; ++j) {
                    const float a = h[i + j], b = h[i + j + stride];
                    h[i + j] = a + b;
                    h[i + j + stride] = a - b;
                }
        for (int j = 0; j < 256; ++j) h[j] *= 1.0f / 16.0f;
    }
    for (int i = 0; i < KV; ++i) out[i] = t[i];
}

double dot(const float* a, const float* b) {
    double s = 0.0;
    for (int i = 0; i < KV; ++i) s += (double) a[i] * b[i];
    return s;
}
double dot(const std::vector<float>& a, const std::vector<float>& b) { return dot(a.data(), b.data()); }
double dot(const std::vector<float>& a, const float* b) { return dot(a.data(), b); }

}  // namespace

int main() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0) {
        std::fprintf(stderr, "glm_lat4_test: no GPU\n");
        return 2;
    }
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    int bad = 0;

    // 1 + 2: orthonormality and self-inverse, 64 random rows through fwht512_rows_cuda (dense rows)
    const int n_rows = 64;
    std::vector<float> x((size_t) n_rows * KV);
    for (float& v : x) v = uni(rng);
    float* dx = nullptr;
    cudaMalloc(&dx, x.size() * sizeof(float));
    cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    fwht512_rows_cuda(dx, (int64_t) KV, 1, n_rows, nullptr);       // H x (dense: col_stride KV, rows_per_col 1)
    std::vector<float> hx((size_t) n_rows * KV);
    cudaMemcpy(hx.data(), dx, hx.size() * sizeof(float), cudaMemcpyDeviceToHost);
    fwht512_rows_cuda(dx, (int64_t) KV, 1, n_rows, nullptr);       // H(H x), in place on the rotated rows
    std::vector<float> hh((size_t) n_rows * KV);
    cudaMemcpy(hh.data(), dx, hh.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    for (int r = 0; r < n_rows; ++r) {
        const float* xr = x.data() + (size_t) r * KV;
        const float* hxr = hx.data() + (size_t) r * KV;
        std::vector<float> ref(KV), y(KV), hy(KV);
        fwht512_host(xr, ref.data());
        double worst = 0.0;
        for (int i = 0; i < KV; ++i) worst = std::max(worst, (double) std::fabs(ref[i] - hxr[i]));
        if (worst > 1e-4) {
            std::fprintf(stderr, "glm_lat4_test: row %d: fwht512_rows_cuda != the host FWHT reference (worst %.3e)\n",
                         r, worst);
            ++bad;
        }
        const float* hhr = hh.data() + (size_t) r * KV;
        double self = 0.0;
        for (int i = 0; i < KV; ++i) self = std::max(self, (double) std::fabs(hhr[i] - xr[i]));
        if (self > 1e-4) {
            std::fprintf(stderr, "glm_lat4_test: row %d: H is not its own inverse (worst %.3e)\n", r, self);
            ++bad;
        }
        for (float& v : y) v = uni(rng);
        fwht512_host(y.data(), hy.data());
        const double want = dot(xr, y.data());                    // <x, y>
        const double got = dot(hxr, hy.data());                    // <H x, H y>
        if (std::fabs(got - want) > 1e-6 * std::max(1.0, std::fabs(want))) {   // relative: 512-term doubles
            std::fprintf(stderr, "glm_lat4_test: row %d: <Hx, Hy> != <x, y> (%.6f vs %.6f)\n", r, want, got);
            ++bad;
        }
    }

    // 3: the record round-trip - a warp stores, the decode helpers read back (q - 8) * d bit for bit
    {
        std::vector<float> row(KV);
        for (float& v : row) v = uni(rng) * 8.0f;
        uint8_t* drec = nullptr;
        cudaMalloc(&drec, REC);
        cudaMemset(drec, 0, REC);
        // (the row must be on the DEVICE: passing row.data() - host memory - made the kernel fault, write nothing
        // and leave the record zeroed, so every decode was -0.  The launch error was never checked either.)
        float* drow = nullptr;
        cudaMalloc(&drow, KV * sizeof(float));
        cudaMemcpy(drow, row.data(), KV * sizeof(float), cudaMemcpyHostToDevice);
        store_kernel<<<1, 512>>>(drec, drow, KV);
        const cudaError_t st = cudaDeviceSynchronize();
        if (st != cudaSuccess) {
            std::fprintf(stderr, "glm_lat4_test: store_kernel failed: %s\n", cudaGetErrorString(st));
            ++bad;
        }
        cudaFree(drow);
        std::vector<uint8_t> rec(REC);
        cudaMemcpy(rec.data(), drec, REC, cudaMemcpyDeviceToHost);
        cudaFree(drec);
        for (int c = 0; c < KV; ++c) {
            const int blk = c >> 5, t = c & 31;
            const float d = __half2float(((const __half*) rec.data())[blk * 9]);
            const uint8_t b = rec[blk * 18 + 2 + (t & 15)];   // element t low, t + 16 high (see lat4_f)
            const int q = t < 16 ? (b & 0xF) : (b >> 4);
            const float want = (float) (q - 8) * d;
            const float got = lat4_f(rec.data(), KV, c);
            if (got != want) {
                std::fprintf(stderr, "glm_lat4_test: lat4_f(%d) = %g, want %g\n", c, got, want);
                ++bad;
            }
            if (std::fabs(got - row[c]) > std::fabs(d) + 1e-3f) {   // one quantization step at most
                std::fprintf(stderr, "glm_lat4_test: lat4_f(%d) = %g is not row %g within a step (d %g)\n", c, got,
                             row[c], d);
                ++bad;
            }
        }
        // the vectorized reads agree with lat4_f element for element
        for (int c8 = 0; c8 < KV / 8; ++c8) {
            const uint4 v = lat4_h8(rec.data(), KV, c8);
            const __half* h = (const __half*) &v;
            for (int k = 0; k < 8; ++k) {
                const float want = __half2float(__float2half(lat4_f(rec.data(), KV, 8 * c8 + k)));   // h8 returns FP16
                if (__half2float(h[k]) != want) {
                    std::fprintf(stderr, "glm_lat4_test: lat4_h8(%d,%d) = %g, want %g\n", c8, k,
                                 __half2float(h[k]), want);
                    ++bad;
                }
            }
        }
        for (int c4 = 0; c4 < KV / 4; ++c4) {
            const uint2 v = lat4_h4(rec.data(), KV, c4);
            const __half* h = (const __half*) &v;
            for (int k = 0; k < 4; ++k) {
                const float want = __half2float(__float2half(lat4_f(rec.data(), KV, 4 * c4 + k)));   // h4 returns FP16
                if (__half2float(h[k]) != want) {
                    std::fprintf(stderr, "glm_lat4_test: lat4_h4(%d,%d) = %g, want %g\n", c4, k,
                                 __half2float(h[k]), want);
                    ++bad;
                }
            }
        }
    }

    if (bad == 0) {
        std::printf("glm_lat4_test: OK - orthonormal H512 (<Hx,Hy> == <x,y> to 1e-6), self-inverse, "
                    "records round-trip\n");
        return 0;
    }
    std::fprintf(stderr, "glm_lat4_test: %d failures\n", bad);
    return 1;
}
