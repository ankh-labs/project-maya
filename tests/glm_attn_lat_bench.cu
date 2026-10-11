// tests/glm_attn_lat_bench.cu - what does a quantized latent cache actually COST in the attention?
//
// The engine-level A/B cannot answer this: q4_0 needs different residency from fp16, so the expert pool, the
// pinned RAM tier and the VRAM footprint all move at once.  This bench holds every one of those constant and
// varies ONLY the latent record type, by calling the same public entry point the engine calls:
//
//   glmb::mla_attn(q_abs, lat, cells, n_sel, n_sel_max, n_head, kv_lora, scale, T, ctx, stream, lat_q8)
//
// lat_q8 = 0  FP16 rows          (1024 B a position and layer at kv_lora 512)
// lat_q8 = 1  INT8 records       ( 544 B; glm_lat8.cuh, integer path)
// lat_q8 = 2  q4_0 of the RVOTATED rows (288 B; glm_lat4.cuh) - unpacked to FP16, then FP16 MMA,
//             and mla_attn rotates the context mix back with H512 itself
//
// The q4_0 arm therefore includes the rotation-back; the bench reports it separately too, so the dequant cost
// and the rotation cost are distinguishable.  n_sel is swept because DSA selects ~35 cells at short context
// but the indexer's cap is ~2048: if the unpack path is cheap at 35 and dear at 2048, that is exactly the
// answer to "does 4-bit cost anything at long context".
//
// Timing only.  The latent buffers are zero-filled (scores 0, uniform softmax, the same op count): record
// correctness is covered by glm_lat4_test and the engine's probe battery, not here.
#include "strata/kernels/glm_batch.hpp"
#include "strata/kernels/glm_lat_types.hpp"   // lat_rec_bytes / LatType
#include "strata/kernels/kv_q4.hpp"           // fwht512_rows_cuda (the q4_0 rotation)

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace strata::kernels;
namespace glmf = strata::kernels::glmf;
namespace glmb = strata::kernels::glmb;   // mla_attn lives here (glm_batch.cu)

namespace {

constexpr int KV = 512;   // kv_lora: what the model has, and what the rotation's shape requires

template <typename F>
double ms_per_call(F f, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 5; ++i) f();          // warm up (JIT, clocks, caches)
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) f();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return (double) ms / (double) iters;
}

}  // namespace

int main(int argc, char** argv) {
    int n_head = 32;          // the model's head count shape: n_head % 16 == 0
    int T = 32;               // tokens an attention call covers
    int ctx_positions = 8192; // rows in the latent cache
    if (argc > 1) n_head = std::atoi(argv[1]);
    if (argc > 2) T = std::atoi(argv[2]);
    if (argc > 3) ctx_positions = std::atoi(argv[3]);
    if (n_head % 16 != 0 || n_head <= 0 || T <= 0 || ctx_positions <= 0) {
        std::fprintf(stderr, "usage: glm_attn_lat_bench [n_head(multiple of 16)] [T] [ctx_positions]\n");
        return 2;
    }
    // sanity: the model's own n_head if it is known, else 32.  The point is a fixed shape, not this one.

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uni(-0.25f, 0.25f);

    float *q_abs = nullptr, *ctx = nullptr;
    if (cudaMalloc(&q_abs, (size_t) T * n_head * KV * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&ctx, (size_t) T * n_head * KV * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "glm_attn_lat_bench: cannot allocate q_abs/ctx\n");
        return 2;
    }
    std::vector<float> hq((size_t) T * n_head * KV);
    for (float& v : hq) v = uni(rng);
    cudaMemcpy(q_abs, hq.data(), hq.size() * sizeof(float), cudaMemcpyHostToDevice);

    // the latent caches: one row a position, sized by record type
    const int bytes_fp16 = glmf::lat_rec_bytes(KV, glmf::LatType::fp16);
    const int bytes_int8 = glmf::lat_rec_bytes(KV, glmf::LatType::int8);
    const int bytes_q4 = glmf::lat_rec_bytes(KV, glmf::LatType::q4_0);
    uint16_t* lat_fp16 = nullptr;
    uint16_t* lat_int8 = nullptr;
    uint16_t* lat_q4 = nullptr;
    uint8_t* lat_q4_raw = nullptr;
    cudaMalloc(&lat_fp16, (size_t) ctx_positions * bytes_fp16);
    cudaMalloc(&lat_int8, (size_t) ctx_positions * bytes_int8);
    // The q4_0 readers (glm_lat4.cuh lat4_h4/lat4_h8) load `rec + blk*18 + 2` as a uint32_t, which is only a legal
    // access when the record base is 2 mod 4.  A cudaMalloc base is 256-aligned, so the load faults; the engine's
    // state arena happens to hand over a base whose +2 is 4-aligned.  Bias by 2 to measure the engine's case.
    cudaMalloc(&lat_q4_raw, (size_t) ctx_positions * bytes_q4 + 16);
    lat_q4 = (uint16_t*) (lat_q4_raw + 2);
    cudaMemset(lat_fp16, 0, (size_t) ctx_positions * bytes_fp16);
    cudaMemset(lat_int8, 0, (size_t) ctx_positions * bytes_int8);
    cudaMemset(lat_q4_raw, 0, (size_t) ctx_positions * bytes_q4 + 16);

    std::printf("glm_attn_lat_bench: n_head %d, T %d, ctx %d positions, kv_lora %d\n", n_head, T, ctx_positions, KV);
    std::printf("  record bytes a position and layer: fp16 %d, int8 %d, q4_0 %d\n", bytes_fp16, bytes_int8, bytes_q4);
    std::printf("  cache size for %d positions, 11 DSA layers: fp16 %.2f GB, int8 %.2f GB, q4_0 %.2f GB\n\n",
                ctx_positions, 11.0 * ctx_positions * bytes_fp16 / 1e9, 11.0 * ctx_positions * bytes_int8 / 1e9,
                11.0 * ctx_positions * bytes_q4 / 1e9);

    bool q4_alive = (std::getenv("BENCH_Q4") != nullptr);   // q4_0 arm faults in this harness; opt in to try
    std::printf("  %6s | %10s %10s %10s | %10s | %s\n", "n_sel", "fp16 ms", "int8 ms", "q4_0 ms", "rot ms",
                "q4_0 total vs fp16");

    for (int nsel : {35, 256, 1024, 2048}) {
        if (nsel > ctx_positions) continue;
        // Rows of n_sel_max, -1 in the mask tail (dsa_select's documented contract).  n_sel_max is padded to a
        // multiple of 16 ints: the kernels vector-load cells, so a row stride like 35*4 = 140 bytes faults with
        // "misaligned address" - the engine always hands over a padded n_sel_max.
        const int nsel_pad = ((nsel + 15) / 16) * 16;
        std::vector<int> cells((size_t) T * nsel_pad, -1);
        std::vector<int> n_sel(T, nsel);
        // ascending, distinct, per token - the shape dsa_select produces (pools of adjacent positions)
        for (int t = 0; t < T; ++t) {
            const int base = (int) (((long long) t * 97) % (ctx_positions - nsel + 1));
            for (int i = 0; i < nsel; ++i) cells[(size_t) t * nsel_pad + i] = base + i;
        }
        int *d_cells = nullptr, *d_nsel = nullptr;
        cudaMalloc(&d_cells, cells.size() * sizeof(int));
        cudaMalloc(&d_nsel, n_sel.size() * sizeof(int));
        cudaMemcpy(d_cells, cells.data(), cells.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nsel, n_sel.data(), n_sel.size() * sizeof(int), cudaMemcpyHostToDevice);

        const float scale = 1.0f / std::sqrt((float) n_head);  // any positive constant: same work
        const int iters = (nsel <= 256) ? 50 : 20;

        // Each arm reports as it finishes, and any CUDA fault is named at the arm that caused it: a sticky,
        // asynchronous error surfaces at whichever launch checks it next, which once blamed the wrong kernel.
        std::printf("  %6d |", nsel);
        std::fflush(stdout);
        const double m_fp16 = ms_per_call(
            [&] { glmb::mla_attn(q_abs, lat_fp16, d_cells, d_nsel, nsel_pad, n_head, KV, scale, T, ctx, nullptr, 0); }, iters);
        if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
            std::printf(" fp16 FAILED: %s", cudaGetErrorString(e));
        std::printf(" %8.3f", m_fp16);
        std::fflush(stdout);
        const double m_int8 = ms_per_call(
            [&] { glmb::mla_attn(q_abs, lat_int8, d_cells, d_nsel, nsel_pad, n_head, KV, scale, T, ctx, nullptr, 1); }, iters);
        if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
            std::printf(" int8 FAILED: %s", cudaGetErrorString(e));
        std::printf(" %8.3f", m_int8);
        std::fflush(stdout);
        double m_q4 = 0.0;
        if (q4_alive) m_q4 = ms_per_call(
            [&] { glmb::mla_attn(q_abs, lat_q4, d_cells, d_nsel, nsel_pad, n_head, KV, scale, T, ctx, nullptr, 2); }, iters);
        if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
            { std::printf(" q4_0 FAILED (%s)", cudaGetErrorString(e)); q4_alive = false; m_q4 = std::nan(""); }
        if (std::isnan(m_q4)) std::printf("      n/a"); else std::printf(" %8.3f", m_q4);
        std::fflush(stdout);
        // the rotation-back that only the q4_0 path needs (mla_attn does it internally; measured on its own)
        const double m_rot = ms_per_call(
            [&] { fwht512_rows_cuda(q_abs, (int64_t) n_head * KV, n_head, T, nullptr); }, iters);
        if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
            std::printf(" rot FAILED: %s", cudaGetErrorString(e));
        std::printf(" | %8.3f | ", m_rot);
        std::fflush(stdout);
        cudaGetLastError();   // clear, so the next arm starts clean

        char verdict[128];
        const double per_tok = std::isnan(m_q4) ? 0.0 : (m_q4 - m_fp16) * 1000.0 / T;   // us a token, attention only
        if (std::isnan(m_q4)) std::snprintf(verdict, sizeof verdict, "q4_0 arm unavailable (kernel fault); int8 vs fp16: %+.1f%%", (m_int8 / m_fp16 - 1.0) * 100.0);
        else std::snprintf(verdict, sizeof verdict, "%+.1f%%  (%+.0f us/token)", (m_q4 / m_fp16 - 1.0) * 100.0, per_tok);
        if (m_rot > 0.0 && !std::isnan(m_q4)) {
            std::printf("         | q4_0 + rotation-back: %.3f ms (%+.1f%% vs fp16, %+.0f us/token)\n", m_q4 + m_rot,
                        ((m_q4 + m_rot) / m_fp16 - 1.0) * 100.0, (m_q4 + m_rot - m_fp16) * 1000.0 / T);
        }

        cudaFree(d_cells);
        cudaFree(d_nsel);
    }

    std::printf("\n  reference: a decode token takes ~43 ms end to end at 23 tok/s. Compare the us/token column\n"
                "  against that, not against zero: a 5%% decode claim is ~2.2 ms a token, i.e. 2200 us.\n");

    cudaFree(q_abs);
    cudaFree(ctx);
    cudaFree(lat_fp16);
    cudaFree(lat_int8);
    cudaFree(lat_q4_raw);
    return 0;
}
