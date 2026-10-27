// src/kernels/cuda/glm_lat4.cuh - the q4_0 latent records (STRATA_GLM_KV_Q4; glm_lat_types.hpp lat4_rec_bytes).
// Stage B of the GLM latent q4_0 work.  The format is kv_q4.hpp's block_q4_0 exactly (18 B: fp16 scale + 16
// nibble bytes, element t in the low nibble of qs[t], t + 16 in the high one, value = (q - 8) * d), and the
// latents are ROTATED by the 512-point Hadamard before quantizing (two fwht256 halves - H512 = H256 . H256,
// orthonormal at 1/16 a half): the rotation spreads outlier channels over the block, which is what makes
// 4 bits usable (kv_q4.cu q4_group, PR #21).  The absorbed q_abs lives in the same 512-dim space and is
// rotated the same way, so <Hq, Hk> = <q, k> and the scores are unchanged; the context mix of rotated
// latents is rotated back with H (self-inverse) before the wv_b projection.  The DSA indexer caches (ik/ig)
// are separate projections from the hidden state - untouched.
//
// INVARIANT (hard test, glm_lat4_test): <H512 x, H512 y> == <x, y> to 1e-6 on random vectors.  An unscaled
// transform multiplies every score by 256^k and shifts the logits systematically - which reads as a precision
// problem, not a bug, and would be blamed on q4_0 forever.
//
// NOT INCLUDED anywhere yet: the read arms and the rotations land with the stage-B kernel sweep (glm_batch.cu's
// eight attention kernels, glm_fast.cu's lat_any/lat_put/mla_kernel/dsa_prep, the fused mla_kernel's in-smem
// butterflies).  Until that sweep is complete lat_records_ stays INT8 - a partial q4_0 path (some kernels
// reading records, others fp16 rows) is silent corruption, so stage B lands all at once or not at all.
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

namespace strata::kernels::glmf {

__device__ __forceinline__ const uint8_t* lat4_rec(const uint16_t* lat, int kv, int cell) {
    return (const uint8_t*) lat + (size_t) (kv / 2 + kv / 32) * cell;
}

// value c of a record: block c >> 5, the scale one fp16 at the block's head, nibble byte (c % 32) / 2 with
// c % 32 < 16 in the low nibble (element t) and 16..31 in the high one (element t + 16)
__device__ __forceinline__ float lat4_f(const uint8_t* rec, int kv, int c) {
    const int blk = c >> 5, t = c & 31;
    const float d = __half2float(((const __half*) rec)[blk * 9]);
    const uint8_t b = rec[blk * 18 + 2 + t / 2];
    const int q = t < 16 ? (b & 0xF) : (b >> 4);
    return (float) (q - 8) * d;
}

// values [4 c4, 4 c4 + 4) of a record as 4 FP16 (an FP16 row's uint2 there); 4 values sit in one block
__device__ __forceinline__ uint2 lat4_h4(const uint8_t* rec, int kv, int c4) {
    const int blk = c4 >> 3, j = (c4 & 3) * 4;   // (4 * c4) % 32 elements into the block
    const float d = __half2float(((const __half*) rec)[blk * 9]);
    const uint32_t w = ((const uint32_t*) (rec + blk * 18 + 2))[j / 4];
    __half v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint8_t b = (uint8_t) (w >> (8 * k));
        const int q0 = b & 0xF, q1 = b >> 4;
        v[2 * k] = __float2half((float) (q0 - 8) * d);
        v[2 * k + 1] = __float2half((float) (q1 - 8) * d);
    }
    uint2 out;
    out.x = (uint32_t) (__ushort_as_ushort(((const uint16_t*) v)[0]) |
                        ((uint32_t) __ushort_as_ushort(((const uint16_t*) v)[1]) << 16));
    out.y = (uint32_t) (__ushort_as_ushort(((const uint16_t*) v)[2]) |
                        ((uint32_t) __ushort_as_ushort(((const uint16_t*) v)[3]) << 16));
    return out;
}

// values [8 c8, 8 c8 + 8) of a record as 8 FP16 (an FP16 row's uint4 there); 8 values sit in one block
__device__ __forceinline__ uint4 lat4_h8(const uint8_t* rec, int kv, int c8) {
    const int blk = c8 >> 2, j = (c8 & 3) * 4;
    const float d = __half2float(((const __half*) rec)[blk * 9]);
    const uint32_t w = ((const uint32_t*) (rec + blk * 18 + 2))[j / 4];
    __half v[8];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint8_t b = (uint8_t) (w >> (8 * k));
        const int q0 = b & 0xF, q1 = b >> 4;
        v[2 * k] = __float2half((float) (q0 - 8) * d);
        v[2 * k + 1] = __float2half((float) (q1 - 8) * d);
    }
    uint4 out;
    unsigned* o = (unsigned*) &out;
    const unsigned* vv = (const unsigned*) v;
    o[0] = vv[0];
    o[1] = vv[1];
    o[2] = vv[2];
    o[3] = vv[3];
    return out;
}

// one 32-value group in one warp (lane = element), ggml's q4_0 with kv_q4.cu q4_group's tie-safe reduction:
// d = (the value of largest |x|) / -8, q = clamp(trunc(x / d + 8.5), 0, 15); ties in |x| resolve to the larger
// value in EVERY lane, so all lanes agree on d.  Returns the scale bits; `byte` is lane t's packed byte for
// t < 16.  Mirrors kv_q4.cu's q4_group (kept local: that one is internal to kv_q4.cu).
__device__ __forceinline__ uint16_t lat4_group(float x, int lane, uint8_t& byte) {
    float amax = fabsf(x), mval = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float a = __shfl_xor_sync(0xffffffffu, amax, o);
        const float v = __shfl_xor_sync(0xffffffffu, mval, o);
        if (a > amax || (a == amax && v > mval)) { amax = a; mval = v; }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int q = __float2int_rz(x * id + 8.5f);
    const uint8_t qc = (uint8_t) (q < 0 ? 0 : (q > 15 ? 15 : q));
    const uint8_t qhi = __shfl_down_sync(0xffffffffu, qc, 16);
    byte = (uint8_t) (qc | (qhi << 4));
    (void) lane;
    return __half_as_ushort(__float2half(d));
}

// value c of a latent row y into row `row` of a cache as a q4_0 record (one warp = one 32-value block)
__device__ __forceinline__ void lat4_store_warp(float y, uint8_t* rec, int kv_lora, int c, int lane) {
    uint8_t byte;
    const uint16_t d = lat4_group(y, lane, byte);
    const int blk = c >> 5;
    if (lane == 0) ((__half*) rec)[blk * 9] = __ushort_as_half(d);
    if (lane < 16) rec[blk * 18 + 2 + lane] = byte;
}

}  // namespace strata::kernels::glmf
