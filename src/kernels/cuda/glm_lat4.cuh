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

// there is no __ushort_as_ushort intrinsic: these are bit patterns already held as uint16_t
__host__ __device__ __forceinline__ unsigned short lat4_bits(unsigned short u) { return u; }

#include <cuda_fp16.h>

#include <cstdint>

// in namespace strata::kernels (NOT glmf/glmb): the kernels live in strata::kernels::glmf (glm_fast.cu) and
// strata::kernels::glmb (glm_batch.cu), and unqualified lookup walks the parent namespace, so both see these
// without a using-directive.
namespace strata::kernels {

__host__ __device__ __forceinline__ const uint8_t* lat4_rec(const uint16_t* lat, int kv, int cell) {
    // (the block stride is 18 bytes - 2 scale + 16 nibble - so 16 blocks are 288 at kv 512; lat4_rec_bytes in
    // glm_lat_types.hpp must agree or every row overlaps the next)
    return (const uint8_t*) lat + (size_t) (kv / 2 + kv / 16) * cell;
}

// value c of a record: block c >> 5, the scale one fp16 at the block's head, nibble byte (c % 32) & 15 holding
// element t = c % 32 in its LOW nibble (< 16) and element t + 16 in its HIGH one - what lat4_store_warp writes
// (lane t low, its __shfl_down 16 partner high) and what ggml's q4_0 does.  NOT t / 2 (consecutive pairs a byte):
// that is the other q4 packing and it decodes the wrong element for every t but 0, which is how a cache can look
// populated and still poison the scores.
__host__ __device__ __forceinline__ float lat4_f(const uint8_t* rec, int kv, int c) {
    const int blk = c >> 5, t = c & 31;
    const float d = __half2float(((const __half*) rec)[blk * 9]);
    const uint8_t b = rec[blk * 18 + 2 + (t & 15)];
    const int q = t < 16 ? (b & 0xF) : (b >> 4);
    return (float) (q - 8) * d;
}

// the four nibble bytes of word `w` of block `blk` of a record, as one uint32.  NOT a uint32 load: that word sits at
// rec + blk * 18 + 2 + 4 w, which is 2 (mod 4) aligned whenever blk is even - a 32-bit access there is a
// misaligned-address fault on the device (it reads fine on the host, which is why it survived until the q4_0 kernel
// branch first ran for real).  Two 16-bit loads are naturally aligned (2 | 2 + 4 w).
__host__ __device__ __forceinline__ uint32_t lat4_w32(const uint8_t* rec, int blk, int w) {
    const uint16_t* h = (const uint16_t*) (rec + blk * 18 + 2 + 4 * w);
    return (uint32_t) h[0] | ((uint32_t) h[1] << 16);
}

// values [4 c4, 4 c4 + 4) of a record as 4 FP16 (an FP16 row's uint2 there); 4 values sit in one block, and a
// 4-value group is entirely in the LOW nibbles of its bytes (groups 0..3 of a block) or entirely in the HIGH ones
// (groups 4..7) - see lat4_f.
__host__ __device__ __forceinline__ uint2 lat4_h4(const uint8_t* rec, int kv, int c4) {
    const int blk = c4 >> 3, g = c4 & 7;             // 8 groups of 4 values a block
    const float d = __half2float(((const __half*) rec)[blk * 9]);
    const uint32_t w = lat4_w32(rec, blk, g & 3);
    const bool hi = g >= 4;                          // elements 16..31 sit in the high nibbles
    __half v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint8_t b = (uint8_t) (w >> (8 * k));
        const int q = hi ? (b >> 4) : (b & 0xF);
        v[k] = __float2half((float) (q - 8) * d);
    }
    uint2 out;
    out.x = (uint32_t) (lat4_bits(((const uint16_t*) v)[0]) |
                        ((uint32_t) lat4_bits(((const uint16_t*) v)[1]) << 16));
    out.y = (uint32_t) (lat4_bits(((const uint16_t*) v)[2]) |
                        ((uint32_t) lat4_bits(((const uint16_t*) v)[3]) << 16));
    return out;
}

// values [8 c8, 8 c8 + 8) of a record as 8 FP16 (an FP16 row's uint4 there); 8 values sit in one block, in the low
// nibbles of bytes 0..7 (c8 & 3 == 0), of bytes 8..15 (1), or the high nibbles of those (2, 3)
__host__ __device__ __forceinline__ uint4 lat4_h8(const uint8_t* rec, int kv, int c8) {
    const int blk = c8 >> 2, g = c8 & 3;             // 4 groups of 8 values a block
    const float d = __half2float(((const __half*) rec)[blk * 9]);
    const bool hi = g >= 2;                          // elements 16..31 sit in the high nibbles
    __half v[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {                    // element k of the group: byte k, word (g & 1) * 2 + k / 4
        const uint32_t w = lat4_w32(rec, blk, (g & 1) * 2 + (k >> 2));
        const uint8_t b = (uint8_t) (w >> (8 * (k & 3)));
        const int q = hi ? (b >> 4) : (b & 0xF);
        v[k] = __float2half((float) (q - 8) * d);
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

}  // namespace strata::kernels
