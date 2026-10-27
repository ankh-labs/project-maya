// include/strata/kernels/glm_lat_types.hpp - the latent cache's record types (glm_fast.hpp lat8_rec_bytes,
// generalized): fp16 rows, INT8 records, q4_0 records.  Shared by the model (glm_model.hpp names the cache's
// type and sizes its arenas from it) and the kernels (glm_lat8.cuh holds the record convention; stage B of
// the q4_0 work reads it there).
#pragma once

#include <cstdint>

namespace strata::kernels::glmf {

/// INT8 records: per position kv_lora int8 codes, then one FP16 scale per 32 values (544 B a position and
/// layer at kv_lora 512, against the FP16 row's 1024).  glm_lat8.cuh holds the record convention.
__host__ __device__ constexpr inline int lat8_rec_bytes(int kv_lora) { return kv_lora + kv_lora / 16; }

/// q4_0 records: per 32 values an FP16 scale + 16 nibble bytes (ggml's convention, value = (q - 8) * d):
/// 288 B a position and layer at kv_lora 512.
__host__ __device__ constexpr inline int lat4_rec_bytes(int kv_lora) { return kv_lora / 2 + kv_lora / 32; }

enum class LatType : int { fp16 = 0, int8 = 1, q4_0 = 2 };

/// the bytes a position and layer occupy in the cache, per record type
__host__ __device__ constexpr inline int lat_rec_bytes(int kv_lora, LatType t) {
    return t == LatType::int8 ? lat8_rec_bytes(kv_lora)
         : t == LatType::q4_0 ? lat4_rec_bytes(kv_lora)
         : kv_lora * 2;
}

/// the name the slot identity's kv= field and the INFO line report
__host__ inline const char* lat_type_name(LatType t) {
    return t == LatType::int8 ? "int8" : t == LatType::q4_0 ? "q4_0" : "fp16";
}

}  // namespace strata::kernels::glmf
