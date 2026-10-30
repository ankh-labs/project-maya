#pragma once
// The batched rows' state (glm_rows.cu, glm_mtp_rows.cu): each part's row activations and one row's binding.
#include "glm_fast_state.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::core {

namespace glmrows {
constexpr int kQ8 = 36;   // bytes per q8_1 block
inline size_t up256(size_t b) { return (b + 255u) & ~(size_t) 255u; }
inline float* row_at(float* b, int64_t per, int i) { return b + (size_t) i * (size_t) per; }
inline uint8_t* rowq_at(uint8_t* b, int64_t n_in, int i) { return b + (size_t) i * (size_t) (n_in / 32) * kQ8; }
inline const float* image_row_in(const std::vector<int64_t>& pos, const std::vector<float>& rows, int64_t p,
                                 int n_embd) {
    const auto it = std::lower_bound(pos.begin(), pos.end(), p);
    if (it == pos.end() || *it != p) return nullptr;
    return rows.data() + (size_t) (it - pos.begin()) * (size_t) n_embd;
}
}  // namespace glmrows

// rows_max() rows of each activation, one row after the other (mv_nt reads and writes them so)
struct Glm5Model::RowsState {
    void* arena = nullptr;
    size_t bytes = 0;
    float *x = nullptr, *mixer = nullptr, *ffn = nullptr, *pre = nullptr, *post = nullptr, *comb = nullptr;
    uint8_t* xq = nullptr;
    float* proj[3] = {nullptr, nullptr, nullptr};
    float *fa = nullptr, *ga = nullptr, *beta = nullptr;
    uint8_t* gated_q = nullptr;
    float *qr_raw = nullptr, *qr = nullptr, *kv_raw = nullptr, *ik_raw = nullptr, *ig_raw = nullptr, *iw = nullptr;
    float *q = nullptr, *iq = nullptr;
    uint8_t *qr_q = nullptr, *attn_q = nullptr;
    float *rlog = nullptr, *sh_g = nullptr, *sh_u = nullptr, *dg = nullptr, *du = nullptr;
    float* plog = nullptr;                  // STRATA_GLM_ROWS_PRED_PF: each row's next-layer prediction logits
    unsigned long long* pf_src = nullptr;  // ... per-row prefetch staging (glm_fast.cu's pred block writes these)
    unsigned long long* pf_dst = nullptr;
    // ---- the cross-token layer pipeline's map (docs/LAYER_PIPELINE.md, increment 2)
    //
    // One cursor a row: which request it is, which layer it is about to run, and which stage of that layer it is in
    // (x = fetch the layer's experts, y = execute against them, z = route the NEXT layer's set).  The row's own
    // position and KV state live in its SeqCtx; this is only the progress through the layer stack, which is what lets
    // one request sit at layer L while another is at L-1.  Host-side on purpose: the scheduler reads and advances it
    // between launches, and nothing in a kernel needs it.
    // the slot's phase (DEFUSED_SCHEDULER_SPEC): one owner per phase, so the per-layer path needs no mutex.  A slot is
    // ROUTED when its request is published, FETCHING while its L+1 experts are in flight, READY when the transfer lane
    // has them, COMPUTING while the ordered compute stream runs its layer.
    enum SlotPhase : int { kEmpty = 0, kRouted = 1, kFetching = 2, kReady = 3, kComputing = 4 };
    struct RowCursor {
        int slot = -1;      // the row index in this step's set (-1: free)
        int il = 0;         // the layer it will run next
        int stage = 0;      // 0 x, 1 y, 2 z
        int lead = 0;       // layers ahead of the slowest row (the pipeline depth it is using)
        bool done = false;  // past l1_: nothing left for this token
        int phase = kEmpty; // SlotPhase: the owner writes it, everyone else reads it between launches
        // A layer the ladder answers Reorder for is NOT postponed by rewriting a round index: the row simply does not
        // advance `il` this round, and runs the same layer on the next one.  `deferred` is therefore "this row has
        // already used its one deferral since its last real run" and is cleared whenever a layer runs - not a fuse.
        int deferred = 0;
    };
    RowCursor cur[8];
    // WHO RECORDED THE CROSS-ROW EVENTS (2026-10-30).  ev_fill[il] and ev_layer[il] are single events per layer shared by
    // both rows, and each row waits on the other's record: "this layer's slots were filled by whichever row ran it first".
    // Two streams each waiting on an event the other records is a textbook event deadlock, and it is timing-dependent,
    // indifferent to BF16, and invisible at n = 1 - exactly this fault's shape.  The stall line names the recorder so the
    // suspicion becomes a statement: "row 1 waited on ev_fill[38] last recorded by row 0 on cs, generation stale".
    std::vector<int> ev_fill_row, ev_layer_row;   // the row that last recorded each, -1 if never

    int* pf_n = nullptr;
    uint8_t* dhq = nullptr;
    float* head_x = nullptr;
    uint8_t* head_xq = nullptr;
    float* logits = nullptr;
    int* tok = nullptr;
    // STRATA_GLM_ROWS_BATCH: each row's own copies of the token path's KDA scratch (F->conv, F->g1, F->g2) and hc
    // partials / barrier counter (F->part, F->counter)
    float* kconv[3] = {nullptr, nullptr, nullptr};
    float *kg1 = nullptr, *kg2 = nullptr, *hc_part = nullptr;
    uint8_t* hc_ctr = nullptr;
    int64_t hc_part_n = 0;
    // STRATA_GLM_ROWS_ATTN: the DSA attention of all the rows in one launch (gf::mla_attend_rows) - each row's q_abs and
    // ctx (n_head x kv_lora), its own selection and streamed slot cells (n_sel_max), and the launch's partials
    float *aq = nullptr, *actx = nullptr, *apart = nullptr;
    int *acells = nullptr, *aslot = nullptr;
    int apart_tokens = 0;
    float* ascore = nullptr;   // ... and each row's indexer scores (ascore_n a row), the rows' selection in one launch
    int64_t ascore_n = 0;
    int* tok_h = nullptr;   // pinned
    cudaEvent_t ev_done = nullptr;
};
// one row's sequence on this part: state, position, caches, KV stream and image row
struct Glm5Model::RowSeq {
    int seq = 0;
    float* state = nullptr;
    float *Rc = nullptr, *Ro = nullptr;   // the residual pair (the token path's convention, per row)
    int64_t p = 0;
    int ring = 0;
    const std::vector<int64_t> *kda_S = nullptr, *kda_conv = nullptr, *dsa_lat = nullptr, *dsa_ik = nullptr,
                               *dsa_ig = nullptr, *dsa_pool = nullptr;
    const float* img = nullptr;
    const KvStream* kv = nullptr;   // KV streaming: that sequence's stream (null: not streamed)
    // the residual pair's base: the sequence's arena start (rows_bind), a verify window's row its own pair
    // (rows_bind_win, STRATA_GLM_MULTI_MTP) - the embedding goes in there, the last layer leaves the residual there
    float* res = nullptr;
    float* kin = nullptr;   // a verify window's row: its conversation's KDA input records (null: none kept)
    int kin_t = 0;          // ... the row's index in them
};

}  // namespace strata::core
