// src/core/glm_rows.cu - several conversations decoded together (Glm5Model::multi_rows / multi_issue_rows).
//
// A decode token reads its part's weights once.  The one-GPU path takes ONE token of each of n conversations through
// the model together, reading the dense weights once for all; the split path does the same on each part per group.
// The attention projections, shared experts, dense FFN and head go through mv_nt, fetching each weight row once for the
// n activations and giving each activation the very bits mv() gives it alone.  What
// depends on a conversation's own history runs per row, through the token path's own kernels: the residual mixing
// (mHC), the KDA recurrence, the DSA indexer and attention against that conversation's caches, and the routed
// experts (the tiers serve the n routes one after another, as they serve the layers of a token).  So each row's
// token is the one the token path gives that conversation alone - as long as the tiers hold the same experts
// (greedy with the CPU lane off: the multi bench compares them).  STRATA_GLM_GROUP_EXPERTS=1 instead routes every row of
// the step first and runs each distinct routed expert once for all the rows that chose it (rows_moe_grouped, the verify
// window's batched MoE) - each row's output bit for bit the per-row kernels' for it.
//
// Each part's row activations live in its own arena (rows_alloc, at load, before the expert pool takes the rest of the
// card); the per-row scratch the token path's kernels use in between (the conv outputs, the gates, the DSA scores and
// cells, the experts' h) is the token path's own, used by one row after the other on that part's stream.
//
// STRATA_GLM_ROWS_BATCH=1: the residual mixing (mHC) and the KDA recurrence of all the rows in one launch each instead
// of one a row (gf::hc_rows, gf::kda_layers over the rows) - each row with its own scratch (rows_alloc) and the same
// device functions as the per-row kernels, so its arithmetic is meant to be theirs (glm_rows_kernels_test compares
// them bitwise).
//
// STRATA_GLM_ROWS_ATTN=1: a DSA layer's absorbed attention of all the rows in one launch instead of one a row: wk_b . q
// for every row at once (gf::mla_q_abs), each row's selection (and, streamed, its slot cells) into its own buffers, the
// split attention of all the rows over their own caches in one launch (gf::mla_attend_rows: a row's blocks on
// blockIdx.z, so its 64 blocks no longer leave half the card idle), then wv_b . ctx for every row at once
// (gf::mla_out) - the verify window's batched MLA, for rows of different conversations.  Each row's output is mla()'s
// bit for bit (glm_rows_attn_test compares them).
#include "glm_rows_state.hpp"

#include "strata/core/mtp_rows.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/sampler.hpp"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace strata::core {

namespace {
using glmrows::kQ8;
using glmrows::up256;
using glmrows::row_at;
using glmrows::rowq_at;
using glmrows::image_row_in;
bool rows_batch_on() {   // STRATA_GLM_ROWS_BATCH=1 (off by default)
    static const bool on = [] {
        const char* v = getenv("STRATA_GLM_ROWS_BATCH");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}
bool rows_small_on() {   // STRATA_GLM_ROWS_SMALL=1 (off by default)
    static const bool on = [] {
        const char* v = getenv("STRATA_GLM_ROWS_SMALL");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}
bool rows_attn_on() {   // STRATA_GLM_ROWS_ATTN=1 (off by default)
    // NOT made to follow the rotation: this flag is consulted OUTSIDE the rotor too (prefill and the fused path pick
    // per-row DSA buffers from it), and turning it on globally crashed those with illegal accesses in dsa_prep /
    // mla_q_abs / dsa_select_all / kv_q4 - while allocating the buffers and leaving the flag off crashed the same
    // kernels the other way.  Opening the gate needs the per-row DSA path to be the default EVERYWHERE, which is a
    // larger change than the rotor's own gate.
    static const bool on = [] {
        const char* v = getenv("STRATA_GLM_ROWS_ATTN");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}
}  // namespace

// 8 rows (one a conversation); with STRATA_GLM_MULTI_MTP a group's verify windows, multi_mtp_cap() rows
int Glm5Model::rows_max() { return multi_mtp_setting() > 0 ? multi_mtp_cap() : 8; }

int Glm5Model::split_rows_setting() {
    const char* v = getenv("STRATA_GLM_SPLIT_ROWS");
    return v != nullptr ? std::clamp((int) std::atof(v), 0, rows_max()) : 0;
}

bool Glm5Model::rows_alloc(std::string& err) {
    if (rows_ != nullptr) return true;
    const Glm5Geometry& g = g_;
    const size_t R = (size_t) rows_max();
    const int64_t E = g.n_embd, DI = g.d_inner(), FFs = (int64_t) g.n_ff_exp * g.n_shared;
    auto* S = new RowsState();
    uint8_t* base = nullptr;
    size_t at = 0;
    const auto take = [&](size_t row_bytes) -> uint8_t* {   // R rows of row_bytes (256-aligned as a block)
        const size_t off = at;
        at += up256(row_bytes * R);
        return base == nullptr ? nullptr : base + off;
    };
    const auto take_all = [&](size_t bytes) -> uint8_t* {   // once, not a row
        const size_t off = at;
        at += up256(bytes);
        return base == nullptr ? nullptr : base + off;
    };
    cudaSetDevice(dev_);   // (mla_rows_ok asks the current device)
    // the rotation runs two rows on two streams, which needs the per-row score/selection buffers even when
    // STRATA_GLM_ROWS_ATTN is unset - otherwise the rotor declines for want of them and the default carries nothing
    const bool attn = rows_attn_on() && gf::mla_rows_ok(g.kv_lora);
    const int attn_tokens = (int) std::min<size_t>(R, (size_t) gf::kMlaTokMax);
    int64_t attn_ctx = max_ctx_;   // (the longest context a sequence has: its completed pools' scores)
    for (const SeqCtx& c : seq_) attn_ctx = std::max(attn_ctx, c.max_ctx);
    const int64_t attn_pools = attn_ctx / g.idx_kpool + 1;
    const auto f32 = [&](int64_t n) { return (float*) take((size_t) n * sizeof(float)); };
    const auto q8 = [&](int64_t n) { return take((size_t) n / 32 * kQ8); };
    const auto carve = [&] {
        S->x = f32(E);
        S->xq = q8(E);
        S->mixer = f32(E);
        S->ffn = f32(E);
        S->pre = f32(8);
        S->post = f32(8);
        S->comb = f32(16);
        for (auto& p : S->proj) p = f32(DI);
        S->fa = f32(g.kda_head_dim);
        S->ga = f32(g.kda_head_dim);
        S->beta = f32(g.n_head);
        S->gated_q = q8(DI);
        S->qr_raw = f32(g.q_lora);
        S->qr = f32(g.q_lora);
        S->qr_q = q8(g.q_lora);
        S->kv_raw = f32(g.kv_lora);
        S->ik_raw = f32(g.idx_key);
        S->ig_raw = f32(g.idx_key);
        S->iw = f32(g.idx_heads);
        S->q = f32((int64_t) g.n_head * g.qk_nope);
        S->iq = f32((int64_t) g.idx_heads * g.idx_key);
        S->attn_q = q8((int64_t) g.n_head * g.v_head);
        S->rlog = f32(g.n_expert);
        S->plog = f32(g.n_expert);
        S->sh_g = f32(FFs);
        S->sh_u = f32(FFs);
        S->pf_src = (unsigned long long*) take(8 * sizeof(unsigned long long));
        S->pf_dst = (unsigned long long*) take(8 * sizeof(unsigned long long));
        S->pf_n = (int*) take(sizeof(int));
        S->dg = f32(g.n_ff_dense);
        S->du = f32(g.n_ff_dense);
        S->dhq = q8(g.n_ff_dense);
        S->head_x = f32(E);
        S->head_xq = q8(E);
        S->logits = f32(g.n_vocab);
        S->tok = (int*) take(sizeof(int));
        // the KDA mixer's per-row scratch: the batched path (STRATA_GLM_ROWS_BATCH) AND the layer pipeline need it - a
        // row run on its own stream must not share the token path's F->conv / F->g1 / F->g2, which is what it used when
        // n == 1 (two pipelined rows then wrote the same buffers: coherent-looking garbage, no fault)
        for (auto& p : S->kconv) p = f32(DI);
        S->kg1 = f32(DI);
        S->kg2 = f32(DI);
        if (rows_batch_on()) {
            S->hc_part_n = (4 * E / 512) * 26 + 64;   // (as the token path's F->part)
            S->hc_part = f32(S->hc_part_n);
            S->hc_ctr = take(64);
        }
        if (attn) {   // (STRATA_GLM_ROWS_ATTN's buffers: none with the setting off)
            S->aq = f32((int64_t) g.n_head * g.kv_lora);
            S->actx = f32((int64_t) g.n_head * g.kv_lora);
            S->acells = (int*) take((size_t) g.n_sel_max() * sizeof(int));
            S->aslot = (int*) take((size_t) g.n_sel_max() * sizeof(int));
            S->apart = (float*) take_all(gf::mla_rows_part_floats(g.n_head, g.kv_lora) * attn_tokens * sizeof(float));
            S->apart_tokens = attn_tokens;
            S->ascore_n = attn_pools;
            S->ascore = f32(attn_pools);
        }
    };
    carve();   // the sizes
    cudaSetDevice(dev_);
    if (cudaMalloc(&base, at) != cudaSuccess) {
        cudaGetLastError();
        delete S;
        err = "glm rows: the activations of " + std::to_string(R) + " rows (" + std::to_string(at >> 20) +
              " MB) do not fit on CUDA" + std::to_string(dev_);
        return false;
    }
    cudaMemset(base, 0, at);
    S->arena = base;
    S->bytes = at;
    at = 0;
    carve();   // the pointers
    if (attn)
        std::fprintf(stderr, "glm rows: CUDA%d the DSA attention of up to %d rows a launch (MAYA_GLM_ROWS_ATTN), %zu MB of "
                             "the rows' %zu MB\n", dev_, attn_tokens,
                     (size_t) ((2 * R * g.n_head * g.kv_lora * sizeof(float) + 2 * R * g.n_sel_max() * sizeof(int) +
                                gf::mla_rows_part_floats(g.n_head, g.kv_lora) * attn_tokens * sizeof(float) +
                                R * attn_pools * sizeof(float)) >> 20),
                     at >> 20);
    if (cudaHostAlloc((void**) &S->tok_h, R * sizeof(int), cudaHostAllocDefault) != cudaSuccess ||
        cudaEventCreateWithFlags(&S->ev_done, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        rows_ = S;
        rows_destroy();
        err = "glm rows: the token buffers did not allocate";
        return false;
    }
    rows_ = S;
    return true;
}

void Glm5Model::rows_destroy() {
    RowsState* S = rows_;
    if (S == nullptr) return;
    cudaSetDevice(dev_);
    if (S->arena) cudaFree(S->arena);
    if (S->tok_h) cudaFreeHost(S->tok_h);
    if (S->ev_done) cudaEventDestroy(S->ev_done);
    delete S;
    rows_ = nullptr;
}

bool Glm5Model::rows_bind(const int* seqs, int n, RowSeq* rw, std::string& err) {
    // Gate on the FLAG, not on rows_stream_: rows_bind runs BEFORE the rotor sets rows_stream_, so gating on the stream
    // made the check unreachable (it silently never ran).
    static const bool pipe_flag_on = glmfast::layer_pipe_mode() >= 2;   // MODE, not presence (unset means 2)
    if (pipe_flag_on) {
        // The ladder's precedence, checked on the first pipelined bind.  FastState is NESTED in Glm5Model, so its
        // members are not injected into this scope: the names must be qualified (FastState::PipeStage), which is what
        // the first two attempts at this assertion got wrong.
        static const bool ladder_ok = [] {
            using S = FastState;
            const S::PipeStage r[6] = {S::pipe_stage(true, true, true, true, true),
                                       S::pipe_stage(false, true, true, true, true),
                                       S::pipe_stage(false, false, true, true, true),
                                       S::pipe_stage(false, false, false, true, true),
                                       S::pipe_stage(false, false, false, false, true),
                                       S::pipe_stage(false, false, false, false, false)};
            const S::PipeStage w[6] = {S::kStageResident, S::kStageInFlight, S::kStageReorder,
                                       S::kStageCpuLane,  S::kStageDisk,   S::kStageReorder};
            for (int i = 0; i < 6; ++i)
                if (r[i] != w[i]) return false;
            return true;
        }();
        if (!ladder_ok) {
            err = "glm rows: the demotion ladder's precedence is not monotone";
            return false;
        }
    }
    // the pipeline's map: one cursor a row, starting at this part's first layer (docs/LAYER_PIPELINE.md, incr. 2)
    if (RowsState* RS = rows_) {
        for (int c = 0; c < 8; ++c) RS->cur[c] = RowsState::RowCursor{};
        for (int c = 0; c < n && c < 8; ++c) {
            RS->cur[c].slot = c;
            RS->cur[c].il = l0_;
            RS->cur[c].stage = 0;
            RS->cur[c].done = false;
        }
    }
    const Glm5Geometry& g = g_;
    const int E = g.n_embd;
    for (int i = 0; i < n; ++i) {
        const int sq = seqs[i];
        if (sq < 0 || sq >= seq_count()) {
            err = "glm rows: no sequence " + std::to_string(sq);
            return false;
        }
        for (int k = 0; k < i; ++k)
            if (seqs[k] == sq) {
                err = "glm rows: sequence " + std::to_string(sq) + " twice in one step";
                return false;
            }
        RowSeq& r = rw[i];
        r.seq = sq;
        int64_t max_ctx = 0;
        if (sq == cur_seq_) {
            r.state = state_;
            r.p = pos_;
            r.ring = ik_ring_;
            r.kda_S = &kda_S_;
            r.kda_conv = &kda_conv_;
            r.dsa_lat = &dsa_lat_;
            r.dsa_ik = &dsa_ik_;
            r.dsa_ig = &dsa_ig_;
            r.dsa_pool = &dsa_pool_;
            r.kv = kv_;
            r.img = image_row(r.p);
            max_ctx = max_ctx_;
        } else {
            const SeqCtx& c = seq_[(size_t) sq];
            r.state = c.state;
            r.p = c.pos;
            r.ring = c.ik_ring;
            r.kda_S = &c.kda_S;
            r.kda_conv = &c.kda_conv;
            r.dsa_lat = &c.dsa_lat;
            r.dsa_ik = &c.dsa_ik;
            r.dsa_ig = &c.dsa_ig;
            r.dsa_pool = &c.dsa_pool;
            r.kv = c.kv;
            r.img = image_row_in(c.img_pos, c.img_rows, r.p, E);
            max_ctx = c.max_ctx;
        }
        if (r.p + 1 >= max_ctx) {
            err = "glm rows: the context of sequence " + std::to_string(sq) + " is full";
            return false;
        }
        r.Rc = r.state;
        r.Ro = r.state + (int64_t) g.hc * E;
        r.res = r.state;
        r.kin = nullptr;
        r.kin_t = 0;
    }
    return true;
}

bool Glm5Model::rows_embed(const RowSeq* rw, const int32_t* tokens, int n, std::string& err) {
    FastState* F = fast_;
    const int E = g_.n_embd;
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm rows: no embedding dequantizer";
        return false;
    }
    cudaSetDevice(dev_);
    cudaStream_t s = F->cs;
    // ---- the embeddings: each row's token (or image row) into its own residual's 4 streams
    const size_t emb_row = ggml_row_size((ggml_type) pack_emb_type_, E);
    for (int i = 0; i < n; ++i) {
        SeqCtx& c = seq_[(size_t) rw[i].seq];
        // its own pinned row: the step before has completed, so no copy still reads it
        tt->to_float(pack_emb_src_ + (size_t) tokens[i] * emb_row, c.emb_h, E);
        if (rw[i].img != nullptr) std::memcpy(c.emb_h, rw[i].img, (size_t) E * sizeof(float));
        cudaMemcpyAsync(F->emb, c.emb_h, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s);
        gf::embed_streams(F->emb, rw[i].res, E, s);
    }
    return true;
}

bool Glm5Model::rows_layers(RowSeq* rw, int n, std::string& err, int il_from, int il_to) {
    // il_from/il_to default (-1) to the whole part.  A caller may drive ONE layer at a time by passing
    // [il, il+1): the loop below is the only place the bounds are used, and the Rc/Ro swap keys on the
    // global l0_ (so the per-layer alternation stays correct across separate calls).  This is the seam a
    // cross-token layer pipeline needs (docs/LAYER_PIPELINE.md, increment 1) and it is arithmetic-neutral.
    const int ilf = il_from < 0 ? l0_ : std::max(l0_, il_from);
    const int ilt = il_to < 0 ? l1_ : std::min(l1_, il_to);
    FastState* F = fast_;
    RowsState* S = rows_;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd, DI = g.d_inner();
    cudaSetDevice(dev_);
    cudaStream_t s = rows_stream_ != nullptr ? rows_stream_ : F->cs;   // (a pipeline driver gives a row its own)
    auto& rows = row_at;
    auto& rows_q = rowq_at;
    // the row's OWN slice of the 8-wide per-row buffers.  row_base_ is 0 for an ordinary call, so this is an identity
    // unless a driver runs one row per call - in which case the row must take the slice it owns, not slice 0, or two
    // concurrent rows write the same activation (docs/LAYER_PIPELINE.md).
    const int rb = row_base_;   // this row's block of every n-row buffer below
    const auto R = [&](int k) { return row_base_ + k; };
    const auto mv = [&](gf::MvJob* j, int nj, const char* what) -> bool {
        if (gf::mv_nt(j, nj, n, s)) return true;
        err = std::string("glm rows: ") + what;
        return false;
    };
    // a verify window's rows of one conversation advance its recurrent states one after the other: the per-row
    // kernels in row order then (rows_bind_win), never one launch for all of them
    bool dup = false, keep_in = false;
    for (int i = 0; i < n; ++i) {
        keep_in = keep_in || rw[i].kin != nullptr;
        for (int k = 0; k < i && !dup; ++k) dup = rw[k].seq == rw[i].seq;
    }
    const bool batch = rows_batch_on() && n > 1;
    // (kda_layers is built for head_dim 128, as the window commit's replay: else the KDA one row at a time)
    const bool kda_batch = batch && !dup && g.kda_head_dim == 128;
    // a pipeline call runs ONE row, so `batch` is false for it, but it still needs per-row buffers rather than the token
    // path's single set (docs/LAYER_PIPELINE.md)
    const bool per_row_bufs = batch || rows_per_row_;
    gf::HcArgs hb[gf::kMaxMvNt];
    for (int il = ilf; il < ilt; ++il) {
        const auto& Ly = F->L[(size_t) il];
        gf::HcArgs h;
        h.norm_eps = g.norm_eps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = E;
        h.part = F->part;
        h.counter = F->counter;
        const auto hc_row = [&](int i, const float* block_out) {
            h.block_out = block_out;
            h.R_old = rw[i].Rc;
            h.R_new = rw[i].Ro;
            h.post_in = rows(S->post, 8, R(i));
            h.comb_in = rows(S->comb, 16, R(i));
            h.pre = rows(S->pre, 8, R(i));
            h.post = rows(S->post, 8, R(i));
            h.comb = rows(S->comb, 16, R(i));
            h.x = rows(S->x, E, R(i));
            h.xq = rows_q(S->xq, E, i);
            if (!batch) {
                gf::hc(h, s);
                return;
            }
            hb[i] = h;   // (launched for all the rows at once: hc_rows)
            hb[i].part = rows(S->hc_part, S->hc_part_n, R(i));
            hb[i].counter = (unsigned int*) (S->hc_ctr + (size_t) R(i) * 64);   // (take(64): 64 bytes a row)
        };
        // ---- the attention-side read (fused with the previous FFN's write), per row
        h.w_fn = Ly.hc_attn_fn;
        h.w_scale = Ly.hc_attn_scale;
        h.w_base = Ly.hc_attn_base;
        h.norm_w = Ly.attn_norm;
        for (int i = 0; i < n; ++i) {
            hc_row(i, il == l0_ ? nullptr : rows(S->ffn, E, R(i)));
            if (il != l0_) std::swap(rw[i].Rc, rw[i].Ro);
            if (rows_ != nullptr && row_base_ + i < 8) rows_->cur[row_base_ + i].il = il + 1;   // (the map: this row is one layer on)
        }
        if (batch) gf::hc_rows(hb, n, s);

        if (Ly.recr) {
            gf::MvJob j[6];
            j[0] = {Ly.q.q, rows_q(S->xq, E, rb), rows(S->x, E, rb), rows(S->proj[0], DI, rb), nullptr, 1.0f, Ly.q.type, E, DI};
            j[1] = {Ly.k.q, rows_q(S->xq, E, rb), rows(S->x, E, rb), rows(S->proj[1], DI, rb), nullptr, 1.0f, Ly.k.type, E, DI};
            j[2] = {Ly.v.q, rows_q(S->xq, E, rb), rows(S->x, E, rb), rows(S->proj[2], DI, rb), nullptr, 1.0f, Ly.v.type, E, DI};
            j[3] = {Ly.f_a, nullptr, rows(S->x, E, rb), rows(S->fa, g.kda_head_dim, rb), nullptr, 1.0f, gf::kTypeBF16, E, g.kda_head_dim};
            j[4] = {Ly.g_a, nullptr, rows(S->x, E, rb), rows(S->ga, g.kda_head_dim, rb), nullptr, 1.0f, gf::kTypeBF16, E, g.kda_head_dim};
            j[5] = {Ly.beta, nullptr, rows(S->x, E, rb), rows(S->beta, g.n_head, rb), nullptr, 1.0f, gf::kTypeBF16, E, g.n_head};
            if (!mv(j, 6, "kda projections")) return false;
            if (keep_in) {   // (STRATA_GLM_MULTI_MTP: the window rows' inputs, for the commit's replay)
                gf::KdaGatherArgs ga;
                ga.src[0] = S->proj[0];
                ga.src[1] = S->proj[1];
                ga.src[2] = S->proj[2];
                ga.src[3] = S->fa;
                ga.src[4] = S->ga;
                ga.src[5] = S->beta;
                ga.n_rows = n;
                ga.tmax = mtpw_window();
                ga.di = DI;
                ga.hd = g.kda_head_dim;
                ga.nh = g.n_head;
                const size_t off = mtpw_kin_offset(il);
                for (int i = 0; i < n; ++i) {
                    ga.dst[i] = rw[i].kin != nullptr ? rw[i].kin + off : nullptr;
                    ga.t[i] = rw[i].kin_t;
                }
                gf::kda_gather(ga, s);
            }
            gf::KdaPrepArgs kp[gf::kMaxMvNt];
            gf::KdaRecArgs kr[gf::kMaxMvNt];
            for (int i = 0; i < n; ++i) {
                gf::KdaPrepArgs k;
                for (int c = 0; c < 3; ++c) {
                    k.proj[c] = rows(S->proj[c], DI, R(i));
                    k.conv_w[c] = Ly.conv[c];
                    k.out[c] = per_row_bufs ? rows(S->kconv[c], DI, R(i)) : F->conv[c];
                }
                k.conv_state = rw[i].state + (*rw[i].kda_conv)[(size_t) il];
                k.fa = rows(S->fa, g.kda_head_dim, R(i));
                k.ga = rows(S->ga, g.kda_head_dim, R(i));
                k.f_b = Ly.f_b;
                k.g_b = Ly.g_b;
                k.dt_bias = Ly.dt_bias;
                k.ssm_a = Ly.ssm_a;
                k.lower_bound = g.kda_lb;
                k.g1 = per_row_bufs ? rows(S->kg1, DI, R(i)) : F->g1;
                k.g2 = per_row_bufs ? rows(S->kg2, DI, R(i)) : F->g2;
                k.n_head = g.n_head;
                k.head_dim = g.kda_head_dim;
                k.d_conv = g.d_conv;
                if (kda_batch) {   // (both kernels for all the rows below: kda_layers)
                    kp[i] = k;
                    kr[i].q = k.out[0];
                    kr[i].k = k.out[1];
                    kr[i].v = k.out[2];
                    kr[i].g1 = k.g1;
                    kr[i].beta_raw = rows(S->beta, g.n_head, R(i));
                    kr[i].state = rw[i].state + (*rw[i].kda_S)[(size_t) il];
                    kr[i].g2 = k.g2;
                    kr[i].norm_w = Ly.ssm_norm;
                    kr[i].out = rows_q(S->gated_q, DI, R(i));
                    continue;
                }
                gf::kda_prep(k, s);
                gf::kda_rec(per_row_bufs ? rows(S->kconv[0], DI, R(i)) : F->conv[0],
                            per_row_bufs ? rows(S->kconv[1], DI, R(i)) : F->conv[1],
                            per_row_bufs ? rows(S->kconv[2], DI, R(i)) : F->conv[2],
                            per_row_bufs ? rows(S->kg1, DI, R(i)) : F->g1, rows(S->beta, g.n_head, R(i)),
                            rw[i].state + (*rw[i].kda_S)[(size_t) il],
                            per_row_bufs ? rows(S->kg2, DI, R(i)) : F->g2, Ly.ssm_norm, g.norm_eps, g.n_head,
                            g.kda_head_dim, rows_q(S->gated_q, DI, R(i)), s);
            }
            for (int i0 = 0; kda_batch && i0 < n; i0 += gf::kKdaLayers)
                gf::kda_layers(kp + i0, kr + i0, std::min(gf::kKdaLayers, n - i0), g.norm_eps, s);
            gf::MvJob o = {Ly.out.q, rows_q(S->gated_q, DI, rb), nullptr, rows(S->mixer, E, rb), nullptr, 1.0f, Ly.out.type, DI, E};
            if (!mv(&o, 1, "kda output")) return false;
        } else {
            if (!rows_dsa(il, rw, n, err)) return false;
        }

        // ---- the FFN-side read (fused with the mixer's write), per row
        h.w_fn = Ly.hc_ffn_fn;
        h.w_scale = Ly.hc_ffn_scale;
        h.w_base = Ly.hc_ffn_base;
        h.norm_w = Ly.ffn_norm;
        for (int i = 0; i < n; ++i) {
            hc_row(i, rows(S->mixer, E, R(i)));
            std::swap(rw[i].Rc, rw[i].Ro);
        }
        if (batch) gf::hc_rows(hb, n, s);

        if (rows_per_row_)   // stage 1: the mixer's output (attention done, FFN not yet)
            gf::nonfinite_probe(rows(S->mixer, E, R(0)), (int) E, (il << 8) | (1 << 4) | rb, s);
        if (rows_per_row_) gf::pipe_probe(rows(S->mixer, E, R(0)), (int) E, rb * 2048 + il * 4 + 1, s);
        if (!Ly.moe) {
            gf::MvJob j[2];
            j[0] = {Ly.ffn_gate.q, rows_q(S->xq, E, rb), rows(S->x, E, rb), rows(S->dg, g.n_ff_dense, rb), nullptr, 1.0f, Ly.ffn_gate.type, E, g.n_ff_dense};
            j[1] = {Ly.ffn_up.q, rows_q(S->xq, E, rb), rows(S->x, E, rb), rows(S->du, g.n_ff_dense, rb), nullptr, 1.0f, Ly.ffn_up.type, E, g.n_ff_dense};
            if (!mv(j, 2, "dense ffn")) return false;
            for (int i = 0; i < n; ++i)
                gf::swiglu_q8(rows(S->dg, g.n_ff_dense, R(i)), rows(S->du, g.n_ff_dense, R(i)), g.swiglu_shexp,
                              g.n_ff_dense, rows_q(S->dhq, g.n_ff_dense, R(i)), s);
            gf::MvJob dn = {Ly.ffn_down.q, rows_q(S->dhq, g.n_ff_dense, rb), nullptr, rows(S->ffn, E, rb), nullptr, 1.0f, Ly.ffn_down.type, g.n_ff_dense,
                            E};
            if (!mv(&dn, 1, "dense ffn down")) return false;
        } else {
            if (!rows_moe(il, rw, n, err)) return false;
        }
        if (rows_per_row_)   // stage 2: this layer's FFN output (still pre-residual)
            gf::nonfinite_probe(rows(S->ffn, E, R(0)), (int) E, (il << 8) | (2 << 4) | rb, s);
        if (rows_per_row_) gf::pipe_probe(rows(S->ffn, E, R(0)), (int) E, rb * 2048 + il * 4 + 2, s);
    }
    // the last layer's write half, per row; the residual back at the arena's start (the token path's convention)
    for (int i = 0; i < n; ++i) {
        gf::hc_post(rows(S->ffn, E, R(i)), rw[i].Rc, rows(S->post, 8, R(i)), rows(S->comb, 16, R(i)), E, rw[i].Ro, s);
        if (rw[i].Ro != rw[i].res)
            cudaMemcpyAsync(rw[i].res, rw[i].Ro, (size_t) g.hc * E * sizeof(float), cudaMemcpyDeviceToDevice, s);
    }
    return true;
}

// a DSA layer's mixer for the rows (S->x / S->xq in, S->mixer out): the projections for every row at once, each row's
// caches, selection and attention through the token path's kernels against its own sequence (rw[i])
bool Glm5Model::rows_dsa(int il, RowSeq* rw, int n, std::string& err) {
    FastState* F = fast_;
    RowsState* S = rows_;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd, FFs = g.n_ff_exp * g.n_shared;
    (void) FFs;
    cudaStream_t s = rows_stream_ != nullptr ? rows_stream_ : F->cs;   // (a pipeline driver gives a row its own)
    auto& rows = row_at;
    auto& rows_q = rowq_at;
    // the row's OWN slice of the 8-wide per-row buffers.  row_base_ is 0 for an ordinary call, so this is an identity
    // unless a driver runs one row per call - in which case the row must take the slice it owns, not slice 0, or two
    // concurrent rows write the same activation (docs/LAYER_PIPELINE.md).
    const auto R = [&](int k) { return row_base_ + k; };
    const int rb = row_base_;   // this row's block of aq / actx (ad floats each)
    const auto mv = [&](gf::MvJob* j, int nj, const char* what) -> bool {
        if (gf::mv_nt(j, nj, n, s)) return true;
        err = std::string("glm rows: ") + what;
        return false;
    };
    const auto& Ly = F->L[(size_t) il];
    const float prescale = 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads));
    gf::MvJob j[5];
    j[0] = {Ly.q_a.q, rows_q(S->xq, E, rb), rows(S->x, E, rb), rows(S->qr_raw, g.q_lora, rb), nullptr, 1.0f, Ly.q_a.type, E, g.q_lora};
    j[1] = {Ly.kv_a.q, rows_q(S->xq, E, rb), rows(S->x, E, rb), rows(S->kv_raw, g.kv_lora, rb), nullptr, 1.0f, Ly.kv_a.type, E, g.kv_lora};
    j[2] = {Ly.idx_k, nullptr, rows(S->x, E, rb), rows(S->ik_raw, g.idx_key, rb), nullptr, 1.0f, gf::kTypeBF16, E, g.idx_key};
    j[3] = {Ly.idx_gate, nullptr, rows(S->x, E, rb), rows(S->ig_raw, g.idx_key, rb), nullptr, 1.0f, gf::kTypeBF16, E, g.idx_key};
    j[4] = {Ly.idx_proj, nullptr, rows(S->x, E, rb), rows(S->iw, g.idx_heads, rb), nullptr, prescale, gf::kTypeBF16, E, g.idx_heads};
    if (!mv(j, 5, "dsa projections")) return false;
    for (int i = 0; i < n; ++i) {
        gf::DsaPrepArgs d;
        d.qr_raw = rows(S->qr_raw, g.q_lora, R(i));
        d.q_a_norm = Ly.q_a_norm;
        d.qr = rows(S->qr, g.q_lora, R(i));
        d.qr_q = rows_q(S->qr_q, g.q_lora, i);
        d.q_lora = g.q_lora;
        d.kv_raw = rows(S->kv_raw, g.kv_lora, R(i));
        d.kv_norm = Ly.kv_a_norm;
        kv_prep_targets(rw[i].kv, il, (uint16_t*) (rw[i].state + (*rw[i].dsa_lat)[(size_t) il]), d);
        d.kv_lora = g.kv_lora;
        d.lat_q8 = lat_q8_;
        d.ik_raw = rows(S->ik_raw, g.idx_key, R(i));
        d.k_norm_w = Ly.k_norm_w;
        d.k_norm_b = Ly.k_norm_b;
        d.ik_cache = rw[i].state + (*rw[i].dsa_ik)[(size_t) il];
        d.ig_raw = rows(S->ig_raw, g.idx_key, R(i));
        d.ig_cache = rw[i].state + (*rw[i].dsa_ig)[(size_t) il];
        d.ape = Ly.ape;
        d.pooled = rw[i].state + (*rw[i].dsa_pool)[(size_t) il];
        d.idx_key = g.idx_key;
        d.kpool = g.idx_kpool;
        d.ring = rw[i].ring;
        d.p = (int) rw[i].p;
        d.eps = g.norm_eps;
        gf::dsa_prep(d, s);
    }
    gf::MvJob j2[2];
    j2[0] = {Ly.q_b.q, rows_q(S->qr_q, g.q_lora, rb), rows(S->qr, g.q_lora, rb), rows(S->q, g.n_head * g.qk_nope, rb), nullptr, 1.0f, Ly.q_b.type, g.q_lora, g.n_head * g.qk_nope};
    j2[1] = Ly.idx_q_b_x.q != nullptr   // (an EXL3 pack quantizes the indexer's q_b)
                ? gf::MvJob{Ly.idx_q_b_x.q, rows_q(S->qr_q, g.q_lora, rb), rows(S->qr, g.q_lora, rb), rows(S->iq, g.idx_heads * (int64_t) g.idx_key, rb), nullptr, 1.0f, Ly.idx_q_b_x.type, g.q_lora,
                            g.idx_heads * g.idx_key}
                : gf::MvJob{Ly.idx_q_b, nullptr, rows(S->qr, g.q_lora, rb), rows(S->iq, g.idx_heads * (int64_t) g.idx_key, rb), nullptr, 1.0f, gf::kTypeBF16, g.q_lora,
                            g.idx_heads * g.idx_key};
    if (!mv(j2, 2, "dsa q")) return false;
    // STRATA_GLM_ROWS_ATTN: the attention of every row in one launch (each row's selection into its own buffers).  A
    // verify window's rows of one conversation (consecutive rows) on a streamed layer share its slot pool: one resolve
    // over all their selections then (a resolve never evicts a block it names), not one a row - a row's resolve could
    // evict a block the row before it still has to read.  Rows of one conversation that are not consecutive: one by one
    const int64_t qd = (int64_t) g.n_head * g.qk_nope, ad = (int64_t) g.n_head * g.kv_lora, nsm = g.n_sel_max();
    // n > 1 OR a per-row call (row_base_): a pipeline row runs alone, so n == 1, and without this it took the token
    // path's shared score / cells buffers instead of its own
    bool attn = rows_attn_on() && (n > 1 || rows_per_row_) && S->apart != nullptr && g.v_head % 32 == 0 &&
                gf::mla_rows_ok(g.kv_lora);
    int run0[gf::kMaxMvNt];   // the first row of each row's run of consecutive rows of its conversation
    for (int i = 0; i < n; ++i) {
        run0[i] = i > 0 && rw[i - 1].seq == rw[i].seq ? run0[i - 1] : i;
        for (int k = 0; k < run0[i] && attn; ++k)
            if (rw[k].seq == rw[i].seq) attn = false;
    }
    if (attn)   // (a run's selections are resolved as one list: the cells past a row's own count must read as padding)
        for (int i = 0; i < n; ++i)
            if (run0[i] == i && i + 1 < n && run0[i + 1] == i && kv_streamed(rw[i].kv, il)) {
                int len = 1;
                while (i + len < n && run0[i + len] == i) ++len;
                cudaMemsetAsync(S->acells + (size_t) R(i) * nsm, 0xff, (size_t) len * nsm * sizeof(int), s);
            }
    const uint16_t* a_lat[gf::kMaxMvNt];
    const int* a_cells[gf::kMaxMvNt];
    int a_nsel[gf::kMaxMvNt];
    if (attn) gf::mla_q_abs(S->q, qd, Ly.k_b, g.n_head, g.qk_nope, g.kv_lora, n, S->aq + rb * ad, ad, s);
    gf::DsaSelectRows sr;   // (with the rows' attention: the rows' selections in one launch, then each one resolved)
    int n_sr = 0;
    const bool small = attn && rows_small_on();   // (STRATA_GLM_ROWS_SMALL: the rows' scores in one launch too)
    gf::DsaScoreRows sc;
    int n_sc = 0;
    for (int i = 0; i < n; ++i) {
        const int64_t p = rw[i].p;
        const int pool_done = (int) ((p + 1) / g.idx_kpool);
        const float* pooled = rw[i].state + (*rw[i].dsa_pool)[(size_t) il];
        const bool all = pool_done <= g.top_pools_max() && gf::dsa_select_all_on();   // (as fast_dsa)
        float* const score = attn ? S->ascore + (size_t) R(i) * S->ascore_n : F->score;
        if (pool_done > 0 && !all && small) {
            sc.iq[n_sc] = rows(S->iq, (int64_t) g.idx_heads * g.idx_key, R(i));
            sc.pooled[n_sc] = pooled;
            sc.iw[n_sc] = rows(S->iw, g.idx_heads, R(i));
            sc.score[n_sc] = score;
            sc.n_pools[n_sc++] = pool_done;
        } else if (pool_done > 0 && !all) {
            gf::dsa_score(rows(S->iq, (int64_t) g.idx_heads * g.idx_key, R(i)), pooled,
                          rows(S->iw, g.idx_heads, R(i)), g.idx_key, g.idx_heads, pool_done, score, s, true);   // (a decode row)
        }
        const int top_pools = std::min(g.top_pools_max(), pool_done);
        const int n_sel = g.idx_kpool * top_pools + (g.idx_select_tail ? g.idx_kpool - 1 : 0);
        int* const sel = attn ? S->acells + (size_t) R(i) * nsm : F->cells;
        uint16_t* const lat = (uint16_t*) (rw[i].state + (*rw[i].dsa_lat)[(size_t) il]);
        if (attn) {
            if (all) {
                gf::dsa_select_all(pool_done, g.idx_kpool, n_sel, (int) p, sel, s);
            } else {
                sr.score[n_sr] = score;
                sr.cells[n_sr] = sel;
                sr.n_vis[n_sr] = pool_done;
                sr.top_pools[n_sr] = top_pools;
                sr.n_sel[n_sr] = n_sel;
                sr.pos[n_sr] = (int) p;
                ++n_sr;
            }
            a_lat[i] = lat;
            a_nsel[i] = n_sel;
            continue;
        }
        if (all) gf::dsa_select_all(pool_done, g.idx_kpool, n_sel, (int) p, sel, s);
        else gf::dsa_select(F->score, pool_done, g.idx_kpool, top_pools, n_sel, (int) p, sel, s);
        const int* cells = kv_attend_cells(rw[i].kv, il, lat, F->cells, n_sel, F->slot_cells, s);
        gf::mla(rows(S->q, qd, R(i)), Ly.k_b, Ly.v_b, lat, cells, n_sel, g.n_head,
                g.qk_nope, g.kv_lora, g.v_head, rows_q(S->attn_q, (int64_t) g.n_head * g.v_head, R(i)), s,
                lat_q8_, nullptr, true);   // (a decode row: the token path's kernels)
    }
    if (attn) {
        if (n_sc > 0) gf::dsa_score_rows(sc, n_sc, g.idx_key, g.idx_heads, s);
        gf::dsa_select_rows(sr, n_sr, g.idx_kpool, s);
        // KV streaming: each run's blocks resident, its rows' selections in slot rows (resident: the cells themselves);
        // STRATA_GLM_ROWS_SMALL: the runs' resolves in one launch (each run its own sequence's map)
        gf::KvResolveRows kr;
        int n_kr = 0, kr_row = 0, kr_page = 0;
        for (int i = 0; i < n;) {
            int len = 1;
            while (i + len < n && run0[i + len] == i) ++len;
            const int cnt = (int) ((len - 1) * nsm) + a_nsel[i + len - 1];
            const KvStream* K = rw[i].kv;
            const bool batch_it = small && kv_streamed(K, il) && n_kr < gf::kKvResolveRowsMax &&
                                  (n_kr == 0 || (K->row_bytes == kr_row && K->page == kr_page));
            if (batch_it) {
                kr.m[n_kr] = K->map[(size_t) il];
                kr.host[n_kr] = K->host[(size_t) il];
                kr.slots[n_kr] = (uint8_t*) a_lat[i];
                kr.cells[n_kr] = S->acells + (size_t) R(i) * nsm;
                kr.slot_cells[n_kr] = S->aslot + (size_t) R(i) * nsm;
                kr.n_sel[n_kr++] = cnt;
                kr_row = K->row_bytes;
                kr_page = K->page;
                for (int k = 0; k < len; ++k) a_cells[i + k] = S->aslot + (size_t) (i + k) * nsm;
            } else {
                const int* c = kv_attend_cells(K, il, const_cast<uint16_t*>(a_lat[i]), S->acells + (size_t) R(i) * nsm, cnt,
                                               S->aslot + (size_t) R(i) * nsm, s);
                for (int k = 0; k < len; ++k) a_cells[i + k] = c + (size_t) k * nsm;
            }
            i += len;
        }
        if (n_kr > 0) gf::glm_kv_resolve_rows(kr, n_kr, kr_row, kr_page, s);
        if (!gf::mla_attend_rows(S->aq + rb * ad, ad, a_lat, a_cells, a_nsel, n, g.n_head, g.qk_nope, g.kv_lora, S->actx, ad,
                                 S->apart + (size_t) rb * gf::mla_rows_part_floats(g.n_head, g.kv_lora),
                                 S->apart_tokens, s, lat_q8_)) {
            err = "glm rows: the rows' attention is not available here";   // (mla_rows_ok said it was)
            return false;
        }
        // per-row slices for BOTH buffers (this commit's content) on upstream's mla_out signature (no lat_q8_).
        gf::mla_out(S->actx + rb * ad, ad, Ly.v_b, g.n_head, g.kv_lora, g.v_head, n,
                    rows_q(S->attn_q, (int64_t) g.n_head * g.v_head, rb),
                    (int64_t) ((int64_t) g.n_head * g.v_head / 32) * kQ8, s);
    }
    gf::MvJob o = {Ly.out.q, rows_q(S->attn_q, (int64_t) g.n_head * g.v_head, rb), nullptr, rows(S->mixer, E, rb), nullptr, 1.0f, Ly.out.type, g.n_head * g.v_head,
                   E};
    if (!mv(&o, 1, "dsa output")) return false;
    return true;
}

// STRATA_GLM_ROWS_ROUTE_LOG=1 (the cross-slot shared-expert count's input): both rows' routed experts, one line a
// step.  The ids go to a pinned double buffer on the part's stream; the previous step's line prints here once its
// copies are done (cudaEventQuery - nothing in the decode syncs).  Part 0 only: the state is process-wide and the
// split's parts run on their own threads.
namespace {
struct RowsRouteLog {
    int* h = nullptr;               // [2 rounds][2 rows][8 ids]
    cudaEvent_t ev[2] = {};          // the round's copies are done
    int round = 0, row = 0;
};
RowsRouteLog& rows_route_log_state() { static RowsRouteLog st; return st; }
bool rows_route_log_on() {
    static const bool on = [] {
        const char* v = getenv("STRATA_GLM_ROWS_ROUTE_LOG");
        return v != nullptr && v[0] != '\0' && std::atoi(v) != 0;
    }();
    return on;
}
// STRATA_GLM_ROWS_PRED_PF=<n> (default off): predicted cross-layer prefetch on the rows path - after a layer's
// routes, the next layer's predicted experts (its router weight on the current x) that are not in VRAM claim that
// layer's spares and are copied on the side stream while this layer computes.  n caps the prefetches per row per
// layer (0 = off); STRATA_GLM_PREFETCH_RANK still caps the ranks.  Mutually exclusive with STRATA_GLM_ROWS_PF
// (they would share ev_pf_prev and the tier tables under both arms).
int rows_pred_pf_n() {
    static const int n = [] {
        const char* v = getenv("STRATA_GLM_ROWS_PRED_PF");
        return v != nullptr && v[0] != '\0' ? std::atoi(v) : 0;
    }();
    return n;
}void rows_route_log_capture(cudaStream_t s, const int* dev_ids, int k) {
    RowsRouteLog& st = rows_route_log_state();
    if (st.h == nullptr) {
        cudaHostAlloc(&st.h, 2 * 2 * 8 * sizeof(int), cudaHostAllocDefault);
        for (int e = 0; e < 2; ++e) cudaEventCreateWithFlags(&st.ev[e], cudaEventDisableTiming);
    }
    if (st.row == 0 && cudaEventQuery(st.ev[st.round ^ 1]) == cudaSuccess) {
        const int* r0 = st.h + (st.round ^ 1) * 16;
        const int* r1 = r0 + 8;
        int shared = 0;
        for (int a = 0; a < k; ++a) {
            for (int b = 0; b < k; ++b)
                if (r1[b] == r0[a]) { ++shared; break; }
        }
        std::fprintf(stderr,
                     "glm rows route: row0 [%d %d %d %d %d %d %d %d] row1 [%d %d %d %d %d %d %d %d] shared %d\n",
                     r0[0], r0[1], r0[2], r0[3], r0[4], r0[5], r0[6], r0[7],
                     r1[0], r1[1], r1[2], r1[3], r1[4], r1[5], r1[6], r1[7], shared);
    }
    cudaMemcpyAsync(st.h + st.round * 16 + (size_t) st.row * 8, dev_ids, 8 * sizeof(int),
                    cudaMemcpyDeviceToHost, s);
    if (++st.row == 2) {
        cudaEventRecord(st.ev[st.round], s);
        st.round ^= 1;
        st.row = 0;
    }
}
}  // namespace

// a MoE layer's FFN for the rows (S->x / S->xq in, S->ffn out)
bool Glm5Model::rows_moe(int il, RowSeq* rw, int n, std::string& err) {
    FastState* F = fast_;
    RowsState* S = rows_;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd, FFs = g.n_ff_exp * g.n_shared;
    (void) FFs;
    cudaStream_t s = rows_stream_ != nullptr ? rows_stream_ : F->cs;   // (a pipeline driver gives a row its own)
    auto& rows = row_at;
    auto& rows_q = rowq_at;
    // the row's OWN slice of the 8-wide per-row buffers.  row_base_ is 0 for an ordinary call, so this is an identity
    // unless a driver runs one row per call - in which case the row must take the slice it owns, not slice 0, or two
    // concurrent rows write the same activation (docs/LAYER_PIPELINE.md).
    const auto R = [&](int k) { return row_base_ + k; };
    const auto mv = [&](gf::MvJob* j, int nj, const char* what) -> bool {
        if (gf::mv_nt(j, nj, n, s)) return true;
        err = std::string("glm rows: ") + what;
        return false;
    };
    const auto& Ly = F->L[(size_t) il];
    (void) rw;
    const float* no_bias[gf::kAhead] = {nullptr, nullptr, nullptr, nullptr};
    bool pf_pending = false;   // (no prefetch in a batched step)
    // the router and the shared expert's gate / up for every row, then each row's routed experts (and the
    // shared expert's down, which rides their launch) through the token path's own code
    gf::MvJob j[3];
    // these three GEMVs run over n rows from ONE base: a driver running a single row per call must start at that row's
    // slice (row_base_ 0 leaves them exactly as they were)
    const int rb = row_base_;
    j[0] = {Ly.router, nullptr, rows(S->x, E, rb), rows(S->rlog, g.n_expert, rb), nullptr, 1.0f, gf::kTypeBF16, E,
            g.n_expert};
    j[1] = {Ly.sh_gate.q, rows_q(S->xq, E, rb), rows(S->x, E, rb), rows(S->sh_g, FFs, rb), nullptr, 1.0f,
            Ly.sh_gate.type, E, FFs};
    j[2] = {Ly.sh_up.q, rows_q(S->xq, E, rb), rows(S->x, E, rb), rows(S->sh_u, FFs, rb), nullptr, 1.0f,
            Ly.sh_up.type, E, FFs};
    if (!mv(j, 3, "router / shared expert")) return false;
    // Inside the FFN block: the router's logits and the shared expert's gate/up, so a divergence here separates a
    // routing/input fault from the shared path and from the routed experts (which the post-FFN stage covers).
    if (rows_per_row_) {
        gf::pipe_probe(rows(S->rlog, g.n_expert, rb), g.n_expert, rb * 2048 + 1024 + il * 4 + 1, s);
        gf::pipe_probe(rows(S->sh_u, FFs, rb), (int) FFs, rb * 2048 + 1024 + il * 4 + 2, s);
    }
    // its own error string: `err` is the scheduler's, and a cancelled or failed prompt leaves its text there
    // until the next prompt - read here, it failed every running request (with the setting off too)
    std::string gerr;
    const bool grouped = glmfast::group_experts_on() && rows_moe_grouped(il, n, gerr);
    if (!gerr.empty()) { err = gerr; return false; }
    // grouped failed here: the previous layer's prefetch is still pending, and this layer's experts read its slots
    if (rows_pred_pf_n() > 0 && !grouped) cudaStreamWaitEvent(s, F->ev_pf_prev, 0);
    for (int i = 0; !grouped && i < n; ++i) {
        const MoeIo io = {rows(S->x, E, R(i)), rows_q(S->xq, E, R(i)), rows(S->rlog, g.n_expert, R(i)),
                          rows(S->sh_g, FFs, R(i)), rows(S->sh_u, FFs, R(i)), rows(S->ffn, E, R(i))};
        if (!fast_moe_experts(il, io, false, 0, no_bias, pf_pending, err, true)) return false;
        // TRANSFER's honest producer on the path that actually runs: one per row per layer, issued here rather than in
        // rows_moe_grouped (which never executes with grouping off - a counter there read zero forever and I read the
        // zero as 'every layer was empty').
        if (rows_per_row_ && F->pipe) F->pipe->transfers_done.fetch_add(1, std::memory_order_relaxed);
        // the route log: this row's ids are still the plan's (the next row's route overwrites them); the copy is
        // async and the line prints one step later, when the previous step's copies are done - nothing syncs
        if (rows_route_log_on() && l0_ == 0 && i < 2 && rb == 0)
            rows_route_log_capture(s, F->md.plan_id, g.n_exp_used);   // (the token path's shared plan)
    }
    return true;
}

// STRATA_GLM_GROUP_EXPERTS=1: the n rows' routed experts as the verify window's batched MoE takes a window's tokens
// (fast_moe_window).  Every row is routed first - into its own plan, fetch sources, disk answer, CPU-lane answer slot
// (the window's route state, its rows the window's tokens) and its own 8 scratch slots (rows_scr), each row's wait and
// fetch before the next row's route (the host's answer to a disk miss is one buffer) - then each distinct expert runs
// once for all the rows that routed it (gf::moe_win_experts, exl3::win_experts: each row's output is moe_gate_up +
// moe_down's for it alone, bit for bit), then the rows' CPU-lane answers in one launch.  An expert two rows route is
// read once instead of twice, and fast_service holds the group's routes and answers their CPU-lane experts as one
// merged request (win_cpu_merged on this part's pool: each RAM-tier expert read once for the group).  The shared
// expert's down rows of all n rows are one mv_nt (mv's bits for each row - the per-row path's fused down's too).
// Its route is the same per row: the spares a row's fetch takes are resident for the rows after it, as they are for
// the next row's route on the per-row path.  False with err empty: not here (the rows run one by one).
bool Glm5Model::rows_moe_grouped(int il, int n, std::string& err) {
    namespace x3 = strata::kernels::exl3;
    FastState* F = fast_;
    RowsState* S = rows_;
    auto& W = F->win;
    // The row's OWN slice (row_base_ 0 = an ordinary call, so this is an identity) and the group window's block for that
    // row: without them a driver running one row per call would put every row in slice 0 of the activations and of the
    // grouping arrays - silent corruption rather than a fault (docs/LAYER_PIPELINE.md).
    const auto R = [&](int k) { return row_base_ + k; };
    const int rb = row_base_;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, FFs = g.n_ff_exp * g.n_shared, K = g.n_exp_used;
    // n == 1 IS a pipeline row when the rotor drives one row per call (rows_per_row_) - including row 0, whose INDEX is
    // 0.  `row_base_ == 0` cannot tell "the lone token call" from "row 0", so row 0 was refused the per-row MoeDev and
    // fell back to the GLOBAL F->md: one plan, one fetch_src, one wait channel for both rows.  Fourth time on this branch
    // that an index carried a boolean.  STRATA_GLM_PIPE_ROW0_FALLBACK=1 restores the old refusal so it can be COUNTED.
    static const bool admit_row0 = getenv("STRATA_GLM_PIPE_ROW0_FALLBACK") == nullptr;
    if (n < 2 && row_base_ == 0 && !(rows_per_row_ && admit_row0)) {
        if (rows_per_row_) {   // a per-row call reaching the global MoeDev is the bug, not a mode
            F->md_fallbacks.fetch_add(1, std::memory_order_relaxed);
            int expect = -1;
            F->md_fallback_first_il.compare_exchange_strong(expect, il, std::memory_order_relaxed);
        }
        return false;
    }
    const bool wide = n > F->rows_group && n <= F->rows_scr_rows && F->all_in_vram;   // (STRATA_GLM_MULTI_MTP)
    if ((n > F->rows_group && !wide) || F->rows_scr == nullptr || W.mplan_ptr == nullptr || W.tmax < n ||
        (Ly.exl3 && (W.wh == nullptr || !x3::several_ok()))) {
        static std::atomic<bool> told{false};   // (a run with the setting on that never groups should say so; both
        if (!told.exchange(true)) {            //  parts' threads get here)
            std::fprintf(stderr, "glm rows: CUDA%d a group of %d rows runs its experts row by row (grouping set up for "
                                 "%d rows%s) - said once\n", dev_, n, F->rows_group,
                         Ly.exl3 && W.wh == nullptr ? ", no EXL3 rows" : "");
        }
        return false;
    }
    cudaStream_t s = rows_stream_ != nullptr ? rows_stream_ : F->cs;   // (a pipeline driver gives a row its own)
    const bool lane = (F->cpu_plan != 0ull || F->helper.device() >= 0) && F->cpu_fmt[(size_t) il].n_ff > 0;
    const bool skip_empty = glmfast::skip_empty_on();
    const bool empty = F->all_resident || (skip_empty && F->layer_resident(il));
    gf::MoeDev md = F->md;
    md.pf_src = F->pf_buf[il & 1];
    md.pf_dst = F->pf_buf[il & 1] + 8;
    md.pf_n = F->pf_n_buf[il & 1];
    // STRATA_GLM_ROWS_PRED_PF=<n>: the next layer's predicted experts (its router weight on the current x) claim
    // that layer's spares at this layer's route and are copied on the side stream while this layer computes - the
    // prefetch overlaps the grouped experts, the biggest phase, not sh_down.  The staging is per row (the token
    // path's pf buffers hold one route's stage; a second row's route would overwrite the first row's claims, whose
    // spares are already taken: tab marked resident, no bytes copied - garbage slots).  Exclusive with
    // STRATA_GLM_ROWS_PF (shared ev_pf_prev and tier tables under both arms).
    static const int pred_pf = rows_pred_pf_n();
    static const int rows_pf = [] {
        const char* v = getenv("STRATA_GLM_ROWS_PF");
        return v != nullptr && v[0] != '\0' ? std::atoi(v) : 0;
    }();
    const bool pred_on = pred_pf > 0 && rows_pf == 0 && il + 1 < l1_ && F->L[(size_t) il + 1].moe;
    // the previous layer's prefetch must have landed before this layer's routes re-stage the buffers (and before
    // the experts read the claimed slots).  Never recorded: cudaStreamWaitEvent is a no-op.
    if (pred_pf > 0 && rows_pf == 0) cudaStreamWaitEvent(s, F->ev_pf_prev, 0);
    if (pred_on) {   // the prediction costs a router GEMV for all rows (the token path pays the same, glm_fast_path.cu)
        gf::MvJob pj = {F->L[(size_t) il + 1].router, nullptr, S->x, S->plog, nullptr, 1.0f, gf::kTypeBF16, E,
                        g.n_expert};
        if (!gf::mv_nt(&pj, 1, n, s)) {
            err = "glm rows: the next layer's prediction GEMV";
            return false;
        }
    }
    const auto row_dev = [&](int i) {   // (fast_moe_window's tok_dev, and the row's own scratch slots)
        gf::MoeDev m = md;
        m.plan_ptr = W.mplan_ptr + 8 * R(i);
        m.plan_w = W.mplan_w + 8 * R(i);
        m.plan_id = W.mplan_id + 8 * R(i);
        m.fetch_src = W.mfetch + 8 * R(i);
        m.cpu_part = W.mcpu_part + (size_t) E * R(i);
        m.cpu_flag = W.mcpu_flag + R(i);
        // NULL + 8*row is POINTER 8, not null.  rows_scr is allocated only with STRATA_GLM_GROUP_EXPERTS, so with
        // grouping off (the default) row 0 got a real nullptr and row 1 got address 8: the trailing row's fetched
        // experts read a scratch slot that is not memory.  No identity check can see it (the tier table is right) and
        // no fill event can fix it (nothing was late) - which is why it survived five theories and row 0 never showed it.
        m.scratch = F->rows_scr != nullptr ? F->rows_scr + 8 * R(i) : nullptr;
        // this row's OWN handshake channel: wait_seq[R(i)] and resp[R(i)].  Row 0 is channel 0, exactly the lone token
        // path, so n = 1 is unchanged; n > 1 gives each waiter its own word and answer buffer.
        //
        // RESTORED (2026-10-30): this said `R(i) & 7` at cd52717 - the tree that ran 796 clean rotor entries - and the
        // v1.1.40 replay reverted it to `i & 7`.  Every other line in this lambda indexes by R(i) = row_base_ + i, and
        // the rotor calls rows_layers(rw + r, 1, ...), so i is ALWAYS 0 there: row 1 has been publishing on CHANNEL 0
        // all evening.  Both rows then answer into resp[0], the service overwrites slot 0 in seq order, and a waiter
        // whose answer was overwritten before it read it spins to its ~60 s ceiling - the drain timeout and the
        // NON-FINITE both.  It also explains the one number we could never square: wait_seq[1] always read 0, because
        // nothing ever wrote channel 1.
        m.wait_ch = R(i) & 7;
        if (pred_on) {   // per-row prefetch staging: the token path's pf buffers hold one route's stage
            m.pf_src = S->pf_src + 8 * R(i);
            m.pf_dst = S->pf_dst + 8 * R(i);
            m.pf_n = S->pf_n + i;
        }
        if (i > 0 && md.cpu_seq != nullptr) {
            m.cpu_seq = W.cpu_seq_w + 16 * R(i);
            m.cpu_ans = W.ans_d[R(i)];
        }
        return m;
    };
    const float* no_bias[gf::kAhead] = {nullptr, nullptr, nullptr, nullptr};
    // STRATA_GLM_ROWS_PF=1 (de-fused fetch): each row's fetch crosses PCIe on the side stream, issued right after
    // its route, so it runs beside the next row's route GEMVs - but the tier tables are shared device state both
    // streams then touch concurrently (tab, wait_seq, the counters): measure before trusting this mode.  =2: every
    // row routed first, then the fetches, which then overlap only the shared expert's down below - no shared-state
    // concurrency.  The grouped experts wait for the fetches on the compute stream (below, before the launches
    // that read the slots).  The addresses are exact: every row is routed into its own plan before the first
    // fetch runs.
    for (int i = 0; i < n; ++i) {
        const gf::MoeDev m = row_dev(i);
        const MoeIo io = {row_at(S->x, E, R(i)), rowq_at(S->xq, E, R(i)), row_at(S->rlog, g.n_expert, R(i)),
                          row_at(S->sh_g, FFs, R(i)), row_at(S->sh_u, FFs, R(i)), row_at(S->ffn, E, R(i))};
        fast_moe_route(il, io, m, (uint8_t*) W.sh_hq + (size_t) R(i) * W.sh_hq_s, pred_on, 0, no_bias, lane,
                       rows_per_row_ ? R(i) : i, rows_per_row_ ? 0 : n, rows_per_row_ ? 0 : 1,
                       row_at(S->plog, g.n_expert, R(i)), pred_pf);
        ++F->expected;
        if (rows_route_log_on() && l0_ == 0 && i < 2) rows_route_log_capture(s, W.mplan_id + 8 * R(i), K);
        if (empty) continue;
        if (rows_pf == 1) {
            cudaEventRecord(F->ev_pf, s);              // the route wrote this row's plan on the compute stream
            cudaStreamWaitEvent(F->ps, F->ev_pf, 0);
            gf::moe_wait(m, E, F->ps);
            gf::moe_fetch(m, K, Ly.blob, F->ps);
        } else if (rows_pf == 2) {
            // this row's fetch runs once, together with the other rows' fetches after the loop, on the side stream -
            // fetching here too would run it twice, and the second fetch re-assigns the scratch spares against tier
            // state the first fetch already updated, so the plans and the slot contents disagree (an illegal access)
        } else if (skip_empty) {
            gf::moe_wait_fetch(m, E, K, Ly.blob, s);
        } else {
            gf::moe_wait(m, E, s);
            gf::moe_fetch(m, K, Ly.blob, s);
            // the counter is incremented where the copy is ISSUED, by the thread that issued it
        }
    }
    if (rows_pf == 2 && !empty) {   // every route is on the compute stream: the fetches follow it, on the side stream
        cudaEventRecord(F->ev_pf, s);
        cudaStreamWaitEvent(F->ps, F->ev_pf, 0);
        for (int i = 0; i < n; ++i) {
            const gf::MoeDev m = row_dev(i);
            gf::moe_wait(m, E, F->ps);
            gf::moe_fetch(m, K, Ly.blob, F->ps);
        }
    }
    // TRANSFER's marker: this layer's fills are on this stream, so the event completes when the last of them lands.
    if (rows_per_row_ && F->plan_cap != nullptr && il < F->plan_cap_layers && rb < F->plan_cap_rows) {
        const size_t off = ((size_t) rb * F->plan_cap_layers + (size_t) il) * 8;
        cudaMemcpyAsync(F->plan_cap + off, W.mplan_ptr + 8 * R(0), 8 * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost, s);
        cudaMemcpyAsync(F->plan_id_cap + off, W.mplan_id + 8 * R(0), 8 * sizeof(int), cudaMemcpyDeviceToHost, s);
    }
    if (rows_per_row_ && !empty && il < (int) F->ev_fill.size() && F->ev_fill[(size_t) il] != nullptr) {
        if (il < (int) F->ev_fill_gen.size()) F->ev_fill_gen[(size_t) il] = F->step_seq;   // stamp BEFORE the record
        if (il < (int) S->ev_fill_row.size()) S->ev_fill_row[(size_t) il] = row_base_;      // ...and WHO records it
        cudaEventRecord(F->ev_fill[(size_t) il], s);
    }
    if (pred_on && !empty) {   // the claims are staged: the side stream copies them while this layer computes
        cudaEventRecord(F->ev_pf, s);
        cudaStreamWaitEvent(F->ps, F->ev_pf, 0);
        for (int i = 0; i < n; ++i) {
            gf::MoeDev m = md;
            m.pf_src = S->pf_src + 8 * R(i);
            m.pf_dst = S->pf_dst + 8 * R(i);
            m.pf_n = S->pf_n + i;
            gf::moe_prefetch(m, F->L[(size_t) il + 1].blob, F->ps);
        }
        cudaEventRecord(F->ev_pf_prev, F->ps);   // the next layer's routes wait for this at its entry
    }
    const float* sh_out = nullptr;
    if (Ly.sh_down.q != nullptr) {
        gf::MvJob sd = {Ly.sh_down.q, (uint8_t*) W.sh_hq + (size_t) rb * W.sh_hq_s, nullptr,
                        W.sh_out + (size_t) E * rb, nullptr, 1.0f,
                        Ly.sh_down.type, FFs, E};
        if (!gf::mv_nt(&sd, 1, n, s)) {
            err = "glm rows: the grouped shared expert's down";
            return false;
        }
        sh_out = W.sh_out + (size_t) E * rb;
    }
    if (rows_pf > 0 && !empty) {   // the grouped experts read the slots: every fetch must have landed
        cudaEventRecord(F->ev_pf_prev, F->ps);
        cudaStreamWaitEvent(s, F->ev_pf_prev, 0);
    }
    gf::MoeWin mw;
    // the group window's arrays are [T * 8] (glm_window.cu): a per-row call takes THIS row's block, so two rows in
    // flight never group into one another's plans.  rb (above) is 0 for an ordinary call, so nothing moves by default.
    mw.plan_ptr = W.mplan_ptr + 8 * rb;
    mw.plan_w = W.mplan_w + 8 * rb;
    mw.cpu_part = W.mcpu_part + (size_t) E * rb;
    mw.cpu_flag = W.mcpu_flag + rb;
    mw.grp_ptr = W.grp_ptr + 8 * rb;
    mw.grp_start = W.grp_start + 8 * rb;
    mw.grp_ent = W.grp_ent + 8 * rb;
    mw.n_groups = W.n_groups + 8 * rb;
    mw.T = n;
    mw.k = K;
    mw.n_embd = E;
    mw.n_ff = g.n_ff_exp;
    mw.limit = g.swiglu_exp;
    if (Ly.exl3) {
        gf::moe_win_group(mw, s, rows_small_on());
        x3::WinArgs xa;
        xa.grp_ptr = W.grp_ptr + 8 * rb;
        xa.grp_start = W.grp_start + 8 * rb;
        xa.grp_ent = W.grp_ent + 8 * rb;
        xa.n_groups = W.n_groups + 8 * rb;
        xa.plan_ptr = W.mplan_ptr + 8 * rb;
        xa.plan_w = W.mplan_w + 8 * rb;
        xa.cpu_part = W.mcpu_part + (size_t) E * rb;
        xa.cpu_flag = W.mcpu_flag + rb;
        xa.x = S->x;
        xa.x_stride = (size_t) E;
        xa.h = W.wh + (size_t) 8 * rb * g.n_ff_exp;
        xa.part = W.wpart;
        xa.sh_out = sh_out;
        xa.out = S->ffn;
        xa.out_stride = (size_t) E;
        xa.lay = Ly.xl;
        xa.T = n;
        xa.k = K;
        xa.n_embd = E;
        xa.n_ff = g.n_ff_exp;
        xa.limit = g.swiglu_exp;
        if (!x3::win_experts(xa, s)) {
            err = "glm rows: the grouped EXL3 experts";
            return false;
        }
    } else {
        mw.xq = S->xq;
        mw.xq_stride = (size_t) (E / 32) * kQ8;
        mw.hq = (uint8_t*) W.whq + (size_t) 8 * rb * ((size_t) g.n_ff_exp / 32 * kQ8);
        mw.part = W.wpart + (size_t) 8 * rb * E;
        mw.sh_out = sh_out;
        mw.out = S->ffn;
        mw.out_stride = (size_t) E;
        mw.down_off = Ly.down_off;
        if (!gf::moe_win_experts(Ly.gu_type, Ly.d_type, mw, s, rows_small_on())) {
            err = "glm rows: the grouped experts (types " + std::to_string(Ly.gu_type) + " / " +
                  std::to_string(Ly.d_type) + ")";
            return false;
        }
    }
    // the CPU lane's answers last: the device's experts ran while the host computed the group's merged request
    if (lane && !empty && md.cpu_seq != nullptr) {
        if (n > gf::kMaxMvNt) {   // (rows_group keeps n at 8 or fewer; never leave the rows' CPU answers out quietly)
            err = "glm rows: " + std::to_string(n) + " rows are more than the CPU-lane wait takes";
            return false;
        }
        gf::CpuWaitRows cw;
        for (int i = 0; i < n; ++i) {
            const gf::MoeDev m = row_dev(i);
            cw.seq[i] = m.cpu_seq;
            cw.ans[i] = m.cpu_ans;
            cw.out[i] = row_at(S->ffn, E, R(i));
        }
        gf::moe_cpu_wait_rows(cw, n, E, s);
    }
    return true;
}

bool Glm5Model::rows_head(const RowSeq* rw, const strata::kernels::SamplerParams* sps, int n, std::string& err) {
    FastState* F = fast_;
    RowsState* S = rows_;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd;
    cudaSetDevice(dev_);
    cudaStream_t s = rows_stream_ != nullptr ? rows_stream_ : F->cs;   // (a pipeline driver gives a row its own)
    auto& rows = row_at;
    auto& rows_q = rowq_at;
    // the row's OWN slice of the 8-wide per-row buffers.  row_base_ is 0 for an ordinary call, so this is an identity
    // unless a driver runs one row per call - in which case the row must take the slice it owns, not slice 0, or two
    // concurrent rows write the same activation (docs/LAYER_PIPELINE.md).
    const int rb = row_base_;   // this row's block of head_x / head_xq / logits
    const auto R = [&](int k) { return row_base_ + k; };
    const auto mv = [&](gf::MvJob* j, int nj, const char* what) -> bool {
        if (gf::mv_nt(j, nj, n, s)) return true;
        err = std::string("glm rows: ") + what;
        return false;
    };
    // ---- the head: one read of the output matrix for every row, then each row's draw
    for (int i = 0; i < n; ++i)
        gf::head_prep(rw[i].res, w_.at("output_norm.weight"), g.norm_eps, E, rows(S->head_x, E, R(i)),
                      rows_q(S->head_xq, E, i), s);
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, rows_q(S->head_xq, E, rb), rows(S->head_x, E, rb), rows(S->logits, g.n_vocab, rb), nullptr, 1.0f, ow.type, E, g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!mv(&o, 1, "output head")) return false;
    // THE DRAW, HASHED (2026-10-30).  A residual that changes while the token does not leaves two suspects: the head not
    // reading this row, or the draw degenerating.  A hash separates them; the magnitude and the gap to second place say
    // WHICH degeneracy, because a saturated or uniform row of logits draws the same token forever without being corrupt.
    // The stride print is the other half: mv_nt replicates activations by advancing xq n_in/32 blocks and xf n_in floats,
    // so if the per-row stride here is padded to anything else, activation 1 reads the wrong memory while 0 is perfect -
    // the exact asymmetry under investigation.
    if (n > 1 || getenv("STRATA_GLM_TOK_ID_LOG") != nullptr) {
        {
            static std::atomic<bool> stride_told{false};
            if (!stride_told.exchange(true)) {
                const char* p0 = (const char*) rows(S->head_x, E, 0);
                const char* p1 = (const char*) rows(S->head_x, E, 1);
                const char* q0 = (const char*) rows_q(S->head_xq, E, 0);
                const char* q1 = (const char*) rows_q(S->head_xq, E, 1);
                const char* l0 = (const char*) rows(S->logits, g.n_vocab, 0);
                const char* l1 = (const char*) rows(S->logits, g.n_vocab, 1);
                std::fprintf(stderr,
                             "glm rows: head strides CUDA%d E=%d n_vocab=%d: head_x %lld (want %lld), head_xq %lld "
                             "(want %lld), logits %lld (want %lld)\n",
                             dev_, E, g.n_vocab, (long long) (p1 - p0), (long long) ((int64_t) E * 4),
                             (long long) (q1 - q0), (long long) ((int64_t) (E / 32) * kQ8),
                             (long long) (l1 - l0), (long long) ((int64_t) g.n_vocab * 4));
            }
        }
        std::vector<float> lg((size_t) g.n_vocab);
        for (int i = 0; i < n; ++i) {
            const float* src = rows(S->logits, g.n_vocab, R(i));
            cudaMemcpyAsync(lg.data(), src, (size_t) g.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, s);
            if (cudaStreamSynchronize(s) != cudaSuccess) {
                (void) cudaGetLastError();
                continue;
            }
            unsigned long long h = 1469598103934665603ull;
            int i1 = 0, i2 = 0;
            float v1 = lg[0], v2 = lg[0];
            for (int k = 0; k < g.n_vocab; ++k) {
                uint32_t b;
                std::memcpy(&b, &lg[(size_t) k], sizeof b);
                h = (h ^ b) * 1099511628211ull;
                const float v = lg[(size_t) k];
                if (v > v1) {
                    v2 = v1;
                    i2 = i1;
                    v1 = v;
                    i1 = k;
                } else if (v > v2 && k != i1) {
                    v2 = v;
                    i2 = k;
                }
            }
            std::fprintf(stderr,
                         "glm rows: head CUDA%d row %d logits %016llx top1 %d %.6f top2 %d %.6f gap %.6f\n", dev_, i, h,
                         i1, (double) v1, i2, (double) v2, (double) (v1 - v2));
        }
    }
    for (int i = 0; i < n; ++i) {
        const float* lg = rows(S->logits, g.n_vocab, R(i));
        if (strata::core::rows_pick_argmax(sps[i]))
            gf::argmax(lg, g.n_vocab, S->tok + i, s);
        else
            pick_tokens(lg, 1, sps[i], nullptr, 0, S->tok + i, s);
    }
    return true;
}

bool Glm5Model::multi_rows(const int* seqs, const int32_t* tokens, const strata::kernels::SamplerParams* sps, int n,
                           int* tokens_out, std::string& err) {
    if (!rows_ok() || n_parts_ != 1 || seq_.empty() || n < 1 || n > rows_max()) {
        err = "glm rows: not available here (one GPU, several sequences, 1.." + std::to_string(rows_max()) + " rows)";
        return false;
    }
    if (seq_[0].hop_h == nullptr && !seq_pipe_ready(err)) return false;   // (each sequence's pinned embedding row)
    cudaSetDevice(dev_);
    // between steps (the device idle): finished promotions go live and the layers get fresh spare slots, as the
    // token path does between tokens - every 32 steps instead, the spares ran dry and the VRAM tier stopped following
    // the routes (two conversations on a V100 streaming its experts from RAM: 27.1 tok/s in all against 34.2)
    if (!fast_boundary(err)) return false;
    FastState* F = fast_;
    RowsState* S = rows_;

    // BF16 KDA OUTPUT (2026-10-28): rows_layers' kda output projection is a bf16 weight, and the rows path only ever
    // has kda_rec's q8_1 activations (gated_q) - no float ones.  mv_nt, the multi-row kernel, has only the
    // row_dot_bf16_cols form and refuses a bf16 job without xf ("mv_nt: a type 30 job without its float activations"),
    // while the single-column mv falls back to any_row_dot, which does handle a q8 activation.  So a step of TWO or
    // more conversations failed outright and generate.cpp's multi_fail killed the whole batch (both requests died in
    // ~0.08 s with "glm rows: kda output"); one conversation never showed it (n == 1 stays on mv).  Until that job is
    // given float activations, decode such a batch one row at a time: every row's arithmetic is exactly the n == 1
    // step's, so the outputs are identical - we only lose the shared weight reads.
    // (RESTORED: 6ae276e deleted this in favour of an mv_nt pre-pass that never compiled and does not cover the
    //  single-job mv() calls at glm_rows.cu:491 and glm_fast_path.cu:5996.)
    // THE ROTOR IS EXEMPT (2026-10-30).  This constraint is about the n-ROW calls: mv_nt refuses a BF16 job without xf.
    // The rotor calls rows_layers with ONE row (the rotor loop below: `rows_layers(rw + r, 1, ...)`), so that job never
    // reaches mv_nt there - and since 737bd2f mv() dequantises a BF16 job's q8_1 activation itself.  The guard used to
    // sit UNGUARDED ahead of the rotor check, so a batch that qualified for the rotation was decoded row by row before
    // the rotor was ever consulted: the feature was not declined, it was skipped (measured: 0 rotor entries on a boot
    // where every precondition - pipe mode, n == 2, cs2, aq - was met).
    const bool rotor_will_run = glmfast::layer_pipe_mode() >= 2 && n == 2 &&
                                F->cs2 != nullptr && S->aq != nullptr;
    // The fallback stays for every case that still needs it: n >= 3, the declined path (aq/cs2 missing), and the fused
    // token path.  Do not re-broaden it without re-reading the reason above.
    if (n > 1 && !rotor_will_run) {
        bool bf16_out = false;
        for (int il = l0_; il < l1_ && !bf16_out; ++il) bf16_out = F->L[(size_t) il].out.type == 30;   // GGML_TYPE_BF16
        if (bf16_out) {
            for (int i = 0; i < n; ++i)
                if (!multi_rows(seqs + i, tokens + i, sps + i, 1, tokens_out + i, err)) return false;
            return true;
        }
    }
    // (the token path's profiler collects one token's marks at a time: none from a batched step)
    struct ProfOff {
        FastState* F;
        bool was;
        ~ProfOff() { F->prof_on = was; }
    } prof_off{F, F->prof_on};
    F->prof_on = false;
    RowSeq rw[gf::kMaxMvNt];
    // THREE-POINT RESIDUAL HASH (2026-10-30).  A row whose position advances while its token never changes is either a row
    // whose content is never written or one that recomputes the same thing - and those are different faults with different
    // homes.  Two points cannot separate "the embedding never landed" from "the layers never wrote it", and the first of
    // those lives UPSTREAM of the rotor: rows_bind/rows_embed run once for the whole batch on F->cs, while everything after
    // them is per row.  So: before the embed, after the embed, and after the last layer.
    unsigned long long res_pre[gf::kMaxMvNt] = {0}, res_emb[gf::kMaxMvNt] = {0}, res_post[gf::kMaxMvNt] = {0};
    const bool reslog = n > 1 || getenv("STRATA_GLM_TOK_ID_LOG") != nullptr;   // the n == 1 baseline on demand
    const auto res_hash = [&](const float* p) -> unsigned long long {
        if (p == nullptr) return 0ull;
        std::vector<float> h((size_t) g_.n_embd);
        cudaMemcpyAsync(h.data(), p, (size_t) g_.n_embd * sizeof(float), cudaMemcpyDeviceToHost, F->cs);
        if (cudaStreamSynchronize(F->cs) != cudaSuccess) {
            (void) cudaGetLastError();
            return 0ull;
        }
        unsigned long long x = 1469598103934665603ull;   // FNV-1a over the row's residual: change is what matters, not value
        for (float v : h) {
            uint32_t b;
            std::memcpy(&b, &v, sizeof b);
            x = (x ^ b) * 1099511628211ull;
        }
        return x;
    };
    if (!rows_bind(seqs, n, rw, err)) return false;
    if (reslog)
        for (int i = 0; i < n; ++i) res_pre[i] = res_hash(rw[i].res);
    if (!rows_embed(rw, tokens, n, err)) return false;
    if (reslog)
        for (int i = 0; i < n; ++i) res_emb[i] = res_hash(rw[i].res);
    // THE ROTATION IS THE DEFAULT (2026-10-30).  STRATA_GLM_LAYER_PIPE unset means 2: row r runs the layer k - r while
    // row 0 runs k, so one row waits on its fetch or its CPU lane while the other computes, and each row keeps its own
    // stream.  Measured against the serial arm inside ONE boot, n=6 per arm, two prompts a measurement, warm-up pair
    // discarded: ~+4 % paired (median +1.2 tok/s, overlap faster in 6 of 6 adjacent pairs), and bit-identical tokens
    // against one stream on the same binary.  STRATA_GLM_LAYER_PIPE=0/1 turns it off; STRATA_GLM_PIPELINE_SERIAL=1 (or
    // strata-pipe-arm.txt) puts both rows back on one stream.
    // The STRATA_GLM_PIPELINE_UNCERTIFIED gate retired with this: its stated reason - two rows sharing the expert spare
    // claims - was the per-row MoeDev bug (they were built in a function this path never called), and floor_violations
    // and spare_steals have read zero since.  A guard naming a cause that no longer exists is worse than no guard.
    const int pipe_mode = glmfast::layer_pipe_mode();
    // This step's schedule figures, reset by the rotor branch below and quoted by the per-step token-id line - which sits
    // OUTSIDE that branch, so these cannot be branch-local: a counter only visible in the error path is one nobody reads
    // (and the round-cap gate never fired, which is how the first two of these went unread for a whole boot).
    int occ_defers = 0, reo_defers = 0, spread_max = 0;   // this step's schedule figures (the token-id line quotes them)
    // THE CHANNEL DELTA, ON EVERY FAILURE EXIT (2026-10-30).  Drain timeout and NON-FINITE are one mechanism seen through
    // two observation windows - a request that is never answered - so a counter that only prints on the drain face is a
    // fail-open instrument: the boot that fails non-finite instead produces no numbers at all, which is exactly the shape
    // the 90 s run gave us.  Rendered here once and appended by every path that returns false from a step.
    const auto ch_delta = [&]() -> std::string {
        std::string out;
        for (int c = 0; c < 8; ++c) {
            const unsigned long long pubd = (unsigned long long) F->pub_ch[c].load(std::memory_order_relaxed);
            const unsigned long long ansd = (unsigned long long) F->ans_ch[c].load(std::memory_order_relaxed);
            if (pubd == 0 && ansd == 0) continue;
            // UNITS: pubd is the HOST's hand-outs, ansd is ANSWER-WRITES.  They are different quantities and their
            // difference is NOT a backlog - a request that needs no answer written never increments ansd, and the
            // service's own position is `next`, which the stall line prints separately.  The old label said "BEHIND",
            // and that word sent two people looking for a queue that does not exist: the service has been caught up in
            // every boot measured (waiting = H + 1, i.e. parked on a seq no step ever handed out).
            out += "; ch " + std::to_string(c) + " hand-outs " + std::to_string(pubd) + " answer-writes " +
                   std::to_string(ansd);
        }
        // AND THE RING ITSELF.  pub_ch counts at LAUNCH, so "published > answered on both channels" is the shape of a seq
        // that was handed out and never written into its slot - the service then waits for it forever and every later
        // request queued behind it on both channels.  Reading the slot the service is parked on separates that from a
        // service thread blocked elsewhere: an old seq in that slot is a kernel that never ran; the right seq there is a
        // consumer that is not consuming.
        // (the ring-scan that stood here is REMOVED: a CUDA stream is in-order, so once one kernel wedges every launch
        // the host enqueued behind it has not run either - H - T would be the host's enqueue lead by construction, not a
        // fault, and would read non-zero in every stalled boot.  It cannot discriminate, so it is not measured.)
        if (F->ring_h != nullptr) {
            const unsigned sn = F->svc_next.load(std::memory_order_relaxed);
            out += "; service waiting for seq " + std::to_string(sn) + " (next in ring:";
            for (int k = 0; k < 4; ++k) {
                const gf::MoeRequest* r = F->ring_h + ((sn + (unsigned) k) % gf::kRingSize);
                out += " " + std::to_string(k) + "=seq " + std::to_string(r->seq) + "/ch " + std::to_string(r->ch);
            }
            out += ")";
        }
        return out;
    };
    if (pipe_mode >= 2 && n == 2 && F->cs2 != nullptr && S->aq != nullptr) {
        // ---- the rotation (docs/LAYER_PIPELINE.md, increment 4): row r runs the layer k - r while row 0 runs k, so one
        // row is waiting on its fetch or its CPU lane while the other computes.  Each row keeps its OWN stream, so its
        // layers stay in order; the two streams are free to overlap.  One layer a row, one row per call - the slices and
        // the handshake channel both key off row_base_, so they cannot collide.
        // BISECT: the same rotation with both rows on ONE stream separates a slicing bug (wrong output either way)
        // from a genuine race (right output serial, wrong when overlapped).
        // RUNTIME ARM SWITCH.  This used to be `static const bool serial = getenv(...)`: fixed at process start, so
        // serial and overlap could never be compared inside one boot - and every cross-boot comparison on this branch
        // has misled at least once.  A one-character file is read at most every 200 ms ('s' = both rows on one stream,
        // anything else = overlap), which is outside the service loop and costs one stat per 200 ms.  The environment
        // still sets the default when the file is absent.
        static const bool serial_default = getenv("STRATA_GLM_PIPELINE_SERIAL") != nullptr;
        const bool serial = [&] {
            static bool cur = serial_default;
            static auto last = std::chrono::steady_clock::now();
            const auto now = std::chrono::steady_clock::now();
            if (now - last < std::chrono::milliseconds(200)) return cur;
            last = now;
            if (FILE* f = std::fopen("strata-pipe-arm.txt", "r")) {
                char c = 0;
                if (std::fread(&c, 1, 1, f) == 1) cur = (c == 's' || c == 'S');
                std::fclose(f);
            }
            return cur;
        }();
        const int L = l1_ - l0_;
        // floor starts at the first layer and only moves forward; a slot may not run past floor + S.  The leader is
        // therefore throttled by construction - no pause logic, no condvar - and the straggler is the one to accelerate.
        int floor = l0_;
        const uint64_t expected0 = F->expected;
        // The embed, the bind and the boundary all ran on F->cs; row 1 computes on F->cs2, and on one stream that
        // ordering was implicit.  Without this the second row's first layer can read its x before the embed that writes
        // it has run - a sporadic non-finite score a layer or two in, i.e. arithmetic, not an index fault.
        if (F->plan_cap == nullptr) {
            F->plan_cap_rows = n;
            F->plan_cap_layers = l1_;
            if (cudaHostAlloc((void**) &F->plan_cap, (size_t) n * l1_ * 8 * sizeof(unsigned long long),
                              cudaHostAllocMapped) != cudaSuccess ||
                cudaHostAlloc((void**) &F->plan_id_cap, (size_t) n * l1_ * 8 * sizeof(int),
                              cudaHostAllocMapped) != cudaSuccess) {
                F->plan_cap = nullptr;
                F->plan_id_cap = nullptr;
            }
        }
        ++F->step_seq;   // every arrival answer below is about THIS step
        // ENTRY, WITH THE STEP INDEX: the log already shows which step produced a token, so this says whether the rotor
        // was entered for the step that did NOT.  Without it, silence from the in-rotor watchdog is ambiguous between
        // 'not entered' and 'entered and healthy' - the same fail-open shape as the event oracle.
        std::fprintf(stderr, "glm pipe: CUDA%d step %u: rotor entered (n=%d, layers %d..%d)\n", dev_, F->step_seq, n,
                     l0_, l1_);
        {
            static std::atomic<bool> told_ans{false};
            if (!told_ans.exchange(true))   // POINTERS, not an inference: is ans_d[0] the same object as cpu_ans_h?
                std::fprintf(stderr, "glm pipe: CUDA%d CPU-lane answer slots: cpu_ans_h=%p ans_d[0]=%p ans_d[1]=%p "
                                     "(row 0 uses cpu_ans_h, row 1 uses ans_d[1])\n",
                             dev_, (void*) F->cpu_ans_h, (void*) F->win.ans_d[0], (void*) F->win.ans_d[1]);
        }
        gf::nonfinite_reset();
        cudaEventRecord(F->ev_hop, F->cs);
        cudaStreamWaitEvent(F->cs2, F->ev_hop, 0);
        gf::moe_wait_fetch_set_rows(n);   // N rows: N wait kernels can be resident at once
        rows_per_row_ = true;   // every layer below is ONE row of the pipeline
        if (F->pipe) F->pipe->watch_active.store(1, std::memory_order_release);   // arm the independent watchdog
        F->pipe_active.store(true, std::memory_order_release);   // from here the boundary's eviction rule applies
        // The rotor's skew IS the lead: row r runs layer k - r, so the spread is exactly n - 1.  S must be at least that
        // or the invariant would forbid the very advance it is meant to allow; the pin set is S x N x 8 slots.
        if (F->pipe_lead < n - 1) F->pipe_lead = n - 1;
        // The round index k is a ROUND, not a layer.  Driving the layer off `l0_ + k - r` meant a row that postponed a
        // layer silently skipped it - the displaced layer was never run by anyone, so the row computed its token with a
        // hole in the stack and the rotor kept submitting layers while nothing completed.  Each row now owns a cursor
        // (`cur[r].il`), advanced only when its layer actually ran, so a deferral postpones instead of dropping.
        for (int r = 0; r < n; ++r) {
            S->cur[r].il = l0_;
            S->cur[r].deferred = 0;
            S->cur[r].done = false;
        }
        if ((int) F->miss_layer.size() < l1_) {   // counted misses, one bucket a layer
            F->miss_layer.assign((size_t) l1_, 0);
            F->bytes_layer.assign((size_t) l1_, 0);
        }
        if ((int) F->ev_layer.size() < l1_) {   // one event a layer, created once
            const size_t had = F->ev_layer.size();
            F->ev_layer.resize((size_t) l1_, nullptr);
            for (size_t i = had; i < F->ev_layer.size(); ++i)
                cudaEventCreateWithFlags(&F->ev_layer[i], cudaEventDisableTiming);
            const size_t hadf = F->ev_fill.size();
            F->ev_fill.resize((size_t) l1_, nullptr);
            for (size_t i = hadf; i < F->ev_fill.size(); ++i)
                cudaEventCreateWithFlags(&F->ev_fill[i], cudaEventDisableTiming);
            F->ev_fill_gen.resize((size_t) l1_, 0u);
            if ((int) S->ev_fill_row.size() < l1_) S->ev_fill_row.assign((size_t) l1_, -1);
            if ((int) S->ev_layer_row.size() < l1_) S->ev_layer_row.assign((size_t) l1_, -1);
            const size_t hadc = F->ev_copy.size();
            F->ev_copy.resize((size_t) l1_, nullptr);
            for (size_t i = hadc; i < F->ev_copy.size(); ++i)
                cudaEventCreateWithFlags(&F->ev_copy[i], cudaEventDisableTiming);
        }
        // Safety net only: with one deferral per row between runs the schedule cannot exceed roughly 2L, and the cap
        // turns a scheduler bug into a visible short step instead of a spin.
        const int round_cap = 4 * L + 64;
        bool all_done = false;
        occ_defers = 0;
        reo_defers = 0;
        spread_max = 0;
        int k = 0;                 // hoisted: the gate below quotes how far the schedule got
        for (; k < round_cap && !all_done; ++k) {
            all_done = true;
            {   // the spread, recorded because it - not n - is what decides whether two rows share a parity
                int lo = l1_, hi = -1;
                for (int q = 0; q < n; ++q)
                    if (!S->cur[q].done) {
                        lo = std::min(lo, S->cur[q].il);
                        hi = std::max(hi, S->cur[q].il);
                    }
                if (hi >= 0) spread_max = std::max(spread_max, hi - lo);
            }
            for (int r = 0; r < n; ++r) {
                if (S->cur[r].done) continue;
                if (k < r) { all_done = false; continue; }   // this row has not started yet (the stagger is unchanged)
                const int il = S->cur[r].il;
                // NO TWO ROWS ON ONE LAYER: LayerPool::spare is three entries PER LAYER and the tier table publishes a
                // slot by (layer, expert), so two rows computing one layer race on a claim the other is mid-read on.
                // Making that impossible beats splitting three spares two ways.  A row whose layer another row has in
                // flight defers - the same mechanism the ladder's Reorder uses.
                {
                    bool occupied = false;
                    for (int q = 0; q < n; ++q)
                        if (q != r && !S->cur[q].done && S->cur[q].il - 1 == il) {
                            occupied = true;
                            break;
                        }
                    if (occupied) {
                        if (F->pipe) F->pipe->layer_busy.fetch_add(1, std::memory_order_relaxed);
                        all_done = false;
                        ++occ_defers;
                        continue;
                    }
                }
                // THE LADDER, with real residency AND real arrival.  layer_resident() only says every expert has a slot
                // assigned; the fill may still be copying (on this row's stream or, after another row ran the layer,
                // on the other one).  RETIRE answers that with cudaEventQuery - never a synchronise in the loop.
                cudaStream_t srow = (serial || (r & 1) == 0) ? F->cs : F->cs2;
                // FAIL CLOSED: an event that was never recorded this step is not evidence of arrival, and neither is
                // one still holding a previous step's record.  Only gen == step_seq makes the query an answer.
                bool filling = true;
                if (il < (int) F->ev_fill.size() && F->ev_fill[(size_t) il] != nullptr &&
                    il < (int) F->ev_fill_gen.size() && F->ev_fill_gen[(size_t) il] == F->step_seq) {
                    filling = cudaEventQuery(F->ev_fill[(size_t) il]) == cudaErrorNotReady;
                    (void) cudaGetLastError();   // NotReady is not an error to leave set
                }
                {
                    const bool res = F->layer_resident(il);
                    const auto st = FastState::pipe_stage(res, filling, /*can_reorder=*/true,
                                                          /*can_cpu=*/true, /*can_disk=*/true);
                    if (st == FastState::kStageInFlight && F->ev_fill[(size_t) il] != nullptr) {
                        // WAIT for the bytes rather than reorder past them: this orders only the fill against this
                        // row's read, so the overlap the design exists for is untouched.
                        cudaStreamWaitEvent(srow, F->ev_fill[(size_t) il], 0);
                    }
                    if (st == FastState::kStageReorder && S->cur[r].deferred == 0) {
                        S->cur[r].deferred = 1;
                        all_done = false;
                        ++reo_defers;
                        continue;   // same layer next round, not a skipped layer
                    }
                    S->cur[r].deferred = 0;
                }
                // floor = the lowest layer an active (unfinished) slot sits at; monotone within the step.
                int floor = l1_;
                for (int q = 0; q < n; ++q)
                    if (!S->cur[q].done) floor = std::min(floor, S->cur[q].il);
                floor = std::max(l0_, floor);
                if (il > floor + F->pipe_lead) { all_done = false; continue; }   // unreachable while S >= n-1
                row_base_ = r;
                rows_stream_ = srow;
                gf::mv_xf_set_slot(r);   // this row's own BF16 q8_1 scratch: two rows run on two streams
                // This layer's slots were filled by whichever row ran it first; that row's fill must have landed
                // before this one reads them.  (Never recorded: the wait is a no-op.)
                if (il < (int) F->ev_layer.size() && F->ev_layer[(size_t) il] != nullptr)
                    cudaStreamWaitEvent(rows_stream_, F->ev_layer[(size_t) il], 0);
                if (F->pipe) F->pipe->layers_submitted.fetch_add(1, std::memory_order_relaxed);   // SUBMIT, really launched
                if (!rows_layers(rw + r, 1, err, il, il + 1)) {
                    err += ch_delta();
                    row_base_ = 0;
                    rows_stream_ = nullptr;
                    rows_per_row_ = false;
                    F->pipe_active.store(false, std::memory_order_release);
                    return false;
                }
                if (il < (int) F->ev_layer.size() && F->ev_layer[(size_t) il] != nullptr) {
                    cudaEventRecord(F->ev_layer[(size_t) il], rows_stream_);
                    if (il < (int) S->ev_layer_row.size()) S->ev_layer_row[(size_t) il] = r;
                }
                S->cur[r].il = il + 1;              // advance ONLY on a real run
                S->cur[r].deferred = 0;
                if (S->cur[r].il >= l1_) S->cur[r].done = true;
                else all_done = false;
            }
        }
        // FAIL CLOSED (2026-10-30).  The cap exists to turn a scheduler bug into a VISIBLE short step instead of a spin,
        // and it did not: all_done was never tested after the loop, so a step in which the cap ran out walked straight on
        // to rows_head with rows that had never finished their stack.  The only symptom was the 20 s drain wait on a
        // kernel still running (measured: four clean steps, then a wedge on the fifth - submitted 446 of an expected 450).
        // A gate that fails open is sampled by exactly the case it was meant to catch.  The message names the state that
        // decides the diagnosis rather than guessing at it: how far each stuck row got, whether it was mid-deferral, and
        // how this step's deferrals split between the two mechanisms that can produce one.
        if (!all_done) {
            std::string why = "glm rows: the rotation exhausted its rounds (" + std::to_string(k) + " of " +
                              std::to_string(round_cap) + ") with rows unfinished - a scheduler bug, reported instead of "
                              "spun on; deferrals this step: " + std::to_string(occ_defers) +
                              " two-rows-on-a-layer, " + std::to_string(reo_defers) + " reorder";
            if (F->pipe) why += ", layer_busy " + std::to_string(F->pipe->layer_busy.load(std::memory_order_relaxed));
            for (int r = 0; r < n; ++r)
                if (!S->cur[r].done)
                    why += "; row " + std::to_string(r) + " stopped at layer " + std::to_string(S->cur[r].il) +
                           (S->cur[r].deferred ? " mid-deferral" : " not deferring");
            err = why + ch_delta();
            gf::mv_xf_set_slot(0);
            row_base_ = 0;
            rows_stream_ = nullptr;
            rows_per_row_ = false;
            F->pipe_active.store(false, std::memory_order_release);
            return false;
        }
        {
            static const bool misslog = getenv("STRATA_GLM_MISS_LOG") != nullptr;
            if (misslog) {   // the figure the copy lane is judged against, COUNTED rather than derived from a hit rate
                static uint64_t lm = 0, lb = 0;
                uint64_t tm = 0, tb = 0;
                for (int i2 = 0; i2 < (int) F->miss_layer.size(); ++i2) {
                    tm += F->miss_layer[(size_t) i2];
                    tb += F->bytes_layer[(size_t) i2];
                }
                const uint64_t dm = tm - lm, db = tb - lb;
                lm = tm;
                lb = tb;
                std::fprintf(stderr, "glm pipe: CUDA%d MISS step %u: %llu experts over PCIe, %.2f MB (%d layers, "
                                     "%.2f experts and %.2f MB a layer)\n",
                             dev_, F->step_seq, (unsigned long long) dm, (double) db / 1e6,
                             std::max(1, l1_ - l0_), (double) dm / std::max(1, l1_ - l0_),
                             (double) db / 1e6 / std::max(1, l1_ - l0_));
            }
        }
        gf::mv_xf_set_slot(0);
        gf::moe_wait_fetch_set_rows(1);
        row_base_ = 0;
        rows_stream_ = nullptr;
        rows_per_row_ = false;
        if (F->pipe) {
            {
                const int nf = gf::nonfinite_take();
                if (nf >= 0) {
                    const int lay = nf >> 8, st = (nf >> 4) & 15, rw_ = nf & 15;
                    static const char* stn[] = {"entry", "post-mixer", "post-FFN", "?"};
                    std::fprintf(stderr, "glm pipe: CUDA%d FIRST NON-FINITE at layer %d %s (row %d)\n", dev_, lay,
                                 st < 3 ? stn[st] : stn[3], rw_);
                }
            }
            {
                static std::atomic<bool> told{false};
                // DIAGNOSTICS OFF BY DEFAULT: both the probe report and the identity check SYNCHRONISE the host, and a
                // host sync while a device kernel waits for a host answer (a CPU-lane route) is a deadlock the
                // instrument creates.  STRATA_GLM_PIPE_DIAG=1 turns them back on for a measurement run.
                {
                    static const bool diag = getenv("STRATA_GLM_PIPE_DIAG") != nullptr;
                    static std::atomic<bool> once{false};
                    if (diag && !once.exchange(true)) {
                        gf::pipe_probe_report(serial ? "serial" : "overlap");
                        if (F->plan_cap != nullptr) {   // diagnostic: one sync, at the end of the step
                            cudaStreamSynchronize(F->cs);
                            cudaStreamSynchronize(F->cs2);
                            int bad_rows = 0;
                            for (int r2 = 0; r2 < F->plan_cap_rows; ++r2) {
                                int bad = 0, first_il = -1, first_id = -1, first_key = -1;
                                for (int l2 = 0; l2 < F->plan_cap_layers; ++l2) {
                                    if (l2 < l0_ || l2 >= l1_) continue;
                                    const size_t off = ((size_t) r2 * F->plan_cap_layers + (size_t) l2) * 8;
                                    auto& P2 = F->lp[(size_t) l2];
                                    for (int e = 0; e < 8; ++e) {
                                        const unsigned long long ptr = F->plan_cap[off + e];
                                        if (ptr == 0ull) continue;
                                        const int slot = P2.slot_index(ptr);
                                        if (slot < 0 || slot >= P2.n) { ++bad; if (first_il < 0) { first_il = l2; first_id = F->plan_id_cap[off + e]; first_key = -1; } continue; }
                                        if (P2.key[(size_t) slot] != F->plan_id_cap[off + e]) {
                                            ++bad;
                                            if (first_il < 0) { first_il = l2; first_id = F->plan_id_cap[off + e];
                                                                first_key = P2.key[(size_t) slot]; }
                                        }
                                    }
                                }
                                if (bad > 0) {
                                    ++bad_rows;
                                    std::fprintf(stderr, "glm pipe: CUDA%d IDENTITY row %d: %d plan entr(ies) name a slot "
                                                         "holding another expert; first at layer %d (id %d, slot holds %d)\n",
                                                 dev_, r2, bad, first_il, first_id, first_key);
                                }
                            }
                            std::fprintf(stderr, "glm pipe: CUDA%d IDENTITY rows with a mismatched plan: %d of %d\n", dev_,
                                         bad_rows, F->plan_cap_rows);
                        }
                    }
                }
                const uint64_t fb = F->md_fallbacks.load(std::memory_order_relaxed);
                if (fb > 0)
                    std::fprintf(stderr, "glm pipe: CUDA%d per-row calls fell back to the GLOBAL MoeDev %llu time(s), "
                                         "first at layer %d\n", dev_, (unsigned long long) fb,
                                 F->md_fallback_first_il.load(std::memory_order_relaxed));
                const uint64_t fv = F->floor_violations.load(std::memory_order_relaxed);
                const uint64_t ss = F->spare_steals.load(std::memory_order_relaxed);
                if (fv > 0 || ss > 0)
                    std::fprintf(stderr, "glm pipe: CUDA%d floor-bearing mutations while the rotor ran: %llu slot_of, "
                                         "%llu spare steals (a trailing row's slot was re-pointed or its spare taken)\n",
                                 dev_, (unsigned long long) fv, (unsigned long long) ss);
                const uint64_t busy = F->pipe->layer_busy.load(std::memory_order_relaxed);
                if (busy > 0 && !told.exchange(true))
                    std::fprintf(stderr, "glm pipe: CUDA%d same-layer collision avoided %llu time(s) - the spare "
                                         "claims were being shared\n", dev_, (unsigned long long) busy);
            }
            // TRANSFER: every launched layer that had experts to move has had its fetch issued by rows_layers (the
            // device-side moe_wait_fetch on the slot's stream); this attributes it.  RETIRE: one per row per step.
            // transfers_done is NOT credited here: rows_layers issues the fetch on the compute stream, which is the
            // fused path's work.  A counter nothing increments is honest; one that borrows another path is not.
            F->pipe->tokens_retired.fetch_add((uint64_t) n, std::memory_order_relaxed);
            F->pipe->watch_active.store(0, std::memory_order_release);
        }
        F->pipe_active.store(false, std::memory_order_release);   // the step's layers are done: the fused rule resumes
        F->pipe_floor.store(-1, std::memory_order_release);
        // rows_head runs on the engine's stream: every row's last layer has to be in before it reads their states.
        // BOUNDED: this used to be a bare cudaStreamSynchronize, i.e. an unbounded host wait between the last layer and
        // rows_head - a kernel still spinning at the end of the step blocked it forever and said nothing, which is why
        // step 2 never entered the rotor and no detector fired.
        {
            // THE BOUND IS A KNOB (2026-10-30): 20 s is shorter than the wait kernel's own ~60 s ceiling, so a step that
            // would have finished after its waiter gave up reads as a wedge.  STRATA_GLM_ROW_DRAIN_SEC settles which of the
            // two a stall is - an unanswered request (the kernel terminates, late, with a bad answer) or a true deadlock
            // (it never does).  A drain that takes more than 2 s now says so, with the time.
            static const int drain_s = [] {
                const char* e = getenv("STRATA_GLM_ROW_DRAIN_SEC");
                const int v = e != nullptr ? std::atoi(e) : 20;
                return v > 0 ? v : 20;
            }();
            const auto t0 = std::chrono::steady_clock::now();
            for (;;) {
                // BOTH STREAMS (2026-10-30).  This used to poll only cs2, so EVERY stall was reported as "the row lane" -
                // and "the wedge is on row 1's stream" has been in every error line today on the strength of where this
                // loop happens to look, not where the block is.  Row 0's work runs on cs, and a stall there is invisible
                // to a cs2-only query.
                const cudaError_t qr = cudaStreamQuery(F->cs2);   // the row lane: row 1
                const cudaError_t qe = cudaStreamQuery(F->cs);    // the engine stream: row 0
                if (qr == cudaSuccess && qe == cudaSuccess) {
                    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    if (el > 2.0)
                        std::fprintf(stderr,
                                     "glm pipe: CUDA%d the lanes took %.1f s to drain after the last layer (bound %d "
                                     "s) - the waiter finished late, it did not deadlock\n",
                                     dev_, el, drain_s);
                    break;
                }
                if (qr != cudaErrorNotReady && qr != cudaSuccess) {
                    (void) cudaGetLastError();
                    err = std::string("glm rows: row lane (cs2): ") + cudaGetErrorString(qr);
                    F->pipe_active.store(false, std::memory_order_release);
                    return false;
                }
                if (qe != cudaErrorNotReady && qe != cudaSuccess) {
                    (void) cudaGetLastError();
                    err = std::string("glm rows: engine stream (cs): ") + cudaGetErrorString(qe);
                    F->pipe_active.store(false, std::memory_order_release);
                    return false;
                }
                (void) cudaGetLastError();
                if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(drain_s)) {
                    // NAME THE LAYER.  "a kernel from this step is still running" is the one thing the log could not
                    // localise, and the step number wanders (5, then 2, then 4), which is the signature of a
                    // data-dependent wedge rather than a wiring fault.  ev_layer[il] is recorded after every layer run,
                    // so the lowest event that has not completed IS the stuck layer, and each row's cursor says which
                    // row's stream it belongs to.
                    std::string why = "glm rows: the row lane did not drain within " + std::to_string(drain_s) +
                                      " s after the last layer";
                    int stuck = -1;
                    for (int il2 = l0_; il2 < l1_; ++il2)
                        if (il2 < (int) F->ev_layer.size() && F->ev_layer[(size_t) il2] != nullptr) {
                            const cudaError_t q2 = cudaEventQuery(F->ev_layer[(size_t) il2]);
                            if (q2 == cudaErrorNotReady) {
                                stuck = il2;
                                break;
                            }
                            (void) cudaGetLastError();
                        }
                    // EVERY STREAM, not just the rows'.  Both cs and cs2 were NotReady in the first two boots that
                    // reported this, which no single row's own work explains: a shared dependency - the prefetch lanes or
                    // the copy lane - would hold both, and then the rows' streams are victims rather than culprits.
                    why += "; NotReady:";
                    {
                        struct SName { cudaStream_t s; const char* n; };
                        const SName all[5] = {{F->cs, " cs(row0)"}, {F->cs2, " cs2(row1)"}, {F->ps, " ps(prefetch)"},
                                              {F->ps2, " ps2(prefetch2)"}, {F->copy, " copy"}};
                        for (int k2 = 0; k2 < 5; ++k2) {
                            if (all[k2].s == nullptr) continue;
                            const cudaError_t q4 = cudaStreamQuery(all[k2].s);
                            (void) cudaGetLastError();
                            if (q4 == cudaErrorNotReady) why += all[k2].n;
                        }
                    }
                    why += "; lowest incomplete layer " + (stuck >= 0 ? std::to_string(stuck) : std::string("none")) +
                           " of " + std::to_string(l0_) + ".." + std::to_string(l1_ - 1);
                    for (int r2 = 0; r2 < n; ++r2)
                        why += "; row " + std::to_string(r2) + " at il " + std::to_string(S->cur[r2].il) + " on cs" +
                               (((serial || (r2 & 1) == 0)) ? "" : "2");
                    why += ch_delta();
                    // THE WAIT WORDS (2026-10-30).  The waiter's release is not "the answer arrived" - it is
                    // *moe_wseq(d) CHANGING: moe_wait reads s_need = *moe_wseq(d), block 0 watches for the response and
                    // clears that word to 0, and every other block spins on while (*moe_wseq(d) == s_need).  So a waiter
                    // can sit forever with its answer lying in resp[ch] if the word it polls is not the word that gets
                    // cleared.  Reading wait_seq against resp[ch].seq separates exactly that.  Read WITHOUT a blocking
                    // sync (a host sync here is the deadlock the diagnostics note warns about): the copy goes on its own
                    // stream and is polled for at most 2 s, and the line says so if it could not complete.
                    {
                        static cudaStream_t ds = nullptr;
                        if (ds == nullptr) {
                            if (cudaStreamCreateWithFlags(&ds, cudaStreamNonBlocking) != cudaSuccess) {
                                (void) cudaGetLastError();
                                ds = nullptr;
                            }
                        }
                        unsigned int wq[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                        bool got = false;
                        if (ds != nullptr && F->md.wait_seq != nullptr) {
                            cudaMemcpyAsync(wq, F->md.wait_seq, sizeof(wq), cudaMemcpyDeviceToHost, ds);
                            for (int t = 0; t < 2000; ++t) {
                                const cudaError_t q3 = cudaStreamQuery(ds);
                                if (q3 == cudaSuccess) {
                                    got = true;
                                    break;
                                }
                                (void) cudaGetLastError();
                                if (q3 != cudaErrorNotReady) break;
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            }
                        }
                        if (stuck >= 0) {
                        const int fr = stuck < (int) S->ev_fill_row.size() ? S->ev_fill_row[(size_t) stuck] : -1;
                        const int lr = stuck < (int) S->ev_layer_row.size() ? S->ev_layer_row[(size_t) stuck] : -1;
                        const bool gen_ok = stuck < (int) F->ev_fill_gen.size() &&
                                            F->ev_fill_gen[(size_t) stuck] == F->step_seq;
                        why += "; layer " + std::to_string(stuck) + " ev_fill recorded by row " + std::to_string(fr) +
                               (gen_ok ? " (generation current)" : " (GENERATION STALE - nothing recorded it this step)") +
                               ", ev_layer recorded by row " + std::to_string(lr);
                    }
                    why += "; wait_seq";
                        if (got)
                            for (int c2 = 0; c2 < 8; ++c2)
                                why += " " + std::to_string(c2) + "=" + std::to_string(wq[c2]);
                        else
                            why += " (unreadable without a blocking sync)";
                        why += "; resp.seq";
                        for (int c2 = 0; c2 < 8; ++c2)
                            why += " " + std::to_string(c2) + "=" +
                                   std::to_string(F->resp_h != nullptr ? (unsigned) F->resp_h[c2].seq : 0u);
                    }
                    err = why;
                    F->pipe_active.store(false, std::memory_order_release);
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    } else if (pipe_mode >= 2 && n >= 3) {
        // KNOWN-WRONG: the fused design double-buffers by layer PARITY (pf_buf[il & 1],
        // pf_n_buf[il & 1]) and by row parity for the window lanes.  With S = n - 1 the spread at n = 3 is k, k-1, k-2,
        // and k and k-2 share parity - two rows would write one buffer.  Refused here with the reason.
        static std::atomic<bool> told3{false};
        if (!told3.exchange(true))
            std::fprintf(stderr, "glm rows: CUDA%d the rotation is defined only for n = 2: at n >= 3 rows sit at k and "
                                 "k-2, which share the parity pf_buf/win_lane double-buffer on - running the layers one "
                                 "at a time\n", dev_);
        for (int il = l0_; il < l1_; ++il)
            if (!rows_layers(rw, n, err, il, il + 1)) return false;
    } else if (pipe_mode >= 2) {
        // the per-row DSA score/selection buffers only exist with STRATA_GLM_ROWS_ATTN=1; without them the rows would
        // share F->score / F->cells, so the pipeline refuses rather than corrupting (said once)
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            // NAME EVERY DECLINED CONDITION.  This message used to report only the aq case, so a null cs2 or a wrong
            // row count read as 'set STRATA_GLM_ROWS_ATTN=1' - the same one-message-for-two-faults error that cost a
            // day on the router, in the gate rather than the kernel.
            std::fprintf(stderr, "glm rows: CUDA%d the rotation declined (n=%d, cs2=%s, aq=%s) - running the layers one "
                                 "at a time\n",
                         dev_, n, F->cs2 != nullptr ? "set" : "NULL", S->aq != nullptr ? "set" : "NULL");
        for (int il = l0_; il < l1_; ++il)
            if (!rows_layers(rw, n, err, il, il + 1)) return false;
    } else if (pipe_mode >= 1) {
        for (int il = l0_; il < l1_; ++il)
            if (!rows_layers(rw, n, err, il, il + 1)) return false;
    } else if (!rows_layers(rw, n, err)) {
        return false;
    }
    if (reslog)
        for (int i = 0; i < n; ++i) res_post[i] = res_hash(rw[i].res);   // what the 45 layers left there
    if (!rows_head(rw, sps, n, err)) return false;
    cudaMemcpyAsync(S->tok_h, S->tok, (size_t) n * sizeof(int), cudaMemcpyDeviceToHost, F->cs);
    cudaEventRecord(S->ev_done, F->cs);
    // the step back (the service threads answer the routes' misses meanwhile)
    if (!glmfast::wait_event(S->ev_done, "glm rows step", err) || !F->route_ok(err)) {
        err += ch_delta();   // the NON-FINITE face of the unanswered request carries the numbers too
        return false;
    }
    if (gf::launch_errors() > 0) {
        err = "glm rows: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)" + ch_delta();
        return false;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm rows: ") + cudaGetErrorString(e);
        return false;
    }
    for (int i = 0; i < n; ++i) {
        const int tok = S->tok_h[i];
        tokens_out[i] = tok;
        if (rw[i].seq == cur_seq_) {
            ++pos_;
            last_tok_ = tok;
        } else {
            SeqCtx& c = seq_[(size_t) rw[i].seq];
            ++c.pos;
            c.last_tok = tok;
        }
    }
    // RAW TOKEN IDs per row per step (2026-10-30).  Text cannot separate three very different faults behind a short or
    // empty reply: the row sampled EOS, it sampled a valid token the server then discarded, or tokens_out[r] was never
    // written and holds a stale value.  Integers separate them, and they also show whether a wrong opening is one wrong
    // token or a correct token followed by divergence.  The position is printed with the id because the two together are
    // the evidence: a row sampling a plausible fragment of the prompt's own subject, at a position that is not where the
    // prompt ended, is generating from the wrong cursor rather than from noise.  Gated on n > 1 so the n == 1 baseline -
    // the thing this is compared against - stays silent.
    static const bool toklog = getenv("STRATA_GLM_TOK_ID_LOG") != nullptr;   // ...and on demand for the n == 1 baseline
    if (n > 1 || toklog) {
        std::string line = "glm rows: CUDA" + std::to_string(dev_) + " step " + std::to_string(F->step_seq) +
                           " token ids | deferrals occ " + std::to_string(occ_defers) + " reo " +
                           std::to_string(reo_defers) + " spread " + std::to_string(spread_max);
        for (int i = 0; i < n; ++i) {
            char hp[24], he[24], ho[24];
            std::snprintf(hp, sizeof hp, "%016llx", res_pre[i]);
            std::snprintf(he, sizeof he, "%016llx", res_emb[i]);
            std::snprintf(ho, sizeof ho, "%016llx", res_post[i]);
            line += " | row " + std::to_string(i) + " seq " + std::to_string(rw[i].seq) + " pos " +
                    std::to_string(rw[i].p) + " id " + std::to_string(tokens_out[i]) + " res " + hp + " " + he +
                    " " + ho;
        }
        std::fprintf(stderr, "%s\n", line.c_str());
    }
    return true;
}

bool Glm5Model::split_rows_ok() const {
    if (n_parts_ <= 1 || seq_.empty()) return false;
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ == nullptr || m->rows_ == nullptr) return false;
    return true;
}

bool Glm5Model::multi_issue_rows(const int* seqs, const int32_t* tokens,
                                 const strata::kernels::SamplerParams* sps, int n, std::string& err) {
    if (!split_rows_ok() || n < 1 || n > rows_max()) {
        err = "glm rows: the split batch is not available here (every part's row arena, 1.." +
              std::to_string(rows_max()) + " rows)";
        return false;
    }
    if (seq_[0].hop_h == nullptr && !seq_pipe_ready(err)) return false;

    struct ProfOff {
        FastState* fast[kMaxParts];
        bool was[kMaxParts];
        int n = 0;
        ~ProfOff() {
            for (int i = n; i-- > 0;) fast[i]->prof_on = was[i];
        }
    } prof_off;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        prof_off.fast[prof_off.n] = m->fast_;
        prof_off.was[prof_off.n] = m->fast_->prof_on;
        m->fast_->prof_on = false;
        ++prof_off.n;
    }

    RowSeq cur[gf::kMaxMvNt];
    if (!rows_bind(seqs, n, cur, err) || !rows_embed(cur, tokens, n, err) || !rows_layers(cur, n, err))
        return false;
    int64_t p0[gf::kMaxMvNt];
    for (int i = 0; i < n; ++i) p0[i] = cur[i].p;

    Glm5Model* tail = this;
    const size_t hop = (size_t) g_.hc * g_.n_embd * sizeof(float);
    for (Glm5Model* B = split_next_.get(); B != nullptr; B = B->split_next_.get()) {
        RowSeq nxt[gf::kMaxMvNt];
        if (!B->rows_bind(seqs, n, nxt, err)) return false;
        for (int i = 0; i < n; ++i)
            if (nxt[i].p != p0[i]) {
                err = "glm rows: sequence " + std::to_string(seqs[i]) + " is at a different position on CUDA" +
                      std::to_string(B->dev_);
                return false;
            }
        cudaSetDevice(tail->dev_);
        for (int i = 0; i < n; ++i) {
            SeqCtx& ca = tail->seq_[(size_t) seqs[i]];
            cudaMemcpyAsync(ca.hop_h, cur[i].state, hop, cudaMemcpyDeviceToHost, tail->fast_->cs);
            cudaEventRecord(ca.ev_hop, tail->fast_->cs);
        }
        cudaSetDevice(B->dev_);
        for (int i = 0; i < n; ++i) {
            SeqCtx& ca = tail->seq_[(size_t) seqs[i]];
            cudaStreamWaitEvent(B->fast_->cs, ca.ev_hop, 0);
            cudaMemcpyAsync(nxt[i].state, ca.hop_h, hop, cudaMemcpyHostToDevice, B->fast_->cs);
        }
        if (!B->rows_layers(nxt, n, err)) return false;
        for (int i = 0; i < n; ++i) cur[i] = nxt[i];
        tail = B;
    }

    if (!tail->rows_head(cur, sps, n, err)) return false;
    cudaSetDevice(tail->dev_);
    for (int i = 0; i < n; ++i) {
        SeqCtx& ct = tail->seq_[(size_t) seqs[i]];
        cudaMemcpyAsync(ct.tok_h, tail->rows_->tok + i, sizeof(int), cudaMemcpyDeviceToHost, tail->fast_->cs);
        cudaEventRecord(ct.ev_done, tail->fast_->cs);
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        for (int i = 0; i < n; ++i) {
            if (seqs[i] == m->cur_seq_)
                ++m->pos_;
            else
                ++m->seq_[(size_t) seqs[i]].pos;
        }
    cudaSetDevice(dev_);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm rows: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

}  // namespace strata::core
