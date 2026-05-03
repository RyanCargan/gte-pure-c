/*
 * GTE-Small Embedding Library Implementation
 * Optimized with SIMDe (NEON)
 */

#include "gte.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdint.h>
#include "simde/arm/neon.h" // IWYU pragma: keep

#define GTE_MAGIC "GTE4"
#define GTE_LAYER_NORM_EPS 1e-12f
#define VOCAB_HASH_SIZE 40009
#define QK4_0 32

/* Special token IDs */
#define TOKEN_UNK 100
#define TOKEN_CLS 101
#define TOKEN_SEP 102

typedef struct {
    float scale;
    uint8_t qs[16];
} block_q4_0;

typedef struct {
    char *word;
    int id;
} vocab_entry;

typedef struct {
    block_q4_0 *query_weight, *key_weight, *value_weight, *attn_output_weight;
    float *query_bias, *key_bias, *value_bias, *attn_output_bias;
    float *attn_ln_weight, *attn_ln_bias;
    block_q4_0 *ffn_inter_weight, *ffn_output_weight;
    float *ffn_inter_bias, *ffn_output_bias, *ffn_ln_weight, *ffn_ln_bias;
} layer_weights;

struct gte_ctx {
    int vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_seq_len, head_dim;
    char **vocab;
    vocab_entry *vocab_hash;
    block_q4_0 *token_embeddings, *position_embeddings, *token_type_embeddings;
    float *embed_ln_weight, *embed_ln_bias;
    layer_weights *layers;
    block_q4_0 *pooler_weight;
    float *pooler_bias;
    float *hidden_states, *attn_scores, *q_proj, *k_proj, *v_proj, *attn_output, *ffn_hidden, *temp_hidden;
};

/* --- Helpers --- */

static void* gte_malloc(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, 16, size) != 0) return malloc(size);
    return ptr;
}

static inline float vsum(simde_float32x4_t v) {
    float t[4];
    simde_vst1q_f32(t, v);
    return t[0] + t[1] + t[2] + t[3];
}

/* --- Optimized Matrix Math --- */

static void linear_q4_0(float *y, const float *x, const block_q4_0 *W, const float *b, int seq_len, int in_dim, int out_dim) {
    int num_blocks = in_dim / QK4_0;
    simde_uint8x16_t mask = simde_vdupq_n_u8(0x0F);
    simde_int8x16_t s8 = simde_vdupq_n_s8(8);

    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + (intptr_t)s * in_dim;
        float *y_row = y + (intptr_t)s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            simde_float32x4_t row_acc = simde_vdupq_n_f32(0.0f);
            const block_q4_0 *w_row = W + (intptr_t)o * num_blocks;
            for (int nb = 0; nb < num_blocks; nb++) {
                const float *xb = x_row + nb * QK4_0;
                simde_uint8x16_t q = simde_vld1q_u8(w_row[nb].qs);
                simde_int8x16_t v0 = simde_vsubq_s8(simde_vreinterpretq_s8_u8(simde_vandq_u8(q, mask)), s8);
                simde_int8x16_t v1 = simde_vsubq_s8(simde_vreinterpretq_s8_u8(simde_vshrq_n_u8(q, 4)), s8);

                simde_float32x4_t xv0 = simde_vld1q_f32(xb);
                simde_float32x4_t xv1 = simde_vld1q_f32(xb+4);
                simde_float32x4_t xv2 = simde_vld1q_f32(xb+8);
                simde_float32x4_t xv3 = simde_vld1q_f32(xb+12);
                simde_float32x4_t xv4 = simde_vld1q_f32(xb+16);
                simde_float32x4_t xv5 = simde_vld1q_f32(xb+20);
                simde_float32x4_t xv6 = simde_vld1q_f32(xb+24);
                simde_float32x4_t xv7 = simde_vld1q_f32(xb+28);

                simde_float32x4_t b_acc = simde_vmulq_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_low_s8(v0))))), xv0);
                b_acc = simde_vmlaq_f32(b_acc, simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_low_s8(v0))))), xv1);
                b_acc = simde_vmlaq_f32(b_acc, simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_high_s8(v0))))), xv2);
                b_acc = simde_vmlaq_f32(b_acc, simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_high_s8(v0))))), xv3);

                b_acc = simde_vmlaq_f32(b_acc, simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_low_s8(v1))))), xv4);
                b_acc = simde_vmlaq_f32(b_acc, simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_low_s8(v1))))), xv5);
                b_acc = simde_vmlaq_f32(b_acc, simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_high_s8(v1))))), xv6);
                b_acc = simde_vmlaq_f32(b_acc, simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_high_s8(v1))))), xv7);

                row_acc = simde_vmlaq_n_f32(row_acc, b_acc, w_row[nb].scale);
            }
            y_row[o] = vsum(row_acc) + (b ? b[o] : 0.0f);
        }
    }
}

static void layer_norm(float *out, const float *x, const float *gamma, const float *beta, int seq_len, int hidden_size) {
    for (int s = 0; s < seq_len; s++) {
        const float *xr = x + (intptr_t)s * hidden_size;
        float *orr = out + (intptr_t)s * hidden_size;
        float m = 0, v = 0;
        for (int i = 0; i < hidden_size; i++) m += xr[i];
        m /= hidden_size;
        for (int i = 0; i < hidden_size; i++) { float d = xr[i]-m; v += d*d; }
        float inv = 1.0f / sqrtf(v/hidden_size + GTE_LAYER_NORM_EPS);
        for (int i = 0; i < hidden_size; i++) orr[i] = gamma[i] * (xr[i]-m) * inv + beta[i];
    }
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i]-mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* --- Transformer Forward --- */

static void transformer_forward(gte_ctx *ctx, const int *token_ids, int seq_len, const int *attn_mask) {
    int h = ctx->hidden_size, nb = h / QK4_0;
    simde_uint8x16_t mask = simde_vdupq_n_u8(0x0F);
    simde_int8x16_t s8 = simde_vdupq_n_s8(8);

    for (int s = 0; s < seq_len; s++) {
        float *hr = ctx->hidden_states + (intptr_t)s * h;
        const block_q4_0 *tb = ctx->token_embeddings + (intptr_t)token_ids[s] * nb;
        const block_q4_0 *pb = ctx->position_embeddings + (intptr_t)s * nb;
        const block_q4_0 *yb = ctx->token_type_embeddings;

        for (int b = 0; b < nb; b++) {
            simde_uint8x16_t qt = simde_vld1q_u8(tb[b].qs);
            simde_uint8x16_t qp = simde_vld1q_u8(pb[b].qs);
            simde_uint8x16_t qy = simde_vld1q_u8(yb[b].qs);

            simde_int8x16_t vt0 = simde_vsubq_s8(simde_vreinterpretq_s8_u8(simde_vandq_u8(qt, mask)), s8);
            simde_int8x16_t vt1 = simde_vsubq_s8(simde_vreinterpretq_s8_u8(simde_vshrq_n_u8(qt, 4)), s8);
            simde_int8x16_t vp0 = simde_vsubq_s8(simde_vreinterpretq_s8_u8(simde_vandq_u8(qp, mask)), s8);
            simde_int8x16_t vp1 = simde_vsubq_s8(simde_vreinterpretq_s8_u8(simde_vshrq_n_u8(qp, 4)), s8);
            simde_int8x16_t vy0 = simde_vsubq_s8(simde_vreinterpretq_s8_u8(simde_vandq_u8(qy, mask)), s8);
            simde_int8x16_t vy1 = simde_vsubq_s8(simde_vreinterpretq_s8_u8(simde_vshrq_n_u8(qy, 4)), s8);

            simde_float32x4_t f_tl = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_low_s8(vt0))))), tb[b].scale);
            simde_float32x4_t f_pl = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_low_s8(vp0))))), pb[b].scale);
            simde_float32x4_t f_yl = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_low_s8(vy0))))), yb[b].scale);
            simde_vst1q_f32(hr + b*32 + 0, simde_vaddq_f32(f_tl, simde_vaddq_f32(f_pl, f_yl)));

            simde_float32x4_t f_th = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_low_s8(vt0))))), tb[b].scale);
            simde_float32x4_t f_ph = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_low_s8(vp0))))), pb[b].scale);
            simde_float32x4_t f_yh = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_low_s8(vy0))))), yb[b].scale);
            simde_vst1q_f32(hr + b*32 + 4, simde_vaddq_f32(f_th, simde_vaddq_f32(f_ph, f_yh)));

            f_tl = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_high_s8(vt0))))), tb[b].scale);
            f_pl = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_high_s8(vp0))))), pb[b].scale);
            f_yl = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_high_s8(vy0))))), yb[b].scale);
            simde_vst1q_f32(hr + b*32 + 8, simde_vaddq_f32(f_tl, simde_vaddq_f32(f_pl, f_yl)));

            f_th = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_high_s8(vt0))))), tb[b].scale);
            f_ph = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_high_s8(vp0))))), pb[b].scale);
            f_yh = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_high_s8(vy0))))), yb[b].scale);
            simde_vst1q_f32(hr + b*32 + 12, simde_vaddq_f32(f_th, simde_vaddq_f32(f_ph, f_yh)));

            f_tl = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_low_s8(vt1))))), tb[b].scale);
            f_pl = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_low_s8(vp1))))), pb[b].scale);
            f_yl = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_low_s8(vy1))))), yb[b].scale);
            simde_vst1q_f32(hr + b*32 + 16, simde_vaddq_f32(f_tl, simde_vaddq_f32(f_pl, f_yl)));

            f_th = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_low_s8(vt1))))), tb[b].scale);
            f_ph = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_low_s8(vp1))))), pb[b].scale);
            f_yh = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_low_s8(vy1))))), yb[b].scale);
            simde_vst1q_f32(hr + b*32 + 20, simde_vaddq_f32(f_th, simde_vaddq_f32(f_ph, f_yh)));

            f_tl = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_high_s8(vt1))))), tb[b].scale);
            f_pl = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_high_s8(vp1))))), pb[b].scale);
            f_yl = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_low_s16(simde_vmovl_s8(simde_vget_high_s8(vy1))))), yb[b].scale);
            simde_vst1q_f32(hr + b*32 + 24, simde_vaddq_f32(f_tl, simde_vaddq_f32(f_pl, f_yl)));

            f_th = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_high_s8(vt1))))), tb[b].scale);
            f_ph = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_high_s8(vp1))))), pb[b].scale);
            f_yh = simde_vmulq_n_f32(simde_vcvtq_f32_s32(simde_vmovl_s16(simde_vget_high_s16(simde_vmovl_s8(simde_vget_high_s8(vy1))))), yb[b].scale);
            simde_vst1q_f32(hr + b*32 + 28, simde_vaddq_f32(f_th, simde_vaddq_f32(f_ph, f_yh)));
        }
    }

    layer_norm(ctx->hidden_states, ctx->hidden_states, ctx->embed_ln_weight, ctx->embed_ln_bias, seq_len, h);

    for (int l = 0; l < ctx->num_layers; l++) {
        layer_weights *lw = &ctx->layers[l];

        linear_q4_0(ctx->q_proj, ctx->hidden_states, lw->query_weight, lw->query_bias, seq_len, h, h);
        linear_q4_0(ctx->k_proj, ctx->hidden_states, lw->key_weight, lw->key_bias, seq_len, h, h);
        linear_q4_0(ctx->v_proj, ctx->hidden_states, lw->value_weight, lw->value_bias, seq_len, h, h);

        float sc = 1.0f / sqrtf((float)ctx->head_dim);
        int head_dim = ctx->head_dim;

        for (int hd = 0; hd < ctx->num_heads; hd++) {
            for (int i = 0; i < seq_len; i++) {
                const float *qi = ctx->q_proj + (intptr_t)i * h + hd * head_dim;
                float *scores = ctx->attn_scores + (intptr_t)hd * seq_len * seq_len + (intptr_t)i * seq_len;

                for (int j = 0; j < seq_len; j++) {
                    const float *kj = ctx->k_proj + (intptr_t)j * h + hd * head_dim;
                    simde_float32x4_t acc = simde_vdupq_n_f32(0.0f);
                    for (int d = 0; d < head_dim; d += 4) {
                        acc = simde_vmlaq_f32(acc, simde_vld1q_f32(qi + d), simde_vld1q_f32(kj + d));
                    }
                    float sum = vsum(acc) * sc;
                    if (attn_mask && !attn_mask[j]) sum = -10000.0f;
                    scores[j] = sum;
                }
                softmax(scores, seq_len);
            }

            for (int i = 0; i < seq_len; i++) {
                const float *scores = ctx->attn_scores + (intptr_t)hd * seq_len * seq_len + (intptr_t)i * seq_len;
                float *out_i = ctx->attn_output + (intptr_t)i * h + hd * head_dim;

                simde_float32x4_t accs[8];
                for (int k = 0; k < 8; k++) accs[k] = simde_vdupq_n_f32(0.0f);

                for (int j = 0; j < seq_len; j++) {
                    simde_float32x4_t s_vec = simde_vdupq_n_f32(scores[j]);
                    const float *vj = ctx->v_proj + (intptr_t)j * h + hd * head_dim;
                    for (int k = 0; k < 8; k++) {
                        accs[k] = simde_vmlaq_f32(accs[k], s_vec, simde_vld1q_f32(vj + k * 4));
                    }
                }
                for (int k = 0; k < 8; k++) simde_vst1q_f32(out_i + k * 4, accs[k]);
            }
        }

        linear_q4_0(ctx->temp_hidden, ctx->attn_output, lw->attn_output_weight, lw->attn_output_bias, seq_len, h, h);

        for (int i = 0; i < seq_len * h; i += 4) {
            simde_vst1q_f32(ctx->temp_hidden + i, simde_vaddq_f32(simde_vld1q_f32(ctx->temp_hidden + i), simde_vld1q_f32(ctx->hidden_states + i)));
        }
        layer_norm(ctx->hidden_states, ctx->temp_hidden, lw->attn_ln_weight, lw->attn_ln_bias, seq_len, h);

        linear_q4_0(ctx->ffn_hidden, ctx->hidden_states, lw->ffn_inter_weight, lw->ffn_inter_bias, seq_len, h, ctx->intermediate_size);
        for (int i = 0; i < seq_len * ctx->intermediate_size; i++) {
            float v = ctx->ffn_hidden[i];
            ctx->ffn_hidden[i] = 0.5f * v * (1.0f + tanhf(0.797884f * (v + 0.044715f * v * v * v)));
        }

        linear_q4_0(ctx->temp_hidden, ctx->ffn_hidden, lw->ffn_output_weight, lw->ffn_output_bias, seq_len, ctx->intermediate_size, h);
        for (int i = 0; i < seq_len * h; i += 4) {
            simde_vst1q_f32(ctx->temp_hidden + i, simde_vaddq_f32(simde_vld1q_f32(ctx->temp_hidden + i), simde_vld1q_f32(ctx->hidden_states + i)));
        }
        layer_norm(ctx->hidden_states, ctx->temp_hidden, lw->ffn_ln_weight, lw->ffn_ln_bias, seq_len, h);
    }
}

/* --- Tokenizer & Utility --- */

static unsigned int hash_s(const char *s) {
    unsigned int h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return h;
}

static void v_ins(vocab_entry *t, const char *w, int id) {
    unsigned int h = hash_s(w) % VOCAB_HASH_SIZE;
    while(t[h].word) h = (h + 1) % VOCAB_HASH_SIZE;
    t[h].word = strdup(w);
    t[h].id = id;
}

static int v_lk(vocab_entry *t, const char *w) {
    unsigned int h = hash_s(w) % VOCAB_HASH_SIZE;
    while(t[h].word) {
        if(!strcmp(t[h].word, w)) return t[h].id;
        h = (h + 1) % VOCAB_HASH_SIZE;
    }
    return -1;
}

static char **b_tok(const char *t, int *n) {
    int cap = 64, c = 0;
    char **toks = malloc(cap * sizeof(char*));
    const char *p = t;
    while(*p) {
        while(*p && isspace(*p)) p++;
        if(!*p) break;
        const char *s = p;
        if(ispunct(*p)) p++;
        else while(*p && !isspace(*p) && !ispunct(*p)) p++;
        int l = p - s;
        toks[c] = malloc(l + 1);
        for(int i = 0; i < l; i++) toks[c][i] = tolower(s[i]);
        toks[c][l] = 0;
        if(++c >= cap) {
            cap *= 2;
            toks = realloc(toks, cap * sizeof(char*));
        }
    }
    *n = c;
    return toks;
}

/* --- Loader --- */

gte_ctx *gte_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if(!f) return NULL;
    char m[4];
    if(fread(m, 1, 4, f) != 4 || memcmp(m, GTE_MAGIC, 4) != 0) { fclose(f); return NULL; }

    gte_ctx *c = calloc(1, sizeof(gte_ctx));
    if(fread(&c->vocab_size, 4, 1, f) != 1) goto error;
    if(fread(&c->hidden_size, 4, 1, f) != 1) goto error;
    if(fread(&c->num_layers, 4, 1, f) != 1) goto error;
    if(fread(&c->num_heads, 4, 1, f) != 1) goto error;
    if(fread(&c->intermediate_size, 4, 1, f) != 1) goto error;
    if(fread(&c->max_seq_len, 4, 1, f) != 1) goto error;

    c->head_dim = c->hidden_size / c->num_heads;
    c->vocab = malloc(c->vocab_size * sizeof(char*));
    c->vocab_hash = calloc(VOCAB_HASH_SIZE, sizeof(vocab_entry));

    for(int i = 0; i < c->vocab_size; i++) {
        uint16_t l;
        if(fread(&l, 2, 1, f) != 1) goto error;
        c->vocab[i] = malloc(l + 1);
        if(fread(c->vocab[i], 1, l, f) != (size_t)l) goto error;
        c->vocab[i][l] = 0;
        v_ins(c->vocab_hash, c->vocab[i], i);
    }

    #define RD_B(ptr, n) { ptr = gte_malloc((n) * sizeof(block_q4_0)); if(fread(ptr, sizeof(block_q4_0), n, f) != (size_t)n) goto error; }
    #define RD_F(ptr, n) { ptr = gte_malloc((n) * sizeof(float)); if(fread(ptr, 4, n, f) != (size_t)n) goto error; }

    RD_B(c->token_embeddings, (intptr_t)c->vocab_size * c->hidden_size / 32);
    RD_B(c->position_embeddings, (intptr_t)c->max_seq_len * c->hidden_size / 32);
    RD_B(c->token_type_embeddings, (intptr_t)2 * c->hidden_size / 32);
    RD_F(c->embed_ln_weight, c->hidden_size);
    RD_F(c->embed_ln_bias, c->hidden_size);

    c->layers = calloc(c->num_layers, sizeof(layer_weights));
    for(int i = 0; i < c->num_layers; i++) {
        layer_weights *l = &c->layers[i];
        RD_B(l->query_weight, (intptr_t)c->hidden_size * c->hidden_size / 32); RD_F(l->query_bias, c->hidden_size);
        RD_B(l->key_weight, (intptr_t)c->hidden_size * c->hidden_size / 32); RD_F(l->key_bias, c->hidden_size);
        RD_B(l->value_weight, (intptr_t)c->hidden_size * c->hidden_size / 32); RD_F(l->value_bias, c->hidden_size);
        RD_B(l->attn_output_weight, (intptr_t)c->hidden_size * c->hidden_size / 32); RD_F(l->attn_output_bias, c->hidden_size);
        RD_F(l->attn_ln_weight, c->hidden_size); RD_F(l->attn_ln_bias, c->hidden_size);
        RD_B(l->ffn_inter_weight, (intptr_t)c->intermediate_size * c->hidden_size / 32); RD_F(l->ffn_inter_bias, c->intermediate_size);
        RD_B(l->ffn_output_weight, (intptr_t)c->hidden_size * c->intermediate_size / 32); RD_F(l->ffn_output_bias, c->hidden_size);
        RD_F(l->ffn_ln_weight, c->hidden_size); RD_F(l->ffn_ln_bias, c->hidden_size);
    }
    RD_B(c->pooler_weight, (intptr_t)c->hidden_size * c->hidden_size / 32); RD_F(c->pooler_bias, c->hidden_size);

    c->hidden_states = gte_malloc((intptr_t)c->max_seq_len * c->hidden_size * 4);
    c->attn_scores = gte_malloc((intptr_t)c->num_heads * c->max_seq_len * c->max_seq_len * 4);
    c->q_proj = gte_malloc((intptr_t)c->max_seq_len * c->hidden_size * 4);
    c->k_proj = gte_malloc((intptr_t)c->max_seq_len * c->hidden_size * 4);
    c->v_proj = gte_malloc((intptr_t)c->max_seq_len * c->hidden_size * 4);
    c->attn_output = gte_malloc((intptr_t)c->max_seq_len * c->hidden_size * 4);
    c->ffn_hidden = gte_malloc((intptr_t)c->max_seq_len * c->intermediate_size * 4);
    c->temp_hidden = gte_malloc((intptr_t)c->max_seq_len * c->hidden_size * 4);

    fclose(f);
    return c;

error:
    fprintf(stderr, "gte_load: error reading model\n");
    fclose(f); gte_free(c); return NULL;
}

/* --- Public API --- */

float *gte_embed(gte_ctx *ctx, const char *text) {
    int nb, *ids = malloc(ctx->max_seq_len * 4), total = 0;
    char **bt = b_tok(text, &nb);
    ids[total++] = TOKEN_CLS;
    for(int i = 0; i < nb && total < ctx->max_seq_len - 1; i++) {
        int start = 0, len = strlen(bt[i]);
        while(start < len) {
            int end = len, fid = -1;
            while(start < end) {
                char cand[256]; int cl = 0;
                if(start > 0) { cand[cl++] = '#'; cand[cl++] = '#'; }
                for(int k = start; k < end && cl < 254; k++) cand[cl++] = bt[i][k];
                cand[cl] = 0;
                int id = v_lk(ctx->vocab_hash, cand);
                if(id >= 0) { fid = id; break; }
                end--;
            }
            if(fid < 0) { ids[total++] = TOKEN_UNK; start++; }
            else { ids[total++] = fid; start = end; }
            if(total >= ctx->max_seq_len - 1) break;
        }
        free(bt[i]);
    }
    free(bt);
    ids[total++] = TOKEN_SEP;

    int *mask = malloc((intptr_t)total * 4);
    for(int i = 0; i < total; i++) mask[i] = 1;

    transformer_forward(ctx, ids, total, mask);

    float *emb = malloc(ctx->hidden_size * 4);
    memset(emb, 0, ctx->hidden_size * 4);
    for(int i = 0; i < total; i++) {
        for(int d = 0; d < ctx->hidden_size; d++) emb[d] += ctx->hidden_states[i * ctx->hidden_size + d];
    }
    for(int d = 0; d < ctx->hidden_size; d++) emb[d] /= total;

    float norm = 0;
    for(int d = 0; d < ctx->hidden_size; d++) norm += emb[d] * emb[d];
    norm = sqrtf(norm);
    for(int d = 0; d < ctx->hidden_size; d++) emb[d] /= (norm > 1e-9f ? norm : 1.0f);

    free(ids); free(mask);
    return emb;
}

void gte_free(gte_ctx *ctx) {
    if (!ctx) return;
    if (ctx->vocab) { for (int i = 0; i < ctx->vocab_size; i++) free(ctx->vocab[i]); free(ctx->vocab); }
    if (ctx->vocab_hash) { for (int i = 0; i < VOCAB_HASH_SIZE; i++) free(ctx->vocab_hash[i].word); free(ctx->vocab_hash); }
    free(ctx->token_embeddings); free(ctx->position_embeddings); free(ctx->token_type_embeddings);
    free(ctx->embed_ln_weight); free(ctx->embed_ln_bias);
    if (ctx->layers) {
        for (int l = 0; l < ctx->num_layers; l++) {
            layer_weights *lw = &ctx->layers[l];
            free(lw->query_weight); free(lw->query_bias); free(lw->key_weight); free(lw->key_bias);
            free(lw->value_weight); free(lw->value_bias); free(lw->attn_output_weight); free(lw->attn_output_bias);
            free(lw->attn_ln_weight); free(lw->attn_ln_bias); free(lw->ffn_inter_weight); free(lw->ffn_inter_bias);
            free(lw->ffn_output_weight); free(lw->ffn_output_bias); free(lw->ffn_ln_weight); free(lw->ffn_ln_bias);
        }
        free(ctx->layers);
    }
    free(ctx->pooler_weight); free(ctx->pooler_bias);
    free(ctx->hidden_states); free(ctx->attn_scores); free(ctx->q_proj); free(ctx->k_proj);
    free(ctx->v_proj); free(ctx->attn_output); free(ctx->ffn_hidden); free(ctx->temp_hidden);
    free(ctx);
}

int gte_dim(gte_ctx *ctx) { return ctx ? ctx->hidden_size : 0; }
int gte_max_seq_len(gte_ctx *ctx) { return ctx ? ctx->max_seq_len : 0; }
float gte_cosine_similarity(const float *a, const float *b, int d) {
    float s = 0;
    for(int i = 0; i < d; i++) s += a[i] * b[i];
    return s;
}
