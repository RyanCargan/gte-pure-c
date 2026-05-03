/*
 * GTE-Small Embedding Library Implementation
 */

#include "gte.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdint.h>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#define GTE_MAGIC "GTE4"
#define GTE_LAYER_NORM_EPS 1e-12f
#define TOKEN_PAD 0
#define TOKEN_UNK 100
#define TOKEN_CLS 101
#define TOKEN_SEP 102
#define TOKEN_MASK 103
#define VOCAB_HASH_SIZE 40009
#define QK4_0 32

typedef struct {
    float scale;
    uint8_t qs[16];
} block_q4_0;

typedef struct {
    char *word;
    int id;
} vocab_entry;

typedef struct {
    block_q4_0 *query_weight;
    float *query_bias;
    block_q4_0 *key_weight;
    float *key_bias;
    block_q4_0 *value_weight;
    float *value_bias;
    block_q4_0 *attn_output_weight;
    float *attn_output_bias;
    float *attn_ln_weight;
    float *attn_ln_bias;
    block_q4_0 *ffn_inter_weight;
    float *ffn_inter_bias;
    block_q4_0 *ffn_output_weight;
    float *ffn_output_bias;
    float *ffn_ln_weight;
    float *ffn_ln_bias;
} layer_weights;

struct gte_ctx {
    int vocab_size;
    int hidden_size;
    int num_layers;
    int num_heads;
    int intermediate_size;
    int max_seq_len;      /* Hard limit from file */
    int current_max_len;  /* Configurable limit */
    int head_dim;
    char **vocab;
    vocab_entry *vocab_hash;
    block_q4_0 *token_embeddings;
    block_q4_0 *position_embeddings;
    block_q4_0 *token_type_embeddings;
    float *embed_ln_weight;
    float *embed_ln_bias;
    layer_weights *layers;
    block_q4_0 *pooler_weight;
    float *pooler_bias;
    float *hidden_states;
    float *attn_scores;
    float *q_proj;
    float *k_proj;
    float *v_proj;
    float *attn_output;
    float *ffn_hidden;
    float *temp_hidden;
};

/* --- Utilities --- */

static char *my_strdup(const char *s) {
    size_t len = strlen(s) + 1;
    char *dup = malloc(len);
    if (dup) strcpy(dup, s);
    return dup;
}

static unsigned int hash_string(const char *str) {
    unsigned int hash = 2166136261u;
    while (*str) {
        hash ^= (unsigned char)*str++;
        hash *= 16777619u;
    }
    return hash;
}

static void vocab_hash_insert(vocab_entry *table, const char *word, int id) {
    unsigned int h = hash_string(word) % VOCAB_HASH_SIZE;
    while (table[h].word != NULL) {
        if (strcmp(table[h].word, word) == 0) return;
        h = (h + 1) % VOCAB_HASH_SIZE;
    }
    table[h].word = my_strdup(word);
    table[h].id = id;
}

static int vocab_lookup(vocab_entry *table, const char *word) {
    unsigned int h = hash_string(word) % VOCAB_HASH_SIZE;
    while (table[h].word != NULL) {
        if (strcmp(table[h].word, word) == 0) return table[h].id;
        h = (h + 1) % VOCAB_HASH_SIZE;
    }
    return -1;
}

/* --- Math Ops --- */

static void linear_q4_0(float *y, const float *x, const block_q4_0 *W, const float *b,
                        int seq_len, int in_dim, int out_dim) {
    int num_blocks = in_dim / QK4_0;
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * in_dim;
        float *y_row = y + s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            float sum = b ? b[o] : 0.0f;
            const block_q4_0 *w_row = W + o * num_blocks;
            for (int nb = 0; nb < num_blocks; nb++) {
                const block_q4_0 *block = &w_row[nb];
                float scale = block->scale;
                const uint8_t *qs = block->qs;
                const float *x_block = x_row + nb * QK4_0;
                float block_sum = 0.0f;
                for (int i = 0; i < 16; i++) {
                    uint8_t q = qs[i];
                    int8_t v0 = (q & 0x0F) - 8;
                    int8_t v1 = (q >> 4) - 8;
                    block_sum += (v0 * x_block[i*2]) + (v1 * x_block[i*2 + 1]);
                }
                sum += block_sum * scale;
            }
            y_row[o] = sum;
        }
    }
}

static float dequantize_embedding_value(const block_q4_0 *blocks, int idx) {
    int block_idx = idx / QK4_0;
    const block_q4_0 *block = &blocks[block_idx];
    int block_offset = idx % QK4_0;
    uint8_t q = block->qs[block_offset / 2];
    int8_t v = (block_offset % 2 == 0) ? (q & 0x0F) - 8 : (q >> 4) - 8;
    return v * block->scale;
}

static void layer_norm(float *out, const float *x, const float *gamma, const float *beta,
                       int seq_len, int hidden_size) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden_size;
        float *out_row = out + s * hidden_size;
        float mean = 0.0f;
        for (int i = 0; i < hidden_size; i++) mean += x_row[i];
        mean /= hidden_size;
        float var = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            float diff = x_row[i] - mean;
            var += diff * diff;
        }
        var /= hidden_size;
        float std_inv = 1.0f / sqrtf(var + GTE_LAYER_NORM_EPS);
        for (int i = 0; i < hidden_size; i++) {
            out_row[i] = gamma[i] * (x_row[i] - mean) * std_inv + beta[i];
        }
    }
}

static void gelu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        x[i] = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
    }
}

static void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

static void l2_normalize(float *x, int n) {
    float norm = 0.0f;
    for (int i = 0; i < n; i++) norm += x[i] * x[i];
    norm = sqrtf(norm);
    if (norm > 0.0f) {
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < n; i++) x[i] *= inv_norm;
    }
}

/* --- Tokenizer --- */

static int is_punctuation(unsigned char c) {
    return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126);
}

static int is_whitespace(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static char **basic_tokenize(const char *text, int *num_tokens) {
    int capacity = 64, count = 0;
    char **tokens = malloc(capacity * sizeof(char *));
    const char *p = text;
    while (*p) {
        while (*p && is_whitespace(*p)) p++;
        if (!*p) break;
        const char *start = p;
        if (is_punctuation(*p)) p++;
        else while (*p && !is_whitespace(*p) && !is_punctuation(*p)) p++;
        int len = p - start;
        char *token = malloc(len + 1);
        for (int i = 0; i < len; i++) token[i] = tolower((unsigned char)start[i]);
        token[len] = '\0';
        if (count >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(char *));
        }
        tokens[count++] = token;
    }
    *num_tokens = count;
    return tokens;
}

static int *wordpiece_tokenize(gte_ctx *ctx, const char *word, int *num_subtokens) {
    int len = strlen(word);
    if (len == 0) { *num_subtokens = 0; return NULL; }
    int *subtokens = malloc((len + 1) * sizeof(int)), count = 0, start = 0;
    while (start < len) {
        int end = len, found_id = -1;
        while (start < end) {
            char candidate[256];
            int cand_len = 0;
            if (start > 0) { candidate[cand_len++] = '#'; candidate[cand_len++] = '#'; }
            for (int i = start; i < end && cand_len < 254; i++) candidate[cand_len++] = word[i];
            candidate[cand_len] = '\0';
            int id = vocab_lookup(ctx->vocab_hash, candidate);
            if (id >= 0) { found_id = id; break; }
            end--;
        }
        if (found_id < 0) { subtokens[count++] = TOKEN_UNK; start++; }
        else { subtokens[count++] = found_id; start = end; }
    }
    *num_subtokens = count;
    return subtokens;
}

static int *tokenize(gte_ctx *ctx, const char *text, int *num_tokens, int max_len) {
    int num_basic;
    char **basic_tokens = basic_tokenize(text, &num_basic);
    int capacity = 128, total = 0;
    int *all_tokens = malloc(capacity * sizeof(int));
    all_tokens[total++] = TOKEN_CLS;
    for (int t = 0; t < num_basic && total < max_len - 1; t++) {
        int num_sub;
        int *subtokens = wordpiece_tokenize(ctx, basic_tokens[t], &num_sub);
        for (int s = 0; s < num_sub && total < max_len - 1; s++) {
            if (total >= capacity) {
                capacity *= 2;
                all_tokens = realloc(all_tokens, capacity * sizeof(int));
            }
            all_tokens[total++] = subtokens[s];
        }
        free(subtokens);
        free(basic_tokens[t]);
    }
    free(basic_tokens);
    all_tokens[total++] = TOKEN_SEP;
    *num_tokens = total;
    return all_tokens;
}

/* --- Transformer --- */

static void self_attention(gte_ctx *ctx, layer_weights *layer, int seq_len, const int *attn_mask) {
    int hidden = ctx->hidden_size, heads = ctx->num_heads, head_dim = ctx->head_dim;
    linear_q4_0(ctx->q_proj, ctx->hidden_states, layer->query_weight, layer->query_bias, seq_len, hidden, hidden);
    linear_q4_0(ctx->k_proj, ctx->hidden_states, layer->key_weight, layer->key_bias, seq_len, hidden, hidden);
    linear_q4_0(ctx->v_proj, ctx->hidden_states, layer->value_weight, layer->value_bias, seq_len, hidden, hidden);
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int h = 0; h < heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += ctx->q_proj[i * hidden + h * head_dim + d] * ctx->k_proj[j * hidden + h * head_dim + d];
                }
                score *= scale;
                if (attn_mask && !attn_mask[j]) score = -10000.0f;
                ctx->attn_scores[h * seq_len * seq_len + i * seq_len + j] = score;
            }
            softmax(&ctx->attn_scores[h * seq_len * seq_len + i * seq_len], seq_len);
        }
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    sum += ctx->attn_scores[h * seq_len * seq_len + i * seq_len + j] * ctx->v_proj[j * hidden + h * head_dim + d];
                }
                ctx->attn_output[i * hidden + h * head_dim + d] = sum;
            }
        }
    }
    linear_q4_0(ctx->temp_hidden, ctx->attn_output, layer->attn_output_weight, layer->attn_output_bias, seq_len, hidden, hidden);
    for (int i = 0; i < seq_len * hidden; i++) ctx->temp_hidden[i] += ctx->hidden_states[i];
    layer_norm(ctx->hidden_states, ctx->temp_hidden, layer->attn_ln_weight, layer->attn_ln_bias, seq_len, hidden);
}

static void feed_forward(gte_ctx *ctx, layer_weights *layer, int seq_len) {
    int hidden = ctx->hidden_size, inter = ctx->intermediate_size;
    linear_q4_0(ctx->ffn_hidden, ctx->hidden_states, layer->ffn_inter_weight, layer->ffn_inter_bias, seq_len, hidden, inter);
    gelu(ctx->ffn_hidden, seq_len * inter);
    linear_q4_0(ctx->temp_hidden, ctx->ffn_hidden, layer->ffn_output_weight, layer->ffn_output_bias, seq_len, inter, hidden);
    for (int i = 0; i < seq_len * hidden; i++) ctx->temp_hidden[i] += ctx->hidden_states[i];
    layer_norm(ctx->hidden_states, ctx->temp_hidden, layer->ffn_ln_weight, layer->ffn_ln_bias, seq_len, hidden);
}

static void transformer_forward(gte_ctx *ctx, const int *token_ids, int seq_len, const int *attn_mask) {
    int hidden = ctx->hidden_size, num_blocks_per_row = hidden / QK4_0;
    const block_q4_0 *type_row = ctx->token_type_embeddings;
    for (int s = 0; s < seq_len; s++) {
        const block_q4_0 *tok_row = ctx->token_embeddings + token_ids[s] * num_blocks_per_row;
        const block_q4_0 *pos_row = ctx->position_embeddings + s * num_blocks_per_row;
        for (int d = 0; d < hidden; d++) {
            ctx->hidden_states[s * hidden + d] = dequantize_embedding_value(tok_row, d) +
                                               dequantize_embedding_value(pos_row, d) +
                                               dequantize_embedding_value(type_row, d);
        }
    }
    layer_norm(ctx->hidden_states, ctx->hidden_states, ctx->embed_ln_weight, ctx->embed_ln_bias, seq_len, hidden);
    for (int l = 0; l < ctx->num_layers; l++) {
        self_attention(ctx, &ctx->layers[l], seq_len, attn_mask);
        feed_forward(ctx, &ctx->layers[l], seq_len);
    }
}

static void mean_pooling(float *output, const float *hidden_states, const int *attn_mask, int seq_len, int hidden_size) {
    memset(output, 0, hidden_size * sizeof(float));
    int count = 0;
    for (int s = 0; s < seq_len; s++) {
        if (attn_mask[s]) {
            for (int d = 0; d < hidden_size; d++) output[d] += hidden_states[s * hidden_size + d];
            count++;
        }
    }
    if (count > 0) {
        float inv = 1.0f / count;
        for (int d = 0; d < hidden_size; d++) output[d] *= inv;
    }
}

/* --- Model Load/Free --- */

static int read_uint32(FILE *f, int *val) {
    unsigned char b[4]; if (fread(b, 1, 4, f) != 4) return 0;
    *val = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24); return 1;
}
static int read_uint16(FILE *f, int *val) {
    unsigned char b[2]; if (fread(b, 1, 2, f) != 2) return 0;
    *val = b[0] | (b[1] << 8); return 1;
}
static float *read_floats(FILE *f, int count) {
    float *data = malloc(count * sizeof(float));
    if (fread(data, sizeof(float), count, f) != (size_t)count) { free(data); return NULL; }
    return data;
}
static block_q4_0 *read_blocks(FILE *f, int num_weights) {
    int num_blocks = num_weights / QK4_0;
    block_q4_0 *data = malloc(num_blocks * sizeof(block_q4_0));
    if (fread(data, sizeof(block_q4_0), num_blocks, f) != (size_t)num_blocks) { free(data); return NULL; }
    return data;
}

gte_ctx *gte_load(const char *model_path) {
    FILE *f = fopen(model_path, "rb");
    if (!f) return NULL;
    char magic[4]; if (fread(magic, 1, 4, f) != 4 || memcmp(magic, GTE_MAGIC, 4) != 0) { fclose(f); return NULL; }
    gte_ctx *ctx = calloc(1, sizeof(gte_ctx));
    if (!read_uint32(f, &ctx->vocab_size) || !read_uint32(f, &ctx->hidden_size) ||
        !read_uint32(f, &ctx->num_layers) || !read_uint32(f, &ctx->num_heads) ||
        !read_uint32(f, &ctx->intermediate_size) || !read_uint32(f, &ctx->max_seq_len)) goto error;

    ctx->current_max_len = (ctx->max_seq_len < 256) ? ctx->max_seq_len : 256;
    ctx->head_dim = ctx->hidden_size / ctx->num_heads;
    ctx->vocab = malloc(ctx->vocab_size * sizeof(char *));
    ctx->vocab_hash = calloc(VOCAB_HASH_SIZE, sizeof(vocab_entry));
    for (int i = 0; i < ctx->vocab_size; i++) {
        int len; read_uint16(f, &len);
        ctx->vocab[i] = malloc(len + 1);
        fread(ctx->vocab[i], 1, len, f); ctx->vocab[i][len] = '\0';
        vocab_hash_insert(ctx->vocab_hash, ctx->vocab[i], i);
    }
    ctx->token_embeddings = read_blocks(f, ctx->vocab_size * ctx->hidden_size);
    ctx->position_embeddings = read_blocks(f, ctx->max_seq_len * ctx->hidden_size);
    ctx->token_type_embeddings = read_blocks(f, 2 * ctx->hidden_size);
    ctx->embed_ln_weight = read_floats(f, ctx->hidden_size);
    ctx->embed_ln_bias = read_floats(f, ctx->hidden_size);
    ctx->layers = calloc(ctx->num_layers, sizeof(layer_weights));
    for (int l = 0; l < ctx->num_layers; l++) {
        layer_weights *ly = &ctx->layers[l];
        ly->query_weight = read_blocks(f, ctx->hidden_size * ctx->hidden_size); ly->query_bias = read_floats(f, ctx->hidden_size);
        ly->key_weight = read_blocks(f, ctx->hidden_size * ctx->hidden_size); ly->key_bias = read_floats(f, ctx->hidden_size);
        ly->value_weight = read_blocks(f, ctx->hidden_size * ctx->hidden_size); ly->value_bias = read_floats(f, ctx->hidden_size);
        ly->attn_output_weight = read_blocks(f, ctx->hidden_size * ctx->hidden_size); ly->attn_output_bias = read_floats(f, ctx->hidden_size);
        ly->attn_ln_weight = read_floats(f, ctx->hidden_size); ly->attn_ln_bias = read_floats(f, ctx->hidden_size);
        ly->ffn_inter_weight = read_blocks(f, ctx->intermediate_size * ctx->hidden_size); ly->ffn_inter_bias = read_floats(f, ctx->intermediate_size);
        ly->ffn_output_weight = read_blocks(f, ctx->hidden_size * ctx->intermediate_size); ly->ffn_output_bias = read_floats(f, ctx->hidden_size);
        ly->ffn_ln_weight = read_floats(f, ctx->hidden_size); ly->ffn_ln_bias = read_floats(f, ctx->hidden_size);
    }
    ctx->pooler_weight = read_blocks(f, ctx->hidden_size * ctx->hidden_size); ctx->pooler_bias = read_floats(f, ctx->hidden_size);
    fclose(f);
    int ms = ctx->max_seq_len, h = ctx->hidden_size, it = ctx->intermediate_size;
    ctx->hidden_states = malloc(ms * h * sizeof(float));
    ctx->attn_scores = malloc(ctx->num_heads * ms * ms * sizeof(float));
    ctx->q_proj = malloc(ms * h * sizeof(float)); ctx->k_proj = malloc(ms * h * sizeof(float)); ctx->v_proj = malloc(ms * h * sizeof(float));
    ctx->attn_output = malloc(ms * h * sizeof(float)); ctx->ffn_hidden = malloc(ms * it * sizeof(float)); ctx->temp_hidden = malloc(ms * h * sizeof(float));
    return ctx;
error:
    if (f) fclose(f); gte_free(ctx); return NULL;
}

void gte_free(gte_ctx *ctx) {
    if (!ctx) return;
    if (ctx->vocab) { for (int i = 0; i < ctx->vocab_size; i++) free(ctx->vocab[i]); free(ctx->vocab); }
    if (ctx->vocab_hash) { for (int i = 0; i < VOCAB_HASH_SIZE; i++) free(ctx->vocab_hash[i].word); free(ctx->vocab_hash); }
    free(ctx->token_embeddings); free(ctx->position_embeddings); free(ctx->token_type_embeddings);
    free(ctx->embed_ln_weight); free(ctx->embed_ln_bias);
    if (ctx->layers) {
        for (int l = 0; l < ctx->num_layers; l++) {
            layer_weights *ly = &ctx->layers[l];
            free(ly->query_weight); free(ly->query_bias); free(ly->key_weight); free(ly->key_bias);
            free(ly->value_weight); free(ly->value_bias); free(ly->attn_output_weight); free(ly->attn_output_bias);
            free(ly->attn_ln_weight); free(ly->attn_ln_bias); free(ly->ffn_inter_weight); free(ly->ffn_inter_bias);
            free(ly->ffn_output_weight); free(ly->ffn_output_bias); free(ly->ffn_ln_weight); free(ly->ffn_ln_bias);
        }
        free(ctx->layers);
    }
    free(ctx->pooler_weight); free(ctx->pooler_bias);
    free(ctx->hidden_states); free(ctx->attn_scores); free(ctx->q_proj); free(ctx->k_proj);
    free(ctx->v_proj); free(ctx->attn_output); free(ctx->ffn_hidden); free(ctx->temp_hidden);
    free(ctx);
}

/* --- API --- */

float *gte_embed(gte_ctx *ctx, const char *text) {
    if (!ctx || !text) return NULL;
    int num_tokens;
    int *token_ids = tokenize(ctx, text, &num_tokens, ctx->current_max_len);
    int *mask = malloc(num_tokens * sizeof(int));
    for (int i = 0; i < num_tokens; i++) mask[i] = 1;
    transformer_forward(ctx, token_ids, num_tokens, mask);
    float *emb = malloc(ctx->hidden_size * sizeof(float));
    mean_pooling(emb, ctx->hidden_states, mask, num_tokens, ctx->hidden_size);
    l2_normalize(emb, ctx->hidden_size);
    free(token_ids); free(mask);
    return emb;
}

float *gte_embed_batch(gte_ctx *ctx, const char **texts, int count) {
    float *embs = malloc(count * ctx->hidden_size * sizeof(float));
    for (int i = 0; i < count; i++) {
        float *e = gte_embed(ctx, texts[i]);
        memcpy(embs + i * ctx->hidden_size, e, ctx->hidden_size * sizeof(float));
        free(e);
    }
    return embs;
}

int gte_dim(gte_ctx *ctx) { return ctx ? ctx->hidden_size : 0; }
int gte_max_seq_len(gte_ctx *ctx) { return ctx ? ctx->max_seq_len : 0; }

void gte_set_seq_len(gte_ctx *ctx, int seq_len) {
    if (!ctx) return;
    if (seq_len > ctx->max_seq_len) ctx->current_max_len = ctx->max_seq_len;
    else if (seq_len < 3) ctx->current_max_len = 3;
    else ctx->current_max_len = seq_len;
}

float gte_cosine_similarity(const float *a, const float *b, int dim) {
    float dot = 0.0f;
    for (int i = 0; i < dim; i++) dot += a[i] * b[i];
    return dot;
}
