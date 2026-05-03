/*
 * GTE-Small Embedding Library
 *
 * A pure C, dependency-free library for running GTE-small text embedding inference.
 */

#ifndef GTE_H
#define GTE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque context type */
typedef struct gte_ctx gte_ctx;

/* Load model from .gtemodel file. Returns NULL on error. */
gte_ctx *gte_load(const char *model_path);

/* Free model and all associated resources. */
void gte_free(gte_ctx *ctx);

/*
 * Generate embedding for a single text.
 * Returns a newly allocated array of floats (size = gte_dim()).
 * Caller must free() the returned pointer.
 * Returns NULL on error.
 */
float *gte_embed(gte_ctx *ctx, const char *text);

/*
 * Generate embeddings for multiple texts.
 * Returns a newly allocated array of floats (size = count * gte_dim()).
 * Embeddings are stored contiguously: [emb0, emb1, emb2, ...]
 * Caller must free() the returned pointer.
 * Returns NULL on error.
 */
float *gte_embed_batch(gte_ctx *ctx, const char **texts, int count);

/* Get the embedding dimension (384 for GTE-small). */
int gte_dim(gte_ctx *ctx);

/* Get the hard maximum sequence length supported by the model file (usually 512). */
int gte_max_seq_len(gte_ctx *ctx);

/*
 * Set the current sequence length limit (default is 256).
 * This truncates input text to save computation time.
 * Cannot exceed the hard limit returned by gte_max_seq_len().
 */
void gte_set_seq_len(gte_ctx *ctx, int seq_len);

/*
 * Compute cosine similarity between two embeddings.
 * Both embeddings must have the same dimension.
 * Returns a value between -1 and 1.
 */
float gte_cosine_similarity(const float *a, const float *b, int dim);

#ifdef __cplusplus
}
#endif

#endif /* GTE_H */
