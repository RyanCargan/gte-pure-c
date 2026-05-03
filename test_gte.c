/*
 * Test program for GTE library
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gte.h"

#define DEFAULT_MODEL_PATH "gte-small.gtemodel"
#define MAX_SENTENCES 64

void print_usage(const char *prog) {
    printf("Usage: %s [OPTIONS] [SENTENCES...]\n\n", prog);
    printf("Options:\n");
    printf("  --model-path PATH   Path to .gtemodel file (default: %s)\n", DEFAULT_MODEL_PATH);
    printf("  --max-len N         Set max sequence length (default: 256)\n");
    printf("  --help              Show this help message\n");
}

void print_embedding(const float *emb, int dim, int n) {
    printf("[");
    for (int i = 0; i < n && i < dim; i++) {
        printf("%.6f", emb[i]);
        if (i < n - 1) printf(", ");
    }
    if (n < dim) printf(", ...");
    printf("]\n");
}

int main(int argc, char **argv) {
    const char *model_path = DEFAULT_MODEL_PATH;
    const char *user_sentences[MAX_SENTENCES];
    int num_user_sentences = 0;
    int custom_max_len = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--model-path") == 0) {
            if (i + 1 >= argc) return 1;
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--max-len") == 0) {
            if (i + 1 >= argc) return 1;
            custom_max_len = atoi(argv[++i]);
        } else if (argv[i][0] != '-') {
            if (num_user_sentences < MAX_SENTENCES)
                user_sentences[num_user_sentences++] = argv[i];
        }
    }

    const char *default_sentences[] = {
        "The weather is lovely today.",
        "It's so sunny outside!",
        "Machine learning is transforming industries.",
        "I love programming in C."
    };
    const char **sentences = num_user_sentences > 0 ? user_sentences : default_sentences;
    int num_sentences = num_user_sentences > 0 ? num_user_sentences : 4;

    printf("Loading model from %s...\n", model_path);
    gte_ctx *ctx = gte_load(model_path);
    if (!ctx) return 1;

    if (custom_max_len > 0) gte_set_seq_len(ctx, custom_max_len);

    printf("Embedding dimension: %d\n", gte_dim(ctx));
    printf("Max sequence length limit: %d\n\n", custom_max_len > 0 ? custom_max_len : 256);

    float **embeddings = malloc(num_sentences * sizeof(float *));
    for (int i = 0; i < num_sentences; i++) {
        clock_t start = clock();
        embeddings[i] = gte_embed(ctx, sentences[i]);
        double ms = (double)(clock() - start) * 1000.0 / CLOCKS_PER_SEC;
        printf("S%d: \"%s\"\n    Time: %.2f ms | ", i + 1, sentences[i], ms);
        print_embedding(embeddings[i], gte_dim(ctx), 3);
    }

    printf("\nCosine similarity matrix:\n");
    for (int i = 0; i < num_sentences; i++) {
        printf("S%d: ", i + 1);
        for (int j = 0; j < num_sentences; j++) {
            printf(" %.3f ", gte_cosine_similarity(embeddings[i], embeddings[j], gte_dim(ctx)));
        }
        printf("\n");
    }

    for (int i = 0; i < num_sentences; i++) free(embeddings[i]);
    free(embeddings);
    gte_free(ctx);
    return 0;
}
