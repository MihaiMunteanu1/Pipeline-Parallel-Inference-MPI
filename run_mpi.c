/*
 * run_mpi.c - pipeline parallel inference for Llama-2 models, on CPU, with MPI.
 *
 * Based on run.c from llama2.c by Andrej Karpathy (MIT License):
 * https://github.com/karpathy/llama2.c
 * Copyright (c) 2023 Andrej Karpathy
 * The model math is unchanged. What changed: the layers are split across MPI
 * ranks, and several sequences (streams) can be in the pipeline at once.
 *
 * Who does what, one token at a time:
 *
 *   rank 0        embedding + its layers --> send x to rank 1
 *   rank 1..N-2   recv x, run its layers --> send x to next rank
 *   rank N-1      recv x, run its layers --> send x back to rank 0
 *   rank 0        final norm + classifier + sampling --> next token
 *
 * Build:  mpicc -std=gnu99 -O3 -o run_mpi run_mpi.c -lm
 * Run:    mpirun -np 2 ./run_mpi stories15M.bin -s 42 -n 200 -i "Once upon a time"
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
    int first_layer; // first layer owned by this rank
    int n_local; // number of layers owned by this rank
} Transformer;

void malloc_run_state(RunState* s, Config* p, int n_local, int streams) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    // the kv cache only holds this rank's layers, with one copy per stream
    size_t cache_size = (size_t)streams * n_local * p->seq_len * kv_dim;
    if (cache_size == 0) cache_size = 1; // a rank can own zero layers
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(cache_size, sizeof(float));
    s->value_cache = calloc(cache_size, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path, int streams) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config, t->n_local, streams);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// Copy the token embedding into x (rank 0 only).
void embed_token(Transformer* transformer, int token, float* x) {
    int dim = transformer->config.dim;
    memcpy(x, transformer->weights.token_embedding_table + token * dim, dim * sizeof(float));
}

// Run the layers owned by this rank on x. The layer math is the same as run.c.
void forward_layers(Transformer* transformer, float* x, int stream, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;
    int n_local = transformer->n_local;

    // forward only this rank's layers
    for (int li = 0; li < n_local; li++) {
        unsigned long long l = (unsigned long long)transformer->first_layer + li;

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        size_t loff = ((size_t)stream * n_local + li) * p->seq_len * kv_dim; // kv cache offset for this stream and layer
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }
}

// Final norm and classifier, which turn x into logits (rank 0 only).
float* classify(Transformer* transformer, float* x) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    rmsnorm(x, x, w->rms_final_weight, p->dim);
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// The pipeline: streams on rank 0, and the receive/compute/send loop on the rest

#define TAG_STOP 0 // a stream with index s travels with the MPI tag s + 1

typedef struct {
    int* prompt_tokens;
    int num_prompt_tokens;
    int token;     // current token
    int pos;       // position in the sequence
    int done;
    unsigned long long rng_state; // each stream has its own random sequence
    char* text;    // generated text, used when there is more than one stream
    size_t text_len;
    size_t text_cap;
    float* sendbuf; // must stay untouched until the MPI_Isend completes
    MPI_Request req;
} Stream;

// what each rank measures, sent to rank 0 at the end
static double t_compute = 0.0; // time spent running the model
static double t_comm = 0.0;    // time inside MPI calls: transfer plus waiting for data

static void append_text(Stream* st, char* piece) {
    if (piece == NULL || piece[0] == '\0') return;
    // same filter as safe_printf: skip single unprintable bytes
    if (piece[1] == '\0') {
        unsigned char byte = (unsigned char)piece[0];
        if (!(isprint(byte) || isspace(byte))) return;
    }
    size_t n = strlen(piece);
    if (st->text_len + n + 1 > st->text_cap) {
        st->text_cap = (st->text_len + n + 1) * 2;
        st->text = realloc(st->text, st->text_cap);
        if (!st->text) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    }
    memcpy(st->text + st->text_len, piece, n + 1);
    st->text_len += n;
}

// Rank 0, first half of a token: embedding and local layers.
static void start_token(Transformer* t, Stream* st, int s, float* x) {
    double t0 = MPI_Wtime();
    embed_token(t, st->token, x);
    forward_layers(t, x, s, st->pos);
    t_compute += MPI_Wtime() - t0;
}

// Rank 0, second half of a token: logits and sampling, same logic as generate()
// in run.c. Returns 1 if the stream has more tokens to produce.
static int finish_token(Transformer* t, Tokenizer* tok, Sampler* sampler, Stream* st, int n_streams, float* x, int steps) {
    double t0 = MPI_Wtime();
    float* logits = classify(t, x);
    int next;
    if (st->pos < st->num_prompt_tokens - 1) {
        // if we are still processing the input prompt, force the next prompt token
        next = st->prompt_tokens[st->pos + 1];
    } else {
        // otherwise sample the next token from the logits
        sampler->rng_state = st->rng_state;
        next = sample(sampler, logits);
        st->rng_state = sampler->rng_state;
    }
    st->pos++;
    int alive = 1;
    // data-dependent terminating condition: the BOS (=1) token delimits sequences
    if (next == 1) {
        alive = 0;
    } else {
        char* piece = decode(tok, st->token, next);
        if (n_streams == 1) {
            safe_printf(piece); // one stream: print live, like run.c
            fflush(stdout);
        } else {
            append_text(st, piece);
        }
        st->token = next;
        if (st->pos >= steps) alive = 0;
    }
    t_compute += MPI_Wtime() - t0;
    if (!alive) st->done = 1;
    return alive;
}

// Rank 0, send the half-done token to rank 1. MPI_Isend means rank 0 can never
// block while sending, so the ring of ranks can't deadlock.
static void send_token(Stream* st, int s, int dim) {
    double t0 = MPI_Wtime();
    MPI_Isend(st->sendbuf, dim, MPI_FLOAT, 1, s + 1, MPI_COMM_WORLD, &st->req);
    t_comm += MPI_Wtime() - t0;
}

static void wait_send(Stream* st) {
    double t0 = MPI_Wtime();
    MPI_Wait(&st->req, MPI_STATUS_IGNORE);
    t_comm += MPI_Wtime() - t0;
}

// Rank 0: returns the number of forward passes done.
static long run_head(Transformer* t, Tokenizer* tok, Sampler* sampler, Stream* streams, int n_streams, int steps, int size) {
    int dim = t->config.dim;
    float* x = t->state.x;
    long passes = 0;
    int active = n_streams;

    if (size == 1) {
        // baseline on one process: same code, no network, streams take turns
        while (active > 0) {
            for (int s = 0; s < n_streams; s++) {
                if (streams[s].done) continue;
                start_token(t, &streams[s], s, x);
                passes++;
                if (!finish_token(t, tok, sampler, &streams[s], n_streams, x, steps)) active--;
            }
        }
        return passes;
    }

    // fill the pipeline: one token from every stream
    for (int s = 0; s < n_streams; s++) {
        start_token(t, &streams[s], s, streams[s].sendbuf);
        send_token(&streams[s], s, dim);
    }
    // each token that returns finishes on this rank, and its stream sends the next one
    while (active > 0) {
        MPI_Status status;
        double t0 = MPI_Wtime();
        MPI_Recv(x, dim, MPI_FLOAT, size - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        t_comm += MPI_Wtime() - t0;
        int s = status.MPI_TAG - 1;
        passes++;
        wait_send(&streams[s]); // the token came back, so its previous send is finished
        if (finish_token(t, tok, sampler, &streams[s], n_streams, x, steps)) {
            start_token(t, &streams[s], s, streams[s].sendbuf);
            send_token(&streams[s], s, dim);
        } else {
            active--;
        }
    }
    MPI_Send(x, 0, MPI_FLOAT, 1, TAG_STOP, MPI_COMM_WORLD);
    return passes;
}

// Ranks 1..N-1: receive x, run the local layers, pass x on.
static void run_worker(Transformer* t, int n_streams, int rank, int size) {
    int dim = t->config.dim;
    float* x = t->state.x;
    int* pos = calloc(n_streams, sizeof(int)); // each stream advances one position per token
    if (!pos) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    for (;;) {
        MPI_Status status;
        double t0 = MPI_Wtime();
        MPI_Recv(x, dim, MPI_FLOAT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        t_comm += MPI_Wtime() - t0;
        if (status.MPI_TAG == TAG_STOP) {
            if (rank < size - 1) MPI_Send(x, 0, MPI_FLOAT, rank + 1, TAG_STOP, MPI_COMM_WORLD);
            break;
        }
        int s = status.MPI_TAG - 1;
        t0 = MPI_Wtime();
        forward_layers(t, x, s, pos[s]);
        pos[s]++;
        t_compute += MPI_Wtime() - t0;
        t0 = MPI_Wtime();
        MPI_Send(x, dim, MPI_FLOAT, (rank + 1) % size, status.MPI_TAG, MPI_COMM_WORLD);
        t_comm += MPI_Wtime() - t0;
    }
    free(pos);
}

// One-way latency between ranks 0 and 1, for a message the size of x. In seconds.
static double measure_latency(float* x, int dim, int rank, int size) {
    int rounds = 1000;
    if (size < 2 || rank > 1) return 0.0;
    double t0 = MPI_Wtime();
    for (int i = 0; i < rounds; i++) {
        if (rank == 0) {
            MPI_Send(x, dim, MPI_FLOAT, 1, 99, MPI_COMM_WORLD);
            MPI_Recv(x, dim, MPI_FLOAT, 1, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(x, dim, MPI_FLOAT, 0, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(x, dim, MPI_FLOAT, 0, 99, MPI_COMM_WORLD);
        }
    }
    return (MPI_Wtime() - t0) / rounds / 2.0;
}

// ----------------------------------------------------------------------------
// Measurements for the report

// Peak resident memory of this process in MB (Linux), or -1 if unknown.
static double peak_rss_mb(void) {
    FILE* f = fopen("/proc/self/status", "r");
    char line[256];
    double kb = -1.0;
    if (!f) return -1.0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) { kb = atof(line + 6); break; }
    }
    fclose(f);
    return kb < 0 ? -1.0 : kb / 1024.0;
}

// Size in MB of the weights this rank really uses.
static double local_weights_mb(Transformer* t, int rank) {
    Config* p = &t->config;
    double dim = p->dim, hidden = p->hidden_dim, vocab = p->vocab_size;
    double kv_dim = (double)(p->dim * p->n_kv_heads) / p->n_heads;
    double per_layer = 2*dim + 2*dim*dim + 2*dim*kv_dim + 3*dim*hidden; // 2 norms, wq+wo, wk+wv, ffn
    double params = per_layer * t->n_local;
    if (rank == 0) {
        params += vocab * dim + dim; // embedding and final norm
        if (t->weights.wcls != t->weights.token_embedding_table) params += vocab * dim;
    }
    return params * sizeof(float) / (1024.0 * 1024.0);
}

// ----------------------------------------------------------------------------
// CLI

static int g_rank = 0;

void error_usage() {
    if (g_rank == 0) {
        fprintf(stderr, "Usage:   mpirun -np <ranks> run_mpi <checkpoint> [options]\n");
        fprintf(stderr, "Example: mpirun -np 2 run_mpi model.bin -n 256 -i \"Once upon a time\"\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
        fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
        fprintf(stderr, "  -s <int>    random seed, default time(NULL). Stream i uses seed + i\n");
        fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
        fprintf(stderr, "  -i <string> input prompt\n");
        fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
        fprintf(stderr, "  -b <int>    streams: sequences generated at once, default 1\n");
        fprintf(stderr, "  -l <list>   layers per rank, e.g. 3,3 or 0,6. Default: even split\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(EXIT_FAILURE);
}

// Layers per rank: from -l if given, otherwise as even as possible.
static void compute_split(char* spec, int n_layers, int size, int* counts) {
    if (spec == NULL) {
        for (int r = 0; r < size; r++) counts[r] = n_layers / size + (r < n_layers % size ? 1 : 0);
        return;
    }
    char* copy = malloc(strlen(spec) + 1);
    if (!copy) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    strcpy(copy, spec);
    int r = 0, sum = 0;
    for (char* tok = strtok(copy, ","); tok != NULL; tok = strtok(NULL, ",")) {
        if (r == size || atoi(tok) < 0) { r = size + 1; break; }
        counts[r] = atoi(tok);
        sum += counts[r];
        r++;
    }
    free(copy);
    if (r != size || sum != n_layers) {
        if (g_rank == 0) fprintf(stderr, "-l needs %d numbers that add up to %d (the model's layers)\n", size, n_layers);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    g_rank = rank;

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    int n_streams = 1;          // sequences generated at once
    char *split_spec = NULL;    // layers per rank

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'b') { n_streams = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'l') { split_spec = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;
    if (n_streams < 1) n_streams = 1;

    // only the header is needed for the split, the weights are mapped below
    Config header;
    FILE* cf = fopen(checkpoint_path, "rb");
    if (!cf || fread(&header, sizeof(Config), 1, cf) != 1) {
        fprintf(stderr, "rank %d: couldn't read %s\n", rank, checkpoint_path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fclose(cf);
    int* counts = malloc(size * sizeof(int));
    if (!counts) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    compute_split(split_spec, header.n_layers, size, counts);
    int first_layer = 0;
    for (int r = 0; r < rank; r++) first_layer += counts[r];

    // build the Transformer via the model .bin file
    Transformer transformer;
    transformer.first_layer = first_layer;
    transformer.n_local = counts[rank];
    build_transformer(&transformer, checkpoint_path, n_streams);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length
    int dim = transformer.config.dim;

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostname_len;
    memset(hostname, 0, sizeof(hostname));
    MPI_Get_processor_name(hostname, &hostname_len);

    double latency = measure_latency(transformer.state.x, dim, rank, size);
    memset(transformer.state.x, 0, dim * sizeof(float));
    MPI_Barrier(MPI_COMM_WORLD);

    Tokenizer tokenizer;
    Sampler sampler;
    Stream* streams = NULL;
    long passes = 0;
    double wall = 0.0;

    if (rank == 0) {
        // build the Tokenizer via the tokenizer .bin file
        build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
        // build the Sampler
        build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

        char *empty_prompt = "";
        if (prompt == NULL) { prompt = empty_prompt; }
        streams = calloc(n_streams, sizeof(Stream));
        if (!streams) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
        for (int s = 0; s < n_streams; s++) {
            Stream* st = &streams[s];
            st->prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
            st->sendbuf = calloc(dim, sizeof(float));
            if (!st->prompt_tokens || !st->sendbuf) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
            encode(&tokenizer, prompt, 1, 0, st->prompt_tokens, &st->num_prompt_tokens);
            if (st->num_prompt_tokens < 1) {
                fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            st->token = st->prompt_tokens[0];
            st->rng_state = rng_seed + s;
            st->req = MPI_REQUEST_NULL;
        }
    }

    // run!
    double start = MPI_Wtime();
    if (rank == 0) {
        passes = run_head(&transformer, &tokenizer, &sampler, streams, n_streams, steps, size);
        for (int s = 0; s < n_streams; s++) wait_send(&streams[s]);
    } else {
        run_worker(&transformer, n_streams, rank, size);
    }
    wall = MPI_Wtime() - start;

    // gather what each rank measured
    double mine[4] = { t_compute, t_comm, local_weights_mb(&transformer, rank), peak_rss_mb() };
    double* all = rank == 0 ? malloc(size * sizeof(mine)) : NULL;
    char* hosts = rank == 0 ? malloc(size * MPI_MAX_PROCESSOR_NAME) : NULL;
    MPI_Gather(mine, 4, MPI_DOUBLE, all, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hosts, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (n_streams == 1) {
            printf("\n");
        } else {
            for (int s = 0; s < n_streams; s++) {
                printf("[stream %d] %s\n", s, streams[s].text ? streams[s].text : "");
            }
        }
        fflush(stdout);

        // the report goes to stderr, so stdout holds only the generated text
        fprintf(stderr, "\n--- ranks: %d | streams: %d | tokens: %ld | time: %.3f s | throughput: %.2f tok/s\n",
                size, n_streams, passes, wall, passes / wall);
        if (size > 1) {
            fprintf(stderr, "--- one-way latency rank 0 <-> 1, %d bytes: %.1f us\n",
                    dim * (int)sizeof(float), latency * 1e6);
        }
        fprintf(stderr, "%-4s %-20s %-7s %11s %13s %12s %12s\n",
                "rank", "host", "layers", "compute (s)", "MPI+wait (s)", "weights (MB)", "peak RSS (MB)");
        int l = 0;
        for (int r = 0; r < size; r++) {
            char range[32];
            if (counts[r] > 0) sprintf(range, "%d-%d", l, l + counts[r] - 1);
            else sprintf(range, "none");
            fprintf(stderr, "%-4d %-20s %-7s %11.3f %13.3f %12.1f %12.1f\n",
                    r, hosts + r * MPI_MAX_PROCESSOR_NAME, range,
                    all[r*4], all[r*4+1], all[r*4+2], all[r*4+3]);
            l += counts[r];
        }

        // cleanup on rank 0
        for (int s = 0; s < n_streams; s++) {
            free(streams[s].prompt_tokens);
            free(streams[s].sendbuf);
            free(streams[s].text);
        }
        free(streams);
        free(all);
        free(hosts);
        free_sampler(&sampler);
        free_tokenizer(&tokenizer);
    }

    // memory and file handles cleanup
    free(counts);
    free_transformer(&transformer);
    MPI_Finalize();
    return 0;
}
