/*
 * embed_mpi.c - sentence embeddings with a BERT encoder (all-MiniLM-L6-v2 or
 * bge-base-en-v1.5), in plain C, with the layers split across MPI ranks (pipeline parallelism).
 *
 * Who does what, for one text:
 *
 *   rank 0        tokenize, embeddings, its layers --> send x to rank 1
 *   rank 1..N-2   recv x, run its layers           --> send x to the next rank
 *   rank N-1      recv x, run its layers, make and normalize the vector
 *                                                  --> send the vector to rank 0
 *
 * Texts don't depend on each other, so rank 0 starts the next text without
 * waiting for the previous one. That is what keeps every rank busy.
 *
 * Build:  mpicc -std=gnu99 -O3 -ffast-math -o embed_mpi embed_mpi.c -lm
 * Index:  mpirun -np 2 ./embed_mpi minilm.bin vocab.txt -f texts.txt -o vectors.bin
 * Search: mpirun -np 2 ./embed_mpi minilm.bin vocab.txt -f texts.txt -v vectors.bin -q "a question"
 * Live:   add -t 1 to see what every rank works on and what it sends
 */

#define _GNU_SOURCE
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define MAGIC_MEAN 0x4d424544 // written by export_minilm.py: the vector is the average of the tokens
#define MAGIC_CLS  0x4d424543 // written by export_bge.py: the vector is the [CLS] token
#define TAG_STOP 0
#define TAG_DATA 1

static int g_rank = 0;

static void die(const char* msg) {
    fprintf(stderr, "rank %d: %s\n", g_rank, msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(EXIT_FAILURE);
}

static void* xcalloc(size_t n, size_t size) {
    void* p = calloc(n ? n : 1, size);
    if (!p) die("out of memory");
    return p;
}

// ----------------------------------------------------------------------------
// The model file: a header of 7 ints, then all the weights as float32

typedef struct {
    int magic;
    int dim;        // 384
    int n_layers;   // 6
    int n_heads;    // 12
    int hidden_dim; // 1536
    int vocab_size; // 30522
    int max_pos;    // 512
} Config;

typedef struct { // pointers into the mapped file, for one layer
    float *qw, *qb, *kw, *kb, *vw, *vb; // query, key, value
    float *ow, *ob;     // attention output
    float *n1w, *n1b;   // layer norm after attention
    float *iw, *ib;     // feed forward: dim -> hidden
    float *fw, *fb;     // feed forward: hidden -> dim
    float *n2w, *n2b;   // layer norm after feed forward
} Layer;

typedef struct {
    Config c;
    float *word_emb, *pos_emb, *type_emb, *emb_norm_w, *emb_norm_b;
    Layer* layers;
    int first_layer, n_local;
    int max_len;
    // work buffers
    float *q, *k, *v, *ctx, *tmp, *hid, *att;
    void* map;
    size_t map_size;
} Model;

static long long layer_params(const Config* c) {
    long long d = c->dim, h = c->hidden_dim;
    return 4 * (d * d + d)        // q, k, v, attention output
         + 2 * d                  // layer norm
         + h * d + h              // dim -> hidden
         + d * h + d              // hidden -> dim
         + 2 * d;                 // layer norm
}

static void load_model(Model* m, const char* path, int first_layer, int n_local, int max_len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("can't open the model file");
    struct stat st;
    if (fstat(fd, &st) != 0) die("can't read the model file");
    m->map_size = (size_t)st.st_size;
    m->map = mmap(NULL, m->map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m->map == MAP_FAILED) die("mmap failed");
    close(fd);
    memcpy(&m->c, m->map, sizeof(Config));
    if (m->c.magic != MAGIC_MEAN && m->c.magic != MAGIC_CLS) die("this is not a file made by the export scripts");

    Config* c = &m->c;
    long long d = c->dim, h = c->hidden_dim;
    long long expected = (long long)sizeof(Config) + 4 * (c->vocab_size * d + c->max_pos * d + d + 2 * d
                         + c->n_layers * layer_params(c));
    if (expected != (long long)m->map_size) die("the model file has the wrong size");

    float* p = (float*)((char*)m->map + sizeof(Config));
    m->word_emb = p;   p += c->vocab_size * d;
    m->pos_emb = p;    p += c->max_pos * d;
    m->type_emb = p;   p += d;
    m->emb_norm_w = p; p += d;
    m->emb_norm_b = p; p += d;
    m->layers = xcalloc(c->n_layers, sizeof(Layer));
    for (int l = 0; l < c->n_layers; l++) {
        Layer* y = &m->layers[l];
        y->qw = p; p += d * d; y->qb = p; p += d;
        y->kw = p; p += d * d; y->kb = p; p += d;
        y->vw = p; p += d * d; y->vb = p; p += d;
        y->ow = p; p += d * d; y->ob = p; p += d;
        y->n1w = p; p += d; y->n1b = p; p += d;
        y->iw = p; p += h * d; y->ib = p; p += h;
        y->fw = p; p += d * h; y->fb = p; p += d;
        y->n2w = p; p += d; y->n2b = p; p += d;
    }

    m->first_layer = first_layer;
    m->n_local = n_local;
    m->max_len = max_len;
    m->q = xcalloc(max_len * d, sizeof(float));
    m->k = xcalloc(max_len * d, sizeof(float));
    m->v = xcalloc(max_len * d, sizeof(float));
    m->ctx = xcalloc(max_len * d, sizeof(float));
    m->tmp = xcalloc(max_len * d, sizeof(float));
    m->hid = xcalloc(max_len * h, sizeof(float));
    m->att = xcalloc(max_len, sizeof(float));
}

// ----------------------------------------------------------------------------
// The tokenizer: lowercase, remove accents, split on punctuation, then WordPiece

typedef struct { char* s; int id; } VocabEntry;

typedef struct {
    VocabEntry* entries;
    int size;
    int cls_id, sep_id, unk_id;
} Vocab;

static int cmp_vocab(const void* a, const void* b) {
    return strcmp(((const VocabEntry*)a)->s, ((const VocabEntry*)b)->s);
}

static int vocab_find(const Vocab* v, const char* s) {
    VocabEntry key;
    key.s = (char*)s;
    VocabEntry* e = bsearch(&key, v->entries, v->size, sizeof(VocabEntry), cmp_vocab);
    return e ? e->id : -1;
}

static void load_vocab(Vocab* v, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) die("can't open the vocabulary file");
    int cap = 40000;
    v->entries = xcalloc(cap, sizeof(VocabEntry));
    v->size = 0;
    char* line = NULL;
    size_t n = 0;
    while (getline(&line, &n, f) != -1) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (v->size == cap) die("the vocabulary file is too long");
        v->entries[v->size].s = strdup(line);
        v->entries[v->size].id = v->size;
        v->size++;
    }
    free(line);
    fclose(f);
    qsort(v->entries, v->size, sizeof(VocabEntry), cmp_vocab);
    v->cls_id = vocab_find(v, "[CLS]");
    v->sep_id = vocab_find(v, "[SEP]");
    v->unk_id = vocab_find(v, "[UNK]");
    if (v->cls_id < 0 || v->sep_id < 0 || v->unk_id < 0) die("the vocabulary has no [CLS], [SEP] or [UNK]");
}

// Letters with accents and their plain form. Covers Latin text, Romanian included.
static const struct { int cp; char base; } ACCENTS[] = {
    {0xC0,'a'},{0xC1,'a'},{0xC2,'a'},{0xC3,'a'},{0xC4,'a'},{0xC5,'a'},{0xC7,'c'},
    {0xC8,'e'},{0xC9,'e'},{0xCA,'e'},{0xCB,'e'},{0xCC,'i'},{0xCD,'i'},{0xCE,'i'},{0xCF,'i'},
    {0xD1,'n'},{0xD2,'o'},{0xD3,'o'},{0xD4,'o'},{0xD5,'o'},{0xD6,'o'},
    {0xD9,'u'},{0xDA,'u'},{0xDB,'u'},{0xDC,'u'},{0xDD,'y'},
    {0xE0,'a'},{0xE1,'a'},{0xE2,'a'},{0xE3,'a'},{0xE4,'a'},{0xE5,'a'},{0xE7,'c'},
    {0xE8,'e'},{0xE9,'e'},{0xEA,'e'},{0xEB,'e'},{0xEC,'i'},{0xED,'i'},{0xEE,'i'},{0xEF,'i'},
    {0xF1,'n'},{0xF2,'o'},{0xF3,'o'},{0xF4,'o'},{0xF5,'o'},{0xF6,'o'},
    {0xF9,'u'},{0xFA,'u'},{0xFB,'u'},{0xFC,'u'},{0xFD,'y'},{0xFF,'y'},
    {0x100,'a'},{0x101,'a'},{0x102,'a'},{0x103,'a'},{0x104,'a'},{0x105,'a'},
    {0x106,'c'},{0x107,'c'},{0x10C,'c'},{0x10D,'c'},{0x10E,'d'},{0x10F,'d'},
    {0x112,'e'},{0x113,'e'},{0x118,'e'},{0x119,'e'},{0x11A,'e'},{0x11B,'e'},
    {0x11E,'g'},{0x11F,'g'},{0x12A,'i'},{0x12B,'i'},{0x13D,'l'},{0x13E,'l'},
    {0x143,'n'},{0x144,'n'},{0x147,'n'},{0x148,'n'},{0x14C,'o'},{0x14D,'o'},{0x150,'o'},{0x151,'o'},
    {0x154,'r'},{0x155,'r'},{0x158,'r'},{0x159,'r'},{0x15A,'s'},{0x15B,'s'},
    {0x15E,'s'},{0x15F,'s'},{0x160,'s'},{0x161,'s'},{0x162,'t'},{0x163,'t'},{0x164,'t'},{0x165,'t'},
    {0x16A,'u'},{0x16B,'u'},{0x16E,'u'},{0x16F,'u'},{0x170,'u'},{0x171,'u'},
    {0x179,'z'},{0x17A,'z'},{0x17B,'z'},{0x17C,'z'},{0x17D,'z'},{0x17E,'z'},
    {0x218,'s'},{0x219,'s'},{0x21A,'t'},{0x21B,'t'}
};

static char plain_letter(int cp) {
    for (size_t i = 0; i < sizeof(ACCENTS) / sizeof(ACCENTS[0]); i++) {
        if (ACCENTS[i].cp == cp) return ACCENTS[i].base;
    }
    return 0;
}

static int is_punct(int cp) {
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return 1;
    if (cp == 0xA1 || cp == 0xA7 || cp == 0xAB || cp == 0xB6 || cp == 0xB7 || cp == 0xBB || cp == 0xBF) return 1;
    if (cp >= 0x2010 && cp <= 0x2027) return 1;
    if (cp >= 0x2030 && cp <= 0x205E) return 1;
    return 0;
}

static int is_space(int cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0xA0
        || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

static int is_dropped(int cp) { // control and invisible characters
    if (cp == 0 || cp == 0xFFFD || cp == 0xAD || cp == 0xFEFF) return 1;
    if (cp >= 0x200B && cp <= 0x200F) return 1;
    if (cp >= 0x202A && cp <= 0x202E) return 1;
    if (cp < 32 && cp != '\t' && cp != '\n' && cp != '\r') return 1;
    if (cp >= 0x7F && cp <= 0x9F) return 1;
    if (cp >= 0x300 && cp <= 0x36F) return 1; // combining accents
    return 0;
}

// Read one UTF-8 character, return how many bytes it uses.
static int next_char(const unsigned char* s, int* cp) {
    if (s[0] < 0x80) { *cp = s[0]; return 1; }
    if ((s[0] & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) {
        *cp = ((s[0] & 0x1F) << 6) | (s[1] & 0x3F); return 2;
    }
    if ((s[0] & 0xF0) == 0xE0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        *cp = ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); return 3;
    }
    if ((s[0] & 0xF8) == 0xF0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
        *cp = ((s[0] & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F); return 4;
    }
    *cp = 0xFFFD; // not valid UTF-8
    return 1;
}

static int write_char(char* out, int cp) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F)); return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

// Lowercase for the non-English letters that have no plain form above.
static int lower_char(int cp) {
    if (cp == 0xC6 || cp == 0xD0 || cp == 0xD8 || cp == 0xDE) return cp + 0x20; // AE, Eth, O with stroke, Thorn
    if (cp == 0x141 || cp == 0x152) return cp + 1;   // Polish L, OE
    if (cp >= 0x391 && cp <= 0x3A9) return cp + 32;  // Greek
    if (cp >= 0x400 && cp <= 0x40F) cp += 80;        // Cyrillic
    if (cp >= 0x410 && cp <= 0x42F) cp += 32;
    if (cp == 0x439) return 0x438;                   // Cyrillic letters that lose their accent
    if (cp == 0x451) return 0x435;
    return cp;
}

static int is_cjk(int cp) {
    return (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x4E00 && cp <= 0x9FFF)
        || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x20000 && cp <= 0x2FA1F);
}

static int push_id(int* ids, int n, int max_ids, int id) {
    if (n < max_ids) ids[n] = id;
    return n + 1;
}

// WordPiece: split one word into the longest pieces found in the vocabulary.
static int wordpiece(const Vocab* v, const char* word, int nchars, int* ids, int n, int max_ids) {
    int len = (int)strlen(word);
    if (nchars > 100) return push_id(ids, n, max_ids, v->unk_id);
    char* piece = malloc(len + 3);
    if (!piece) die("out of memory");
    int* found = malloc((len + 1) * sizeof(int));
    if (!found) die("out of memory");
    int nfound = 0, start = 0, bad = 0;
    while (start < len) {
        int end = len, id = -1;
        while (end > start) {
            if (start > 0) { piece[0] = '#'; piece[1] = '#'; memcpy(piece + 2, word + start, end - start); piece[end - start + 2] = '\0'; }
            else { memcpy(piece, word, end); piece[end] = '\0'; }
            id = vocab_find(v, piece);
            if (id >= 0) break;
            end--;
        }
        if (id < 0) { bad = 1; break; }
        found[nfound++] = id;
        start = end;
    }
    if (bad) n = push_id(ids, n, max_ids, v->unk_id);
    else for (int i = 0; i < nfound; i++) n = push_id(ids, n, max_ids, found[i]);
    free(piece);
    free(found);
    return n;
}

// Text to token ids, with [CLS] and [SEP]. Long texts are cut at max_len tokens.
static int tokenize(const Vocab* v, const char* text, int* ids, int max_len) {
    int max_ids = max_len - 2;
    int n = 0;
    size_t cap = strlen(text) * 2 + 8;
    char* word = malloc(cap);
    if (!word) die("out of memory");
    size_t w = 0;
    int nchars = 0;
    const unsigned char* s = (const unsigned char*)text;
    while (1) {
        int cp = 0;
        int used = *s ? next_char(s, &cp) : 0;
        if (*s == 0 || is_space(cp) || is_punct(cp) || is_cjk(cp)) {
            if (w > 0) {
                word[w] = '\0';
                n = wordpiece(v, word, nchars, ids + 1, n, max_ids);
                w = 0;
                nchars = 0;
            }
            if (*s == 0) break;
            if (is_punct(cp) || is_cjk(cp)) { // these become tokens of their own
                memcpy(word, s, used);
                word[used] = '\0';
                n = wordpiece(v, word, 1, ids + 1, n, max_ids);
            }
        } else if (!is_dropped(cp)) {
            char base = plain_letter(cp);
            if (base) { word[w++] = base; }
            else if (cp < 128) { word[w++] = (char)tolower(cp); }
            else { w += write_char(word + w, lower_char(cp)); }
            nchars++;
        }
        s += used;
    }
    free(word);
    if (n > max_ids) n = max_ids;
    ids[0] = v->cls_id;
    ids[n + 1] = v->sep_id;
    return n + 2;
}

// ----------------------------------------------------------------------------
// The neural network

// out[t] = W @ x[t] + b, for every token t
static void linear(float* out, const float* x, const float* w, const float* b, int n, int in_dim, int out_dim) {
    for (int t = 0; t < n; t++) {
        const float* xt = x + (size_t)t * in_dim;
        float* ot = out + (size_t)t * out_dim;
        for (int i = 0; i < out_dim; i++) {
            const float* row = w + (size_t)i * in_dim;
            float val = b[i];
            for (int j = 0; j < in_dim; j++) val += row[j] * xt[j];
            ot[i] = val;
        }
    }
}

static void layer_norm(float* x, const float* w, const float* b, int n, int dim) {
    for (int t = 0; t < n; t++) {
        float* xt = x + (size_t)t * dim;
        float mean = 0.0f, var = 0.0f;
        for (int i = 0; i < dim; i++) mean += xt[i];
        mean /= dim;
        for (int i = 0; i < dim; i++) var += (xt[i] - mean) * (xt[i] - mean);
        var /= dim;
        float scale = 1.0f / sqrtf(var + 1e-12f);
        for (int i = 0; i < dim; i++) xt[i] = (xt[i] - mean) * scale * w[i] + b[i];
    }
}

// Rank 0: word + position + type embeddings, then a layer norm.
static void embed_tokens(Model* m, const int* ids, int n, float* x) {
    int d = m->c.dim;
    for (int t = 0; t < n; t++) {
        const float* word = m->word_emb + (size_t)ids[t] * d;
        const float* pos = m->pos_emb + (size_t)t * d;
        float* xt = x + (size_t)t * d;
        for (int i = 0; i < d; i++) xt[i] = word[i] + pos[i] + m->type_emb[i];
    }
    layer_norm(x, m->emb_norm_w, m->emb_norm_b, n, d);
}

// Run the layers owned by this rank on x, in place.
static void forward_layers(Model* m, float* x, int n) {
    int d = m->c.dim, h = m->c.hidden_dim;
    int heads = m->c.n_heads, hs = d / heads;
    float scale = 1.0f / sqrtf((float)hs);

    for (int li = 0; li < m->n_local; li++) {
        const Layer* y = &m->layers[m->first_layer + li];
        linear(m->q, x, y->qw, y->qb, n, d, d);
        linear(m->k, x, y->kw, y->kb, n, d, d);
        linear(m->v, x, y->vw, y->vb, n, d, d);

        // multi-head self attention
        for (int a = 0; a < heads; a++) {
            for (int t = 0; t < n; t++) {
                const float* q = m->q + (size_t)t * d + a * hs;
                float best = -1e30f;
                for (int s = 0; s < n; s++) {
                    const float* k = m->k + (size_t)s * d + a * hs;
                    float score = 0.0f;
                    for (int i = 0; i < hs; i++) score += q[i] * k[i];
                    score *= scale;
                    m->att[s] = score;
                    if (score > best) best = score;
                }
                float sum = 0.0f;
                for (int s = 0; s < n; s++) { m->att[s] = expf(m->att[s] - best); sum += m->att[s]; }
                float* c = m->ctx + (size_t)t * d + a * hs;
                memset(c, 0, hs * sizeof(float));
                for (int s = 0; s < n; s++) {
                    const float* v = m->v + (size_t)s * d + a * hs;
                    float w = m->att[s] / sum;
                    for (int i = 0; i < hs; i++) c[i] += w * v[i];
                }
            }
        }

        // attention output, residual, norm
        linear(m->tmp, m->ctx, y->ow, y->ob, n, d, d);
        for (size_t i = 0; i < (size_t)n * d; i++) x[i] += m->tmp[i];
        layer_norm(x, y->n1w, y->n1b, n, d);

        // feed forward with GELU, residual, norm
        linear(m->hid, x, y->iw, y->ib, n, d, h);
        for (size_t i = 0; i < (size_t)n * h; i++) {
            float val = m->hid[i];
            m->hid[i] = 0.5f * val * (1.0f + erff(val / 1.41421356237f));
        }
        linear(m->tmp, m->hid, y->fw, y->fb, n, h, d);
        for (size_t i = 0; i < (size_t)n * d; i++) x[i] += m->tmp[i];
        layer_norm(x, y->n2w, y->n2b, n, d);
    }
}

// Last rank: average the tokens, or take the first one ([CLS]) if the model asks
// for that. Then scale the vector to length 1.
static void pool_and_normalize(const Model* m, const float* x, int n, float* out) {
    int d = m->c.dim;
    int used = m->c.magic == MAGIC_CLS ? 1 : n;
    double* sum = xcalloc(d, sizeof(double));
    for (int t = 0; t < used; t++) {
        for (int i = 0; i < d; i++) sum[i] += x[(size_t)t * d + i];
    }
    double norm = 0.0;
    for (int i = 0; i < d; i++) { sum[i] /= used; norm += sum[i] * sum[i]; }
    norm = sqrt(norm);
    if (norm < 1e-12) norm = 1e-12;
    for (int i = 0; i < d; i++) out[i] = (float)(sum[i] / norm);
    free(sum);
}

// ----------------------------------------------------------------------------
// The pipeline

static double t_compute = 0.0; // time spent on the model and the tokenizer
static double t_comm = 0.0;    // time inside MPI calls: transfer plus waiting for data
static long long g_tokens = 0;

// Live view (-t 1): every rank prints what it works on, from when to when, and what it sends.
static int g_trace = 0;
static double g_start = 0.0;
static char g_host[MPI_MAX_PROCESSOR_NAME];

static void trace_work(const Model* m, int text, int tokens, double t0, double t1, int dest, double kb) {
    char layers[32];
    if (m->n_local > 0) sprintf(layers, "%d-%d", m->first_layer, m->first_layer + m->n_local - 1);
    else sprintf(layers, "none");
    fprintf(stderr, "[%8.3f - %8.3f s] rank %d %s | text %d | %d tokens | layers %s | %.1f ms",
            t0 - g_start, t1 - g_start, g_rank, g_host, text, tokens, layers, (t1 - t0) * 1000);
    if (dest >= 0) fprintf(stderr, " | sends %.1f KB to rank %d", kb, dest);
    fprintf(stderr, "\n");
    fflush(stderr);
}

static void trace_received(int text) {
    fprintf(stderr, "[%8.3f s] rank %d %s | text %d | final vector received\n",
            MPI_Wtime() - g_start, g_rank, g_host, text);
    fflush(stderr);
}

// Rank 0: if a finished vector has arrived, store it. Results come back in the
// same order the texts went in, because every rank handles them one at a time.
static int try_receive(float* out, int d, int size, int* done) {
    int flag = 0;
    MPI_Iprobe(size - 1, TAG_DATA, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    if (flag) {
        MPI_Recv(out + (size_t)(*done) * d, d, MPI_FLOAT, size - 1, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (g_trace) trace_received(*done);
        (*done)++;
    }
    return flag;
}

// Rank 0: turn n texts into n vectors. At most `window` texts are in the pipeline.
static void embed_batch(Model* m, const Vocab* v, char** texts, int n, float* out, int window, int size) {
    int d = m->c.dim;
    float* x = xcalloc((size_t)m->max_len * d, sizeof(float));
    int* ids = xcalloc(m->max_len, sizeof(int));
    int next = 0, done = 0;

    while (done < n) {
        if (next < n && next - done < window) {
            double t0 = MPI_Wtime();
            int len = tokenize(v, texts[next], ids, m->max_len);
            embed_tokens(m, ids, len, x);
            forward_layers(m, x, len);
            g_tokens += len;
            if (size == 1) pool_and_normalize(m, x, len, out + (size_t)next * d);
            double t1 = MPI_Wtime();
            t_compute += t1 - t0;
            if (g_trace) trace_work(m, next, len, t0, t1, size == 1 ? -1 : 1, len * d * sizeof(float) / 1024.0);
            next++;
            if (size == 1) { done++; continue; }

            // Keep collecting results while the send is going on. This way
            // rank 0 can never block the ring of ranks, so it can't deadlock.
            t0 = MPI_Wtime();
            MPI_Request req;
            MPI_Isend(x, len * d, MPI_FLOAT, 1, TAG_DATA, MPI_COMM_WORLD, &req);
            int sent = 0;
            while (!sent) {
                MPI_Test(&req, &sent, MPI_STATUS_IGNORE);
                if (!sent) try_receive(out, d, size, &done);
            }
            t_comm += MPI_Wtime() - t0;
        } else {
            double t0 = MPI_Wtime();
            MPI_Recv(out + (size_t)done * d, d, MPI_FLOAT, size - 1, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (g_trace) trace_received(done);
            done++;
            t_comm += MPI_Wtime() - t0;
        }
    }
    free(x);
    free(ids);
}

// Ranks 1..N-1: receive x, run the local layers, pass the result on.
static void run_worker(Model* m, int rank, int size) {
    int d = m->c.dim;
    float* x = xcalloc((size_t)m->max_len * d, sizeof(float));
    float* vec = xcalloc(d, sizeof(float));
    int text = 0; // texts arrive in order, so counting them gives their number
    for (;;) {
        MPI_Status st;
        int count;
        double t0 = MPI_Wtime();
        MPI_Recv(x, m->max_len * d, MPI_FLOAT, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
        t_comm += MPI_Wtime() - t0;
        if (st.MPI_TAG == TAG_STOP) {
            if (rank < size - 1) MPI_Send(x, 0, MPI_FLOAT, rank + 1, TAG_STOP, MPI_COMM_WORLD);
            break;
        }
        MPI_Get_count(&st, MPI_FLOAT, &count);
        int n = count / d;

        t0 = MPI_Wtime();
        forward_layers(m, x, n);
        if (rank == size - 1) pool_and_normalize(m, x, n, vec);
        double t1 = MPI_Wtime();
        t_compute += t1 - t0;
        if (g_trace) {
            int last = rank == size - 1;
            trace_work(m, text, n, t0, t1, last ? 0 : rank + 1, (last ? d : n * d) * sizeof(float) / 1024.0);
        }
        text++;

        t0 = MPI_Wtime();
        if (rank == size - 1) MPI_Send(vec, d, MPI_FLOAT, 0, TAG_DATA, MPI_COMM_WORLD);
        else MPI_Send(x, n * d, MPI_FLOAT, rank + 1, TAG_DATA, MPI_COMM_WORLD);
        t_comm += MPI_Wtime() - t0;
    }
    free(x);
    free(vec);
}

// ----------------------------------------------------------------------------
// Measurements

// One-way latency between ranks 0 and 1 for a message of one token. In seconds.
static double measure_latency(int d, int rank, int size) {
    int rounds = 1000;
    if (size < 2 || rank > 1) return 0.0;
    float* buf = xcalloc(d, sizeof(float));
    double t0 = MPI_Wtime();
    for (int i = 0; i < rounds; i++) {
        if (rank == 0) {
            MPI_Send(buf, d, MPI_FLOAT, 1, 99, MPI_COMM_WORLD);
            MPI_Recv(buf, d, MPI_FLOAT, 1, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(buf, d, MPI_FLOAT, 0, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buf, d, MPI_FLOAT, 0, 99, MPI_COMM_WORLD);
        }
    }
    double result = (MPI_Wtime() - t0) / rounds / 2.0;
    free(buf);
    return result;
}

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

static double local_weights_mb(const Model* m, int rank) {
    const Config* c = &m->c;
    double params = (double)layer_params(c) * m->n_local;
    if (rank == 0) params += (double)c->vocab_size * c->dim + (double)c->max_pos * c->dim + 3.0 * c->dim;
    return params * sizeof(float) / (1024.0 * 1024.0);
}

// ----------------------------------------------------------------------------
// Input files and search

static char** read_lines(const char* path, int* count) {
    FILE* f = fopen(path, "r");
    if (!f) die("can't open the texts file");
    int cap = 1024, n = 0;
    char** lines = xcalloc(cap, sizeof(char*));
    char* line = NULL;
    size_t len = 0;
    while (getline(&line, &len, f) != -1) {
        size_t l = strlen(line);
        while (l > 0 && (line[l-1] == '\n' || line[l-1] == '\r')) line[--l] = '\0';
        if (l == 0) continue; // empty lines are skipped
        if (n == cap) {
            cap *= 2;
            lines = realloc(lines, cap * sizeof(char*));
            if (!lines) die("out of memory");
        }
        lines[n++] = strdup(line);
    }
    free(line);
    fclose(f);
    *count = n;
    return lines;
}

// The question with the model's recommended prefix in front, if it has one.
static char* with_prefix(const char* prefix, const char* text) {
    size_t a = prefix ? strlen(prefix) : 0, b = strlen(text);
    char* out = xcalloc(a + b + 2, 1);
    if (a > 0) { memcpy(out, prefix, a); out[a] = ' '; a++; }
    memcpy(out + a, text, b);
    return out;
}

static void print_matches(const float* query, const float* vectors, char** texts, int n, int d, int top_k) {
    int* best = xcalloc(top_k, sizeof(int));
    float* score = xcalloc(top_k, sizeof(float));
    int found = 0;
    for (int i = 0; i < n; i++) {
        float s = 0.0f; // both vectors have length 1, so this is the cosine similarity
        const float* v = vectors + (size_t)i * d;
        for (int j = 0; j < d; j++) s += query[j] * v[j];
        int p = found < top_k ? found : top_k - 1;
        if (found == top_k && s <= score[p]) continue;
        while (p > 0 && score[p-1] < s) { score[p] = score[p-1]; best[p] = best[p-1]; p--; }
        score[p] = s;
        best[p] = i;
        if (found < top_k) found++;
    }
    for (int i = 0; i < found; i++) printf("  %.3f  %s\n", score[i], texts[best[i]]);
    fflush(stdout);
    free(best);
    free(score);
}

static void error_usage(void) {
    if (g_rank == 0) {
        fprintf(stderr, "Index:   mpirun -np <ranks> embed_mpi <model.bin> <vocab.txt> -f texts.txt -o vectors.bin\n");
        fprintf(stderr, "Search:  mpirun -np <ranks> embed_mpi <model.bin> <vocab.txt> -f texts.txt -v vectors.bin -q \"question\"\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  -f <file>   texts, one per line\n");
        fprintf(stderr, "  -o <file>   index mode: where to save the vectors\n");
        fprintf(stderr, "  -r <file>   index mode: reference vectors to compare with\n");
        fprintf(stderr, "  -v <file>   search mode: the saved vectors\n");
        fprintf(stderr, "  -q <text>   search mode: the question. Use - to read questions from stdin\n");
        fprintf(stderr, "  -k <int>    search mode: results per question, default 5\n");
        fprintf(stderr, "  -l <list>   layers per rank, e.g. 3,3. Default: even split\n");
        fprintf(stderr, "  -w <int>    most texts in the pipeline at once, default 2 x ranks\n");
        fprintf(stderr, "  -m <int>    most tokens per text, default 256\n");
        fprintf(stderr, "  -p <text>   search mode: prefix added in front of every question\n");
        fprintf(stderr, "  -t 1        live view: every rank prints what it works on and what it sends\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(EXIT_FAILURE);
}

static void compute_split(const char* spec, int n_layers, int size, int* counts) {
    if (spec == NULL) {
        for (int r = 0; r < size; r++) counts[r] = n_layers / size + (r < n_layers % size ? 1 : 0);
        return;
    }
    char* copy = strdup(spec);
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    g_rank = rank;

    char *texts_path = NULL, *out_path = NULL, *ref_path = NULL, *vec_path = NULL, *query = NULL, *split = NULL, *prefix = NULL;
    int top_k = 5, window = 2 * size, max_len = 256;

    if (argc < 3) error_usage();
    char* model_path = argv[1];
    char* vocab_path = argv[2];
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-' || strlen(argv[i]) != 2) error_usage();
        char* val = argv[i + 1];
        switch (argv[i][1]) {
            case 'f': texts_path = val; break;
            case 'o': out_path = val; break;
            case 'r': ref_path = val; break;
            case 'v': vec_path = val; break;
            case 'q': query = val; break;
            case 'k': top_k = atoi(val); break;
            case 'l': split = val; break;
            case 'w': window = atoi(val); break;
            case 'm': max_len = atoi(val); break;
            case 'p': prefix = val; break;
            case 't': g_trace = atoi(val); break;
            default: error_usage();
        }
    }
    if (texts_path == NULL) error_usage();
    if (query != NULL && vec_path == NULL) error_usage();
    if (top_k < 1) top_k = 1;
    if (window < 1) window = 1;

    // only the header is needed for the split, the weights are mapped below
    Config head;
    FILE* cf = fopen(model_path, "rb");
    if (!cf || fread(&head, sizeof(Config), 1, cf) != 1) die("can't read the model file");
    fclose(cf);
    if (head.magic != MAGIC_MEAN && head.magic != MAGIC_CLS) die("this is not a file made by the export scripts");
    if (max_len < 3 || max_len > head.max_pos) die("-m is out of range");

    int* counts = xcalloc(size, sizeof(int));
    compute_split(split, head.n_layers, size, counts);
    int first_layer = 0;
    for (int r = 0; r < rank; r++) first_layer += counts[r];

    Model model;
    load_model(&model, model_path, first_layer, counts[rank], max_len);
    int d = model.c.dim;

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostname_len;
    memset(hostname, 0, sizeof(hostname));
    MPI_Get_processor_name(hostname, &hostname_len);
    memcpy(g_host, hostname, sizeof(g_host));

    double latency = measure_latency(d, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);
    g_start = MPI_Wtime();

    double wall = 0.0;
    int n_texts = 0;

    if (rank != 0) {
        double start = MPI_Wtime();
        run_worker(&model, rank, size);
        wall = MPI_Wtime() - start;
    } else {
        Vocab vocab;
        load_vocab(&vocab, vocab_path);
        char** texts = read_lines(texts_path, &n_texts);
        if (n_texts == 0) die("the texts file is empty");

        if (query == NULL) {
            // index: every text becomes a vector
            float* vectors = xcalloc((size_t)n_texts * d, sizeof(float));
            double start = MPI_Wtime();
            embed_batch(&model, &vocab, texts, n_texts, vectors, window, size);
            wall = MPI_Wtime() - start;
            if (out_path) {
                FILE* f = fopen(out_path, "wb");
                if (!f || fwrite(vectors, sizeof(float) * d, n_texts, f) != (size_t)n_texts) die("can't write the vectors file");
                fclose(f);
            }
            if (ref_path) {
                float* ref = xcalloc((size_t)n_texts * d, sizeof(float));
                FILE* f = fopen(ref_path, "rb");
                if (!f || fread(ref, sizeof(float) * d, n_texts, f) != (size_t)n_texts) die("can't read the reference file");
                fclose(f);
                double worst = 0.0;
                for (size_t i = 0; i < (size_t)n_texts * d; i++) {
                    double diff = fabs((double)vectors[i] - ref[i]);
                    if (diff > worst) worst = diff;
                }
                printf("largest difference from the reference: %.7f (%s)\n", worst, worst < 1e-3 ? "OK" : "TOO LARGE");
                free(ref);
            }
            free(vectors);
        } else {
            // search: load the saved vectors, embed the question, compare
            float* vectors = xcalloc((size_t)n_texts * d, sizeof(float));
            FILE* f = fopen(vec_path, "rb");
            if (!f || fread(vectors, sizeof(float) * d, n_texts, f) != (size_t)n_texts) die("the vectors file doesn't match the texts file");
            fclose(f);
            float* q_vec = xcalloc(d, sizeof(float));
            double start = MPI_Wtime();
            if (strcmp(query, "-") == 0) {
                char* line = NULL;
                size_t len = 0;
                printf("Type a question and press Enter. An empty line ends the program.\n");
                fflush(stdout);
                while (getline(&line, &len, stdin) != -1) {
                    size_t l = strlen(line);
                    while (l > 0 && (line[l-1] == '\n' || line[l-1] == '\r')) line[--l] = '\0';
                    if (l == 0) break;
                    double t0 = MPI_Wtime();
                    char* full = with_prefix(prefix, line);
                    embed_batch(&model, &vocab, &full, 1, q_vec, 1, size);
                    free(full);
                    double t1 = MPI_Wtime();
                    print_matches(q_vec, vectors, texts, n_texts, d, top_k);
                    printf("  (%.1f ms to embed, %.1f ms to search)\n", (t1 - t0) * 1000, (MPI_Wtime() - t1) * 1000);
                    fflush(stdout);
                }
                free(line);
            } else {
                double t0 = MPI_Wtime();
                char* full = with_prefix(prefix, query);
                embed_batch(&model, &vocab, &full, 1, q_vec, 1, size);
                free(full);
                double t1 = MPI_Wtime();
                print_matches(q_vec, vectors, texts, n_texts, d, top_k);
                printf("  (%.1f ms to embed, %.1f ms to search)\n", (t1 - t0) * 1000, (MPI_Wtime() - t1) * 1000);
                fflush(stdout);
            }
            wall = MPI_Wtime() - start;
            free(vectors);
            free(q_vec);
        }
        if (size > 1) {
            float empty = 0.0f;
            MPI_Send(&empty, 0, MPI_FLOAT, 1, TAG_STOP, MPI_COMM_WORLD);
        }
        for (int i = 0; i < n_texts; i++) free(texts[i]);
        free(texts);
    }

    // gather what each rank measured
    double mine[4] = { t_compute, t_comm, local_weights_mb(&model, rank), peak_rss_mb() };
    double* all = rank == 0 ? xcalloc(size, sizeof(mine)) : NULL;
    char* hosts = rank == 0 ? xcalloc(size, MPI_MAX_PROCESSOR_NAME) : NULL;
    MPI_Gather(mine, 4, MPI_DOUBLE, all, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hosts, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0 && query == NULL) {
        fflush(stdout);
        fprintf(stderr, "\n--- ranks: %d | window: %d | texts: %d | tokens: %lld | time: %.3f s | throughput: %.2f texts/s\n",
                size, window, n_texts, g_tokens, wall, n_texts / wall);
        if (size > 1) {
            fprintf(stderr, "--- one-way latency rank 0 <-> 1, %d bytes: %.1f us\n", d * (int)sizeof(float), latency * 1e6);
        }
        fprintf(stderr, "%-4s %-20s %-7s %11s %13s %12s %13s\n",
                "rank", "host", "layers", "compute (s)", "MPI+wait (s)", "weights (MB)", "peak RSS (MB)");
        int l = 0;
        for (int r = 0; r < size; r++) {
            char range[32];
            if (counts[r] > 0) sprintf(range, "%d-%d", l, l + counts[r] - 1);
            else sprintf(range, "none");
            fprintf(stderr, "%-4d %-20s %-7s %11.3f %13.3f %12.1f %13.1f\n",
                    r, hosts + r * MPI_MAX_PROCESSOR_NAME, range,
                    all[r*4], all[r*4+1], all[r*4+2], all[r*4+3]);
            l += counts[r];
        }
    }

    free(all);
    free(hosts);
    free(counts);
    MPI_Finalize();
    return 0;
}
