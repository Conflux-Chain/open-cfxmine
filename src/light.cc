#include "light.h"
#include "fnv.h"
#include "octopus_params.h"
#include "octopus_structs.h"
#include "sha3.h"
#include "siphash.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

static inline bool isprime(uint64_t x) {
  for (uint64_t i = 2; i * i <= x; ++i) {
    if (x % i == 0)
      return false;
  }
  return true;
}

static inline uint8_t octopus_h256_get(const octopus_h256_t *hash,
                                       unsigned int i) {
  return hash->b[i];
}

static inline void octopus_h256_set(octopus_h256_t *hash, unsigned int i,
                                    uint8_t v) {
  hash->b[i] = v;
}

static inline void octopus_h256_reset(octopus_h256_t *hash) {
  memset(hash, 0, 32);
}

static inline octopus_h256_t octopus_get_seedhash(uint64_t block_number) {
  octopus_h256_t ret;
  octopus_h256_reset(&ret);
  const uint64_t epochs = octopus_get_epoch(block_number);
  for (uint32_t i = 0; i < epochs; ++i) {
    SHA3_256(&ret, (uint8_t *)&ret, 32);
  }
  return ret;
}

union node {
  uint8_t bytes[NODE_WORDS * 4];
  uint32_t words[NODE_WORDS];
  uint64_t double_words[NODE_WORDS / 2];
};

// Follows Sergio's "STRICT MEMORY HARD HASHING FUNCTIONS" (2014)
// https://bitslog.files.wordpress.com/2013/12/memohash-v0-3.pdf
// SeqMemoHash(s, R, N)
static inline bool octopus_compute_cache_nodes(node *const nodes,
                                               uint64_t cache_size,
                                               const octopus_h256_t *seed) {
  if (cache_size % sizeof(node) != 0) {
    return false;
  }
  const uint32_t num_nodes = (uint32_t)(cache_size / sizeof(node));

  SHA3_512(nodes[0].bytes, (uint8_t *)seed, 32);

  for (uint32_t i = 1; i != num_nodes; ++i) {
    SHA3_512(nodes[i].bytes, nodes[i - 1].bytes, 64);
  }

  for (uint32_t j = 0; j != OCTOPUS_CACHE_ROUNDS; j++) {
    for (uint32_t i = 0; i != num_nodes; i++) {
      const uint32_t idx = nodes[i].words[0] % num_nodes;
      node data;
      data = nodes[(num_nodes - 1 + i) % num_nodes];
      for (uint32_t w = 0; w != NODE_WORDS; ++w) {
        data.words[w] ^= nodes[idx].words[w];
      }
      SHA3_512(nodes[i].bytes, data.bytes, sizeof(data));
    }
  }

  return true;
}

static inline octopus_light_t
octopus_light_new_internal(uint64_t cache_size, const octopus_h256_t *seed) {
  struct octopus_light *ret;
  node *nodes = nullptr;
  ret = reinterpret_cast<octopus_light *>(calloc(sizeof(*ret), 1));
  if (!ret) {
    return NULL;
  }
  ret->cache = malloc((size_t)cache_size);
  if (!ret->cache) {
    goto fail_free_light;
  }
  nodes = (node *)ret->cache;
  if (!octopus_compute_cache_nodes(nodes, cache_size, seed)) {
    goto fail_free_cache_mem;
  }
  ret->cache_size = cache_size;
  return ret;

fail_free_cache_mem:
  free(ret->cache);
fail_free_light:
  free(ret);
  return NULL;
}

static inline void octopus_calculate_dag_item(node *const ret,
                                              uint32_t node_index,
                                              const octopus_light_t light) {
  uint32_t num_parent_nodes = (uint32_t)(light->cache_size / sizeof(node));
  node const *cache_nodes = (node const *)light->cache;
  node const *init = &cache_nodes[node_index % num_parent_nodes];
  memcpy(ret, init, sizeof(node));
  ret->words[0] ^= node_index;
  SHA3_512(ret->bytes, ret->bytes, sizeof(node));
  for (uint32_t i = 0; i != OCTOPUS_DATASET_PARENTS; ++i) {
    uint32_t parent_index =
        fnv(node_index ^ i, ret->words[i % NODE_WORDS]) % num_parent_nodes;
    node const *parent = &cache_nodes[parent_index];

    for (unsigned w = 0; w != NODE_WORDS; ++w) {
      ret->words[w] = fnv(ret->words[w], parent->words[w]);
    }
  }
  SHA3_512(ret->bytes, ret->bytes, sizeof(node));
}

static inline bool octopus_hash(octopus_return_value_t *ret,
                                const octopus_light_t light, uint64_t full_size,
                                const octopus_h256_t header_hash,
                                const uint64_t nonce) {
  if (full_size % MIX_WORDS != 0) {
    return false;
  }

  node s_mix[MIX_NODES + 1];
  memcpy(s_mix[0].bytes, &header_hash, 32);
  s_mix[0].double_words[4] = nonce;
  SHA3_512(s_mix->bytes, s_mix->bytes, 40);

  node *const mix = s_mix + 1;
  for (uint32_t w = 0; w != MIX_WORDS; ++w) {
    mix->words[w] = s_mix[0].words[w % NODE_WORDS];
  }

  unsigned const page_size = sizeof(uint32_t) * MIX_WORDS;
  unsigned const num_full_pages = (unsigned)(full_size / page_size);

  uint32_t *words =
      const_cast<uint32_t *>(reinterpret_cast<const uint32_t *>(header_hash.b));
  uint32_t A = words[0];
  uint32_t B = words[1];
  uint32_t C = words[2];
  uint32_t W = words[3];
  uint32_t D[OCTOPUS_ACCESSES];

  for (unsigned i = 0; i != OCTOPUS_ACCESSES; ++i) {
    uint32_t const index =
        fnv(s_mix->words[0] ^ i, mix->words[i % MIX_WORDS]) % num_full_pages;
    D[i] = 0;
    for (unsigned n = 0; n != MIX_NODES; ++n) {
      node dag_node;
      octopus_calculate_dag_item(&dag_node, index * MIX_NODES + n, light);
      for (unsigned w = 0; w != NODE_WORDS; ++w) {
        mix[n].words[w] = fnv(mix[n].words[w], dag_node.words[w]);
        D[i] ^= mix[n].words[w];
      }
    }
  }

  for (uint32_t w = 0; w != MIX_WORDS; w += 4) {
    uint32_t reduction = mix->words[w + 0];
    reduction = reduction * FNV_PRIME ^ mix->words[w + 1];
    reduction = reduction * FNV_PRIME ^ mix->words[w + 2];
    reduction = reduction * FNV_PRIME ^ mix->words[w + 3];
    mix->words[w / 4] = reduction;
  }

  static const int mod = 1000000000 + 7;

  uint32_t magic_mix[OCTOPUS_ACCESSES];
  for (int i = 0; i < OCTOPUS_ACCESSES; ++i) {
    uint32_t x = 0;
    uint64_t p = C, q = B;
    for (int j = 0; j < i; ++j) {
      p = ((p * W) % mod * W) % mod;
      q = (q * W) % mod;
    }
    x = (p + q + A) % mod;
    magic_mix[i] = 0;
    uint64_t power = 1;
    for (int j = 0; j < OCTOPUS_ACCESSES; ++j) {
      uint64_t term = (power * D[j]) % mod;
      power = (power * x) % mod;
      magic_mix[i] = (magic_mix[i] + term) % mod;
    }
  }

  uint32_t reduction = 0;
  for (uint32_t i = 0; i < OCTOPUS_ACCESSES; ++i) {
    reduction = reduction * FNV_PRIME ^ magic_mix[i];
  }
  s_mix->words[24] = reduction;

  SHA3_256(&ret->result, s_mix->bytes, 64 + 32 + 4);
  return true;
}

static inline octopus_return_value_t
octopus_light_compute_internal(octopus_light_t light, uint64_t full_size,
                               const octopus_h256_t header_hash,
                               uint64_t nonce) {
  octopus_return_value_t ret;
  ret.success = true;
  if (!octopus_hash(&ret, light, full_size, header_hash, nonce)) {
    ret.success = false;
  }
  return ret;
}
} // namespace

uint64_t octopus_get_cachesize(const uint64_t block_number) {
  static const uint64_t OCTOPUS_CACHE_BYTES_INIT = 3 * (1 << 23);
  static const uint64_t OCTOPUS_CACHE_BYTES_GROWTH = 1 << 16;

  uint64_t sz = OCTOPUS_CACHE_BYTES_INIT +
                OCTOPUS_CACHE_BYTES_GROWTH * octopus_get_epoch(block_number);
  sz -= OCTOPUS_HASH_BYTES;
  while (!isprime(sz / OCTOPUS_HASH_BYTES)) {
    sz -= 2 * OCTOPUS_HASH_BYTES;
  }
  return sz;
}

uint64_t octopus_get_datasize(const uint64_t block_number) {
  static const uint64_t OCTOPUS_DATASET_BYTES_INIT =
      3 * (static_cast<uint64_t>(1) << 31);
  static const uint64_t OCTOPUS_DATASET_BYTES_GROWTH = 1 << 24;

  uint64_t sz = OCTOPUS_DATASET_BYTES_INIT +
                OCTOPUS_DATASET_BYTES_GROWTH * octopus_get_epoch(block_number);
  sz -= OCTOPUS_MIX_BYTES;
  while (!isprime(sz / OCTOPUS_MIX_BYTES)) {
    sz -= 2 * OCTOPUS_MIX_BYTES;
  }
  return sz;
}

octopus_light_t octopus_light_new(uint64_t block_number) {
  octopus_h256_t seedhash = octopus_get_seedhash(block_number);
  octopus_light_t ret;
  ret = octopus_light_new_internal(octopus_get_cachesize(block_number),
                                   &seedhash);
  ret->block_number = block_number;
  return ret;
}

void octopus_light_delete(octopus_light_t light) {
  if (light->cache) {
    free(light->cache);
  }
  free(light);
}

octopus_return_value_t octopus_light_compute(octopus_light_t light,
                                             const octopus_h256_t header_hash,
                                             uint64_t nonce) {
  uint64_t full_size = octopus_get_datasize(light->block_number);
  return octopus_light_compute_internal(light, full_size, header_hash, nonce);
}

bool octopus_check_difficulty(const octopus_h256_t *hash,
                              const octopus_h256_t *boundary) {
  // Boundary is big endian
  for (int i = 0; i < 32; i++) {
    if (octopus_h256_get(hash, i) != octopus_h256_get(boundary, i)) {
      return octopus_h256_get(hash, i) < octopus_h256_get(boundary, i);
    }
  }
  return true;
}
