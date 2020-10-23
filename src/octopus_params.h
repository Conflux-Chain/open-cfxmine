#pragma once

#include <cstdint>

using u32 = uint32_t;
using u64 = uint64_t;

static const u32 OCTOPUS_ACCESSES = 64;

static inline u64 octopus_get_epoch(u64 block_number) {
  static const u64 OCTOPUS_EPOCH_LENGTH = 1 << 19;
  return block_number / OCTOPUS_EPOCH_LENGTH;
}

static const u32 OCTOPUS_MIX_BYTES = 128;
static const u32 OCTOPUS_HASH_BYTES = 64;
static const u32 OCTOPUS_DATASET_PARENTS = 256;
static const u32 OCTOPUS_CACHE_ROUNDS = 3;
static const u32 NODE_WORDS = OCTOPUS_HASH_BYTES / sizeof(int);
static const u32 MIX_WORDS = OCTOPUS_MIX_BYTES / sizeof(int);
static const u32 MIX_NODES = MIX_WORDS / NODE_WORDS;

static const u32 INIT_BLOCK_SIZE = 128;

static const u32 WARP_SIZE = 32;
static const u32 SEARCH_WARP_COUNT = 4;
static const u32 SEARCH_BLOCK_SIZE = WARP_SIZE * SEARCH_WARP_COUNT;

static const u64 OCTOPUS_NONCE_SEGMENT_BITS = 40;

// #define OCTOPUS_DEBUG
