#pragma once

#include <cstdint>

using u32 = uint32_t;
using u64 = uint64_t;

// For the computation
static const u32 OCTOPUS_NK = 10;
static const u32 OCTOPUS_N = 1 << OCTOPUS_NK;
static const u32 OCTOPUS_MOD = 1032193;
static const u32 OCTOPUS_B = 11;

static const u32 WARP_SIZE = 32;
static const u32 OCTOPUS_PTK = OCTOPUS_NK - 5;
static const u32 OCTOPUS_DATA_PER_THREAD = 1 << OCTOPUS_PTK;

static const u32 OCTOPUS_ACCESSES = 32;

static inline u64 octopus_get_epoch(u64 block_number) {
  static const u64 OCTOPUS_EPOCH_LENGTH = 1 << 19;
  return block_number / OCTOPUS_EPOCH_LENGTH;
}

static const u32 OCTOPUS_MIX_BYTES = 256;
static const u32 OCTOPUS_HASH_BYTES = 64;
static const u32 OCTOPUS_DATASET_PARENTS = 256;
static const u32 OCTOPUS_CACHE_ROUNDS = 3;
static const u32 NODE_WORDS = OCTOPUS_HASH_BYTES / sizeof(int);
static const u32 MIX_WORDS = OCTOPUS_MIX_BYTES / sizeof(int);
static const u32 MIX_NODES = MIX_WORDS / NODE_WORDS;

static const u32 INIT_BLOCK_SIZE = 128;

static const u32 SEARCH_WARP_COUNT = 4;
static const u32 SEARCH_BLOCK_SIZE = WARP_SIZE * SEARCH_WARP_COUNT;

// #define OCTOPUS_DEBUG
