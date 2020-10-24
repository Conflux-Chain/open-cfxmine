#pragma once

#include <cstdint>

typedef struct octopus_h256 {
  uint8_t b[32];
} octopus_h256_t;

typedef struct octopus_return_value {
  octopus_h256_t result;
  bool success;
} octopus_return_value_t;

static const uint32_t MAX_SEARCH_RESULTS = 4;

struct SearchResult {
  uint32_t nonce_offset;
  uint32_t pad[1];
};

struct SearchResults {
  SearchResult result[MAX_SEARCH_RESULTS];
  uint32_t count = 0;
};