#pragma once

#include "octopus_structs.h"
#include <cstdint>
#include <utility>
#include <vector>

struct octopus_light {
  void *cache;
  uint64_t cache_size;
  uint64_t block_number;
};

using octopus_light_t = octopus_light *;

struct OctopusABCW {
  OctopusABCW(const octopus_h256_t header_hash);

  uint32_t a, b, c, w;
};

std::pair<uint64_t, std::vector<uint32_t>>
multi_eval(const octopus_h256_t header_hash, const uint64_t nonce);

uint64_t octopus_get_cachesize(const uint64_t block_number);
uint64_t octopus_get_datasize(const uint64_t block_number);

octopus_light_t octopus_light_new(uint64_t block_number);
void octopus_light_delete(octopus_light_t light);
void compute_d(const octopus_h256_t header, uint64_t nonce, uint32_t *d);
octopus_return_value_t octopus_light_compute(octopus_light_t light,
                                             const octopus_h256_t header_hash,
                                             uint64_t nonce);
bool octopus_check_difficulty(const octopus_h256_t *hash,
                              const octopus_h256_t *boundary);
