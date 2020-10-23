#pragma once

#include "octopus_structs.h"
#include <cstdint>

struct octopus_light {
  void *cache;
  uint64_t cache_size;
  uint64_t block_number;
};

using octopus_light_t = octopus_light *;

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
