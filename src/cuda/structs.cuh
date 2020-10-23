#pragma once

#include "../octopus_structs.h"

typedef union {
  uint2 uint2s[32 / sizeof(uint2)];
  uint4 uint4s[32 / sizeof(uint4)];
} hash32_t;

typedef union {
  uint32_t words[64 / sizeof(uint32_t)];
  uint2 uint2s[64 / sizeof(uint2)];
  uint4 uint4s[64 / sizeof(uint4)];
} hash64_t;

typedef struct {
  uint4 uint4s[128 / sizeof(uint4)];
} hash128_t;

typedef union {
  uint32_t words[200 / sizeof(uint32_t)];
  uint2 uint2s[200 / sizeof(uint2)];
  uint4 uint4s[200 / sizeof(uint4)];
} hash200_t;
