#pragma once

#include "../fnv.h"
#include "cuda_helper.cuh"

DEV_INLINE uint2 fnv2(uint2 a, uint2 b) {
  uint2 c;
  c.x = a.x * FNV_PRIME ^ b.x;
  c.y = a.y * FNV_PRIME ^ b.y;
  return c;
}

DEV_INLINE uint4 fnv4(uint4 a, uint4 b) {
  uint4 c;
  c.x = a.x * FNV_PRIME ^ b.x;
  c.y = a.y * FNV_PRIME ^ b.y;
  c.z = a.z * FNV_PRIME ^ b.z;
  c.w = a.w * FNV_PRIME ^ b.w;
  return c;
}

DEV_INLINE uint32_t fnv_reduce(uint4 v) {
  return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}
