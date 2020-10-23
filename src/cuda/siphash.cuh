#pragma once

#include "cuda_helper.cuh"

typedef uint2 sip64;

__inline__ __device__ sip64 rotl(const sip64 a, const int offset) {
  sip64 result;
  if (offset >= 32) {
    asm("shf.l.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.x)
        : "r"(a.x), "r"(a.y), "r"(offset));
    asm("shf.l.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.y)
        : "r"(a.y), "r"(a.x), "r"(offset));
  } else {
    asm("shf.l.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.x)
        : "r"(a.y), "r"(a.x), "r"(offset));
    asm("shf.l.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.y)
        : "r"(a.x), "r"(a.y), "r"(offset));
  }
  return result;
}

template <int rotE = 21> class diphash_state {
public:
  sip64 v0;
  sip64 v1;
  sip64 v2;
  sip64 v3;

  __device__ diphash_state(const uint2 *sk) {
    v0 = sk[0];
    v1 = sk[1];
    v2 = sk[2];
    v3 = sk[3];
  }

  __device__ uint64_t xor_lanes() { return devectorize((v0 ^ v1) ^ (v2 ^ v3)); }

  __device__ void dip_round() {
    v0 += v1;
    v2 += v3;
    v1 = rotl(v1, 13);
    v3 = rotl(v3, 16);
    v1 ^= v0;
    v3 ^= v2;
    v0 = rotl(v0, 32);
    v2 += v1;
    v0 += v3;
    v1 = rotl(v1, 17);
    v3 = rotl(v3, rotE);
    v1 ^= v2;
    v3 ^= v0;
    v2 = rotl(v2, 32);
  }

  __device__ void hash24(const uint64_t nonce) {
    v3 ^= vectorize(nonce);
    dip_round();
    dip_round();
    v0 ^= vectorize(nonce);
    v2 ^= vectorize(0xff);
    dip_round();
    dip_round();
    dip_round();
    dip_round();
  }
};
