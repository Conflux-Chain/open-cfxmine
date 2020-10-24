#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#define DEV_INLINE __device__ __forceinline__

DEV_INLINE uint32_t lane_id() { return threadIdx.x & (WARP_SIZE - 1); }

DEV_INLINE uint32_t warp_id() { return threadIdx.x / WARP_SIZE; }

#if (__CUDACC_VER_MAJOR__ > 8)
#define SHFL(x, y, z) __shfl_sync(0xFFFFFFFF, (x), (y), (z))
#else
#define SHFL(x, y, z) __shfl((x), (y), (z))
#endif

#if (__CUDA_ARCH__ >= 320)
#define LDG(x) __ldg(&(x))
#else
#define LDG(x) (x)
#endif

#ifdef __CUDA_ARCH__
DEV_INLINE uint64_t cuda_swab64(const uint64_t x) {
  uint64_t result;
  uint2 t;
  asm("mov.b64 {%0,%1},%2; \n\t" : "=r"(t.x), "=r"(t.y) : "l"(x));
  t.x = __byte_perm(t.x, 0, 0x0123);
  t.y = __byte_perm(t.y, 0, 0x0123);
  asm("mov.b64 %0,{%1,%2}; \n\t" : "=l"(result) : "r"(t.y), "r"(t.x));
  return result;
}
#else
/* host */
#define cuda_swab64(x)                                                         \
  ((uint64_t)((((uint64_t)(x)&0xff00000000000000ULL) >> 56) |                  \
              (((uint64_t)(x)&0x00ff000000000000ULL) >> 40) |                  \
              (((uint64_t)(x)&0x0000ff0000000000ULL) >> 24) |                  \
              (((uint64_t)(x)&0x000000ff00000000ULL) >> 8) |                   \
              (((uint64_t)(x)&0x00000000ff000000ULL) << 8) |                   \
              (((uint64_t)(x)&0x0000000000ff0000ULL) << 24) |                  \
              (((uint64_t)(x)&0x000000000000ff00ULL) << 40) |                  \
              (((uint64_t)(x)&0x00000000000000ffULL) << 56)))
#endif

// 64-bit ROTATE RIGHT
#if __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 1
/* complicated sm >= 3.5 one (with Funnel Shifter beschleunigt), to bench */
DEV_INLINE uint64_t ROTR64(const uint64_t value, const int offset) {
  uint2 result;
  if (offset < 32) {
    asm("shf.r.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.x)
        : "r"(__double2loint(__longlong_as_double(value))),
          "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
    asm("shf.r.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.y)
        : "r"(__double2hiint(__longlong_as_double(value))),
          "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
  } else {
    asm("shf.r.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.x)
        : "r"(__double2hiint(__longlong_as_double(value))),
          "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
    asm("shf.r.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.y)
        : "r"(__double2loint(__longlong_as_double(value))),
          "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
  }
  return __double_as_longlong(__hiloint2double(result.y, result.x));
}
#elif __CUDA_ARCH__ >= 120 && USE_ROT_ASM_OPT == 2
DEV_INLINE uint64_t ROTR64(const uint64_t x, const int offset) {
  uint64_t result;
  asm("{\n\t"
      ".reg .b64 lhs;\n\t"
      ".reg .u32 roff;\n\t"
      "shr.b64 lhs, %1, %2;\n\t"
      "sub.u32 roff, 64, %2;\n\t"
      "shl.b64 %0, %1, roff;\n\t"
      "add.u64 %0, %0, lhs;\n\t"
      "}\n"
      : "=l"(result)
      : "l"(x), "r"(offset));
  return result;
}
#else
/* host */
#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))
#endif

// 64-bit ROTATE LEFT
#if __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 1
DEV_INLINE uint64_t ROTL64(const uint64_t value, const int offset) {
  uint2 result;
  if (offset >= 32) {
    asm("shf.l.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.x)
        : "r"(__double2loint(__longlong_as_double(value))),
          "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
    asm("shf.l.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.y)
        : "r"(__double2hiint(__longlong_as_double(value))),
          "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
  } else {
    asm("shf.l.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.x)
        : "r"(__double2hiint(__longlong_as_double(value))),
          "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
    asm("shf.l.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.y)
        : "r"(__double2loint(__longlong_as_double(value))),
          "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
  }
  return __double_as_longlong(__hiloint2double(result.y, result.x));
}
#elif __CUDA_ARCH__ >= 120 && USE_ROT_ASM_OPT == 2
DEV_INLINE uint64_t ROTL64(const uint64_t x, const int offset) {
  uint64_t result;
  asm("{\n\t"
      ".reg .b64 lhs;\n\t"
      ".reg .u32 roff;\n\t"
      "shl.b64 lhs, %1, %2;\n\t"
      "sub.u32 roff, 64, %2;\n\t"
      "shr.b64 %0, %1, roff;\n\t"
      "add.u64 %0, lhs, %0;\n\t"
      "}\n"
      : "=l"(result)
      : "l"(x), "r"(offset));
  return result;
}
#elif __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 3
__device__ uint64_t ROTL64(const uint64_t x, const int offset) {
  uint64_t res;
  asm("{\n\t"
      ".reg .u32 tl,th,vl,vh;\n\t"
      ".reg .pred p;\n\t"
      "mov.b64 {tl,th}, %1;\n\t"
      "shf.l.wrap.b32 vl, tl, th, %2;\n\t"
      "shf.l.wrap.b32 vh, th, tl, %2;\n\t"
      "setp.lt.u32 p, %2, 32;\n\t"
      "@!p mov.b64 %0, {vl,vh};\n\t"
      "@p  mov.b64 %0, {vh,vl};\n\t"
      "}"
      : "=l"(res)
      : "l"(x), "r"(offset));
  return res;
}
#else
/* host */
#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))
#endif

DEV_INLINE uint64_t devectorize(uint2 x) {
  uint64_t result;
  asm("mov.b64 %0,{%1,%2}; \n\t" : "=l"(result) : "r"(x.x), "r"(x.y));
  return result;
}

DEV_INLINE uint2 vectorize(const uint64_t x) {
  uint2 result;
  asm("mov.b64 {%0,%1},%2; \n\t" : "=r"(result.x), "=r"(result.y) : "l"(x));
  return result;
}

DEV_INLINE void devectorize2(uint4 inn, uint2 &x, uint2 &y) {
  x.x = inn.x;
  x.y = inn.y;
  y.x = inn.z;
  y.y = inn.w;
}

DEV_INLINE uint4 vectorize2(uint2 x, uint2 y) {
  uint4 result;
  result.x = x.x;
  result.y = x.y;
  result.z = y.x;
  result.w = y.y;

  return result;
}

static DEV_INLINE uint2 operator^(uint2 a, uint2 b) {
  return make_uint2(a.x ^ b.x, a.y ^ b.y);
}

static DEV_INLINE uint2 operator&(uint2 a, uint2 b) {
  return make_uint2(a.x & b.x, a.y & b.y);
}

static DEV_INLINE uint2 operator~(uint2 a) { return make_uint2(~a.x, ~a.y); }
static DEV_INLINE void operator^=(uint2 &a, uint2 b) { a = a ^ b; }
static DEV_INLINE uint2 operator+(uint2 a, uint2 b) {
  uint2 result;
  asm("{\n\t"
      "add.cc.u32 %0,%2,%4; \n\t"
      "addc.u32 %1,%3,%5;   \n\t"
      "}\n\t"
      : "=r"(result.x), "=r"(result.y)
      : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
  return result;
}

static DEV_INLINE void operator+=(uint2 &a, uint2 b) { a = a + b; }

// uint2 method
#if __CUDA_ARCH__ >= 350
DEV_INLINE uint2 ROR2(const uint2 a, const int offset) {
  uint2 result;
  if (offset < 32) {
    asm("shf.r.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.x)
        : "r"(a.x), "r"(a.y), "r"(offset));
    asm("shf.r.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.y)
        : "r"(a.y), "r"(a.x), "r"(offset));
  } else {
    asm("shf.r.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.x)
        : "r"(a.y), "r"(a.x), "r"(offset));
    asm("shf.r.wrap.b32 %0, %1, %2, %3;"
        : "=r"(result.y)
        : "r"(a.x), "r"(a.y), "r"(offset));
  }
  return result;
}
#else
DEV_INLINE uint2 ROR2(const uint2 v, const int n) {
  uint2 result;
  if (n <= 32) {
    result.y = ((v.y >> (n)) | (v.x << (32 - n)));
    result.x = ((v.x >> (n)) | (v.y << (32 - n)));
  } else {
    result.y = ((v.x >> (n - 32)) | (v.y << (64 - n)));
    result.x = ((v.y >> (n - 32)) | (v.x << (64 - n)));
  }
  return result;
}
#endif

DEV_INLINE uint32_t ROL8(const uint32_t x) { return __byte_perm(x, x, 0x2103); }
DEV_INLINE uint32_t ROL16(const uint32_t x) {
  return __byte_perm(x, x, 0x1032);
}
DEV_INLINE uint32_t ROL24(const uint32_t x) {
  return __byte_perm(x, x, 0x0321);
}

DEV_INLINE uint2 ROR8(const uint2 a) {
  uint2 result;
  result.x = __byte_perm(a.y, a.x, 0x0765);
  result.y = __byte_perm(a.y, a.x, 0x4321);

  return result;
}

DEV_INLINE uint2 ROR16(const uint2 a) {
  uint2 result;
  result.x = __byte_perm(a.y, a.x, 0x1076);
  result.y = __byte_perm(a.y, a.x, 0x5432);

  return result;
}

DEV_INLINE uint2 ROR24(const uint2 a) {
  uint2 result;
  result.x = __byte_perm(a.y, a.x, 0x2107);
  result.y = __byte_perm(a.y, a.x, 0x6543);

  return result;
}

DEV_INLINE uint2 ROL8(const uint2 a) {
  uint2 result;
  result.x = __byte_perm(a.y, a.x, 0x6543);
  result.y = __byte_perm(a.y, a.x, 0x2107);

  return result;
}

DEV_INLINE uint2 ROL16(const uint2 a) {
  uint2 result;
  result.x = __byte_perm(a.y, a.x, 0x5432);
  result.y = __byte_perm(a.y, a.x, 0x1076);

  return result;
}

DEV_INLINE uint2 ROL24(const uint2 a) {
  uint2 result;
  result.x = __byte_perm(a.y, a.x, 0x4321);
  result.y = __byte_perm(a.y, a.x, 0x0765);

  return result;
}

#if __CUDA_ARCH__ >= 350
__inline__ __device__ uint2 ROL2(const uint2 a, const int offset) {
  uint2 result;
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
#else
__inline__ __device__ uint2 ROL2(const uint2 v, const int n) {
  uint2 result;
  if (n <= 32) {
    result.y = ((v.y << (n)) | (v.x >> (32 - n)));
    result.x = ((v.x << (n)) | (v.y >> (32 - n)));
  } else {
    result.y = ((v.x << (n - 32)) | (v.y >> (64 - n)));
    result.x = ((v.y << (n - 32)) | (v.x >> (64 - n)));
  }
  return result;
}
#endif

DEV_INLINE uint64_t ROTR16(uint64_t x) {
#if __CUDA_ARCH__ > 500
  short4 temp;
  asm("mov.b64 { %0,  %1, %2, %3 }, %4; "
      : "=h"(temp.x), "=h"(temp.y), "=h"(temp.z), "=h"(temp.w)
      : "l"(x));
  asm("mov.b64 %0, {%1, %2, %3 , %4}; "
      : "=l"(x)
      : "h"(temp.y), "h"(temp.z), "h"(temp.w), "h"(temp.x));
  return x;
#else
  return ROTR64(x, 16);
#endif
}
DEV_INLINE uint64_t ROTL16(uint64_t x) {
#if __CUDA_ARCH__ > 500
  short4 temp;
  asm("mov.b64 { %0,  %1, %2, %3 }, %4; "
      : "=h"(temp.x), "=h"(temp.y), "=h"(temp.z), "=h"(temp.w)
      : "l"(x));
  asm("mov.b64 %0, {%1, %2, %3 , %4}; "
      : "=l"(x)
      : "h"(temp.w), "h"(temp.x), "h"(temp.y), "h"(temp.z));
  return x;
#else
  return ROTL64(x, 16);
#endif
}

static __forceinline__ __device__ uint2 SHL2(uint2 a, int offset) {
#if __CUDA_ARCH__ > 300
  uint2 result;
  if (offset < 32) {
    asm("{\n\t"
        "shf.l.clamp.b32 %1,%2,%3,%4; \n\t"
        "shl.b32 %0,%2,%4; \n\t"
        "}\n\t"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.x), "r"(a.y), "r"(offset));
  } else {
    asm("{\n\t"
        "shf.l.clamp.b32 %1,%2,%3,%4; \n\t"
        "shl.b32 %0,%2,%4; \n\t"
        "}\n\t"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.y), "r"(a.x), "r"(offset));
  }
  return result;
#else
  if (offset <= 32) {
    a.y = (a.y << offset) | (a.x >> (32 - offset));
    a.x = (a.x << offset);
  } else {
    a.y = (a.x << (offset - 32));
    a.x = 0;
  }
  return a;
#endif
}
static __forceinline__ __device__ uint2 SHR2(uint2 a, int offset) {
#if __CUDA_ARCH__ > 300
  uint2 result;
  if (offset < 32) {
    asm("{\n\t"
        "shf.r.clamp.b32 %0,%2,%3,%4; \n\t"
        "shr.b32 %1,%3,%4; \n\t"
        "}\n\t"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.x), "r"(a.y), "r"(offset));
  } else {
    asm("{\n\t"
        "shf.l.clamp.b32 %0,%2,%3,%4; \n\t"
        "shl.b32 %1,%3,%4; \n\t"
        "}\n\t"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.y), "r"(a.x), "r"(offset));
  }
  return result;
#else
  if (offset <= 32) {
    a.x = (a.x >> offset) | (a.y << (32 - offset));
    a.y = (a.y >> offset);
  } else {
    a.x = (a.y >> (offset - 32));
    a.y = 0;
  }
  return a;
#endif
}

DEV_INLINE uint32_t bfe(uint32_t x, uint32_t bit, uint32_t numBits) {
  uint32_t ret;
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(bit), "r"(numBits));
  return ret;
}