#pragma once

#include "../octopus_params.h"

#include <cstring>
#include <vector>

template <int n> struct Precomputation {
  HOST Precomputation(u32 a, u32 b, u32 c, u32 w) {
    const u32 w2 = (u64)w * w % OCTOPUS_MOD;
    u32 wpow = 1, w2pow = 1;
    for (int i = 0; i < n; ++i) {
      x[i] = ((u64)a * w2pow + (u64)b * wpow + c) % OCTOPUS_MOD;
      wpow = (u64)wpow * w % OCTOPUS_MOD;
      w2pow = (u64)w2pow * w2 % OCTOPUS_MOD;
    }
  }

  u32 x[n];
};
