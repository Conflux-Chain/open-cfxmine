#include "common.h"
#include "params.cuh"
#include "structs.cuh"

#include "cuda_helper.cuh"
#include "fnv.cuh"
#include "globals.cuh"
#include "keccak.cuh"
#include "siphash.cuh"

#include <cstdint>
#include <cstdlib>
#include <iostream>

__device__ __forceinline__ bool compute_hash(const u64 nonce) {
  uint4 mix[PARALLEL_HASH];
  uint32_t offset[PARALLEL_HASH];
  uint32_t init0[PARALLEL_HASH];
  uint2 state[25];
  state[4] = vectorize(nonce);

  keccak_f1600_init(state);

  uint32_t D[OCTOPUS_ACCESSES];

  const int thread_id = threadIdx.x & (THREADS_PER_HASH - 1);
  const int mix_idx = thread_id & 3;

  for (int i = 0; i < THREADS_PER_HASH; i += PARALLEL_HASH) {
    for (int p = 0; p < PARALLEL_HASH; p++) {
      uint2 shuffle[8];
      for (int j = 0; j < 8; j++) {
        shuffle[j].x =
            __shfl_sync(0xFFFFFFFF, state[j].x, i + p, THREADS_PER_HASH);
        shuffle[j].y =
            __shfl_sync(0xFFFFFFFF, state[j].y, i + p, THREADS_PER_HASH);
      }
      switch (mix_idx) {
      case 0:
        mix[p] = vectorize2(shuffle[0], shuffle[1]);
        break;
      case 1:
        mix[p] = vectorize2(shuffle[2], shuffle[3]);
        break;
      case 2:
        mix[p] = vectorize2(shuffle[4], shuffle[5]);
        break;
      case 3:
        mix[p] = vectorize2(shuffle[6], shuffle[7]);
        break;
      }
      init0[p] = __shfl_sync(0xFFFFFFFF, shuffle[0].x, 0, THREADS_PER_HASH);
    }

    for (uint32_t a = 0; a < OCTOPUS_ACCESSES; a += 4) {
      int t = bfe(a, 2u, 3u);

      for (uint32_t b = 0; b < 4; b++) {
        for (int p = 0; p < PARALLEL_HASH; p++) {
          offset[p] =
              fnv(init0[p] ^ (a + b), ((uint32_t *)&mix[p])[b]) % d_dag_size;
          offset[p] = __shfl_sync(0xFFFFFFFF, offset[p], t, THREADS_PER_HASH);
        }
        uint32_t d = 0;
#pragma unroll
        for (int p = 0; p < PARALLEL_HASH; p++) {
          mix[p] = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]);
          uint32_t mix_xor = mix[p].x ^ mix[p].y ^ mix[p].z ^ mix[p].w;
          uint32_t reduced_xor =
              __shfl_sync(0xFFFFFFFF, mix_xor, 0, THREADS_PER_HASH);
          reduced_xor ^= __shfl_sync(0xFFFFFFFF, mix_xor, 1, THREADS_PER_HASH);
          reduced_xor ^= __shfl_sync(0xFFFFFFFF, mix_xor, 2, THREADS_PER_HASH);
          reduced_xor ^= __shfl_sync(0xFFFFFFFF, mix_xor, 3, THREADS_PER_HASH);
          reduced_xor ^= __shfl_sync(0xFFFFFFFF, mix_xor, 4, THREADS_PER_HASH);
          reduced_xor ^= __shfl_sync(0xFFFFFFFF, mix_xor, 5, THREADS_PER_HASH);
          reduced_xor ^= __shfl_sync(0xFFFFFFFF, mix_xor, 6, THREADS_PER_HASH);
          reduced_xor ^= __shfl_sync(0xFFFFFFFF, mix_xor, 7, THREADS_PER_HASH);
          if ((i + p) == thread_id) {
            d = reduced_xor;
          }
        }
        if (i <= thread_id && i + PARALLEL_HASH > thread_id) {
          D[a + b] = d;
        }
      }
    }

    for (int p = 0; p < PARALLEL_HASH; p++) {
      uint2 shuffle[4];
      uint32_t thread_mix = fnv_reduce(mix[p]);

      shuffle[0].x = __shfl_sync(0xFFFFFFFF, thread_mix, 0, THREADS_PER_HASH);
      shuffle[0].y = __shfl_sync(0xFFFFFFFF, thread_mix, 1, THREADS_PER_HASH);
      shuffle[1].x = __shfl_sync(0xFFFFFFFF, thread_mix, 2, THREADS_PER_HASH);
      shuffle[1].y = __shfl_sync(0xFFFFFFFF, thread_mix, 3, THREADS_PER_HASH);
      shuffle[2].x = __shfl_sync(0xFFFFFFFF, thread_mix, 4, THREADS_PER_HASH);
      shuffle[2].y = __shfl_sync(0xFFFFFFFF, thread_mix, 5, THREADS_PER_HASH);
      shuffle[3].x = __shfl_sync(0xFFFFFFFF, thread_mix, 6, THREADS_PER_HASH);
      shuffle[3].y = __shfl_sync(0xFFFFFFFF, thread_mix, 7, THREADS_PER_HASH);

      if ((i + p) == thread_id) {
        state[8] = shuffle[0];
        state[9] = shuffle[1];
        state[10] = shuffle[2];
        state[11] = shuffle[3];
      }
    }
  }

  const uint32_t *words = reinterpret_cast<const uint32_t *>(d_header.uint4s);

  static const int mod = 1000000000 + 7;
  const uint32_t A = words[0] % mod;
  const uint32_t B = words[1] % mod;
  const uint32_t C = words[2] % mod;
  const uint32_t W = words[3];

  const uint32_t W2 = static_cast<uint64_t>(W) * W % mod;

  uint32_t reduction = 0;
  uint32_t p_powers = C, q_powers = B;
  for (int i = 0; i < OCTOPUS_ACCESSES; ++i) {
    uint32_t x = p_powers + q_powers;
    if (x >= mod) {
      x -= mod;
    }
    x += A;
    if (x >= mod) {
      x -= mod;
    }
    uint32_t magic_mix = 0;
    uint64_t power = 1;
#pragma unroll
    for (int j = 0; j < OCTOPUS_ACCESSES; ++j) {
      uint64_t term = (power * D[j]) % mod;
      power = (power * x) % mod;
      magic_mix += term;
      if (magic_mix >= mod) {
        magic_mix -= mod;
      }
    }
    reduction = reduction * FNV_PRIME ^ magic_mix;
    p_powers = static_cast<uint64_t>(p_powers) * W2 % mod;
    q_powers = static_cast<uint64_t>(q_powers) * W % mod;
  }
  state[12].x = reduction;

  SHA3_256(state);

  for (int i = 0; i < 4; ++i) {
    auto a = cuda_swab64(devectorize(state[i]));
    auto b = devectorize(d_boundary.uint2s[i]);
    if (a != b) {
      return a < b;
    }
  }
  return true;
}

extern "C" {

__global__ void __launch_bounds__(INIT_BLOCK_SIZE)
    InitDagItems(uint32_t start) {
  uint32_t const node_index = start + blockIdx.x * blockDim.x + threadIdx.x;
  if ((node_index & ~3) >= d_dag_size * MIX_NODES) {
    return;
  }

  hash200_t dag_node;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    dag_node.uint4s[i] = d_light[node_index % d_light_size].uint4s[i];
  }
  dag_node.words[0] ^= node_index;
  SHA3_512(dag_node.uint2s);

  const int thread_id = threadIdx.x & 3;

  for (uint32_t i = 0; i != OCTOPUS_DATASET_PARENTS; ++i) {
    uint32_t parent_index =
        fnv(node_index ^ i, dag_node.words[i % NODE_WORDS]) % d_light_size;
    for (uint32_t t = 0; t < 4; t++) {
      uint32_t shuffle_index = SHFL(parent_index, t, 4);
      uint4 p4 = d_light[shuffle_index].uint4s[thread_id];
      for (int w = 0; w < 4; w++) {
        uint4 s4 = make_uint4(SHFL(p4.x, w, 4), SHFL(p4.y, w, 4),
                              SHFL(p4.z, w, 4), SHFL(p4.w, w, 4));
        if (t == thread_id) {
          dag_node.uint4s[w] = fnv4(dag_node.uint4s[w], s4);
        }
      }
    }
  }
  SHA3_512(dag_node.uint2s);
  hash64_t *dag_nodes = (hash64_t *)d_dag;

  for (uint32_t t = 0; t < 4; t++) {
    uint32_t shuffle_index = SHFL(node_index, t, 4);
    uint4 s[4];
    for (uint32_t w = 0; w < 4; w++) {
      s[w] = make_uint4(
          SHFL(dag_node.uint4s[w].x, t, 4), SHFL(dag_node.uint4s[w].y, t, 4),
          SHFL(dag_node.uint4s[w].z, t, 4), SHFL(dag_node.uint4s[w].w, t, 4));
    }
    if (shuffle_index < d_dag_size * MIX_NODES) {
      dag_nodes[shuffle_index].uint4s[thread_id] = s[thread_id];
    }
  }
}

__launch_bounds__(SEARCH_BLOCK_SIZE) __global__
    void Compute(u64 start_nonce, SearchResults *results) {
  const u64 offset = blockIdx.x * SEARCH_BLOCK_SIZE + threadIdx.x;
  if (compute_hash(start_nonce + offset)) {
    u32 index = atomicInc(reinterpret_cast<u32 *>(&results->count), 0xffffffff);
    if (index < MAX_SEARCH_RESULTS) {
      results->result[index].nonce_offset = offset;
    }
  }
}
}
