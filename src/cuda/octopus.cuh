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

__device__ __forceinline__ bool chase_pointer(const u64 seed, u32* block_d) {
  uint2 state[25];
  state[4] = vectorize(seed);

  u32 *const warp_d = block_d + warp_id() * OCTOPUS_N;

  keccak_f1600_init(state);

  // Threads work together in this phase in groups of 8.
  const int thread_id = threadIdx.x & (THREADS_PER_HASH - 1);
  const int mix_idx = thread_id & 3;

  for (int i = 0; i < THREADS_PER_HASH; i += PARALLEL_HASH) {
    uint4 mix[PARALLEL_HASH];
    uint32_t offset[PARALLEL_HASH];
    uint32_t init0[PARALLEL_HASH];
    u32 lid = (threadIdx.x & (WARP_SIZE - 1) & (~(THREADS_PER_HASH - 1))) | i;

    // share init among threads
    for (int p = 0; p < PARALLEL_HASH; p++) {
      uint2 shuffle[8];
      for (int j = 0; j < 8; j++) {
        shuffle[j].x = SHFL(state[j].x, i + p, THREADS_PER_HASH);
        shuffle[j].y = SHFL(state[j].y, i + p, THREADS_PER_HASH);
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
      init0[p] = SHFL(shuffle[0].x, 0, THREADS_PER_HASH);
    }

    for (uint32_t a = 0; a < OCTOPUS_ACCESSES; a += 4) {
      int t = bfe(a, 2u, 3u);

      for (uint32_t b = 0; b < 4; b++) {
        for (int p = 0; p < PARALLEL_HASH; p++) {
          offset[p] =
              fnv(init0[p] ^ (a + b) ^ warp_d[(a + b) * WARP_SIZE + (lid + p)], ((uint32_t *)&mix[p])[b]) % d_dag_size;
          offset[p] = SHFL(offset[p], t, THREADS_PER_HASH);
          mix[p] = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]);
        }
      }
    }

    for (int p = 0; p < PARALLEL_HASH; p++) {
      uint2 shuffle[8];
      uint32_t thread_mix = fnv_reduce(mix[p]);

      // update mix across threads
      shuffle[0].x = SHFL(thread_mix, 0, THREADS_PER_HASH);
      shuffle[0].y = SHFL(thread_mix, 1, THREADS_PER_HASH);
      shuffle[1].x = SHFL(thread_mix, 2, THREADS_PER_HASH);
      shuffle[1].y = SHFL(thread_mix, 3, THREADS_PER_HASH);
      shuffle[2].x = SHFL(thread_mix, 4, THREADS_PER_HASH);
      shuffle[2].y = SHFL(thread_mix, 5, THREADS_PER_HASH);
      shuffle[3].x = SHFL(thread_mix, 6, THREADS_PER_HASH);
      shuffle[3].y = SHFL(thread_mix, 7, THREADS_PER_HASH);
      shuffle[4].x = SHFL(thread_mix, 8, THREADS_PER_HASH);
      shuffle[4].y = SHFL(thread_mix, 9, THREADS_PER_HASH);
      shuffle[5].x = SHFL(thread_mix, 10, THREADS_PER_HASH);
      shuffle[5].y = SHFL(thread_mix, 11, THREADS_PER_HASH);
      shuffle[6].x = SHFL(thread_mix, 12, THREADS_PER_HASH);
      shuffle[6].y = SHFL(thread_mix, 13, THREADS_PER_HASH);
      shuffle[7].x = SHFL(thread_mix, 14, THREADS_PER_HASH);
      shuffle[7].y = SHFL(thread_mix, 15, THREADS_PER_HASH);

      if ((i + p) == thread_id) {
        // move mix into state:
        state[8] = fnv2(shuffle[0], shuffle[4]);
        state[9] = fnv2(shuffle[1], shuffle[5]);
        state[10] = fnv2(shuffle[2], shuffle[6]);
        state[11] = fnv2(shuffle[3], shuffle[7]);
      }
    }
  }

  return cuda_swab64(keccak_f1600_final(state)) <=
         devectorize(d_boundary.uint2s[0]);
}

extern "C" {

__global__ void __launch_bounds__(INIT_BLOCK_SIZE)
    InitDagItems(uint32_t start) {
  const uint32_t node_index = start + blockIdx.x * blockDim.x + threadIdx.x;
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

__device__ __forceinline__ u64 multi_eval(u64 start_nonce, u32* block_d) {
  const u64 nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;
  const u32 lid = lane_id();
  const u32 wid = warp_id();

  u32 *const warp_d = block_d + wid * OCTOPUS_N;

  diphash_state<> dstate(d_header.uint2s);
  dstate.hash24(nonce);
  for (u32 i = 0, index = lid; i < OCTOPUS_DATA_PER_THREAD;
       ++i, index += WARP_SIZE) {
    dstate.dip_round();
    warp_d[index] = dstate.xor_lanes() & UINT32_MAX;
  }

  u64 thread_result = 0;
  u32 pv[OCTOPUS_DATA_PER_THREAD];
#pragma unroll
  for (u32 i = 0, index = lid; i < OCTOPUS_DATA_PER_THREAD;
       ++i, index += WARP_SIZE) {
    const u32 x = d_x[index];
    pv[i] = 0;
    for (u32 j = OCTOPUS_N; j--;) {
      pv[i] = ((u64)pv[i] * x + warp_d[j]) % OCTOPUS_MOD;
    }
    thread_result = fnv(thread_result, pv[i]);
  }
  __syncwarp();
#pragma unroll
  for (u32 i = 0, index = lid; i < OCTOPUS_DATA_PER_THREAD; ++i, index += WARP_SIZE) {
      warp_d[index] = pv[i];
  }
  return thread_result;
}

__launch_bounds__(SEARCH_BLOCK_SIZE) __global__
    void Compute(u64 start_nonce, SearchResults *results) {
  __shared__ u32 block_d[OCTOPUS_N * SEARCH_WARP_COUNT];

  const u64 thread_result = multi_eval(start_nonce, block_d);
  if (chase_pointer(thread_result, block_d)) {
    u32 index = atomicInc(reinterpret_cast<u32 *>(&results->count), 0xffffffff);
    if (index < MAX_SEARCH_RESULTS) {
      results->result[index].nonce_offset =
          blockIdx.x * SEARCH_BLOCK_SIZE + threadIdx.x;
    }
  }
}
}
