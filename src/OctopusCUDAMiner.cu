#include "OctopusCUDAMiner.h"
#include "StratumClient.h"
#include "cuda/octopus.cuh"
#include "hex.h"
#include "light.h"
#include "octopus_params.h"
#include "octopus_structs.h"

#include <functional>
#include <iostream>

#define checkCudaErrors(call)                                                  \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "CUDA error RUNTIME: '%d' in func '%s' line %d\n", err,  \
              __FUNCTION__, __LINE__);                                         \
      abort();                                                                 \
    }                                                                          \
  } while (0)

class CUDADagManager {
public:
  void reset(uint64_t blockHeight) {
    dagSize = octopus_get_datasize(blockHeight);
    dagNumItems = dagSize / OCTOPUS_MIX_BYTES;
    lightSize = octopus_get_cachesize(blockHeight);
    lightNumItems = lightSize / OCTOPUS_HASH_BYTES;
    if (memoryDagSize < dagSize) {
      if (h_dag) {
        checkCudaErrors(cudaFree(h_dag));
      }
      {
        cudaError_t err = cudaMalloc(&h_dag, dagSize);
        if (cudaSuccess != err) {
          if (cudaErrorMemoryAllocation == err) {
            fprintf(stderr, "cudaMalloc failed. Reason: Insufficient memory\n");
          } else {
            fprintf(stderr, "CUDA error RUNTIME: '%d' in func '%s' line %d",
                    err, __FUNCTION__, __LINE__);
          }
          abort();
        }
      }
      checkCudaErrors(cudaMemcpyToSymbol(d_dag, &h_dag, sizeof(void *)));
      memoryDagSize = dagSize;
    }
    checkCudaErrors(cudaMemcpyToSymbol(d_dag_size, &dagNumItems, sizeof(u32)));
    if (memoryLightSize < lightSize) {
      if (h_light) {
        checkCudaErrors(cudaFree(h_light));
      }
      checkCudaErrors(cudaMalloc(&h_light, lightSize));
      checkCudaErrors(cudaMemcpyToSymbol(d_light, &h_light, sizeof(void *)));
      memoryLightSize = lightSize;
    }
    checkCudaErrors(
        cudaMemcpyToSymbol(d_light_size, &lightNumItems, sizeof(u32)));
    checkCudaErrors(cudaDeviceSynchronize());
  }

  void FreeCUDA() {
    if (h_dag) {
      checkCudaErrors(cudaFree(h_dag));
    }
    if (h_light) {
      checkCudaErrors(cudaFree(h_light));
    }
  }

public:
  void *h_light = 0;

  u32 lightNumItems;
  size_t lightSize;
  u32 dagNumItems;
  size_t dagSize;

private:
  void *h_dag = 0;

  size_t memoryDagSize = 0;
  size_t memoryLightSize = 0;
};

OctopusCUDAMiner::ThreadContext::ThreadContext(OctopusCUDAMiner *miner_,
                                               int device_id_, int context_id_)
    : miner(miner_), device_id(device_id_), context_id(context_id_),
      dagManager(new CUDADagManager()) {}

OctopusCUDAMiner::OctopusCUDAMiner(const OctopusCUDAMinerSettings &settings)
    : AbstractMiner(), settings(settings) {
  int device_count;
  checkCudaErrors(cudaGetDeviceCount(&device_count));

  int context_id = 0;

  for (int device_id : settings.device_ids) {
    if (device_id < device_count) {
      device_ids.push_back(device_id);
      threadContexts.emplace_back(this, device_id, context_id++);
    } else {
      std::cerr << "CUDA device_id = " << device_id << " does not exist."
                << std::endl;
    }
  }

  if (device_ids.empty()) {
    abort();
  }
}

OctopusCUDAMiner::~OctopusCUDAMiner() {}

void OctopusCUDAMiner::Start() {
  workerThreads = std::make_unique<boost::thread_group>();
  for (size_t i = 0; i < threadContexts.size(); ++i) {
    workerThreads->create_thread(
        boost::bind(&OctopusCUDAMiner::Work, this, &threadContexts[i]));
  }
}

void OctopusCUDAMiner::ThreadContext::InitCUDA() {
  checkCudaErrors(cudaSetDevice(device_id));
  checkCudaErrors(cudaMallocHost(&d_search_results, sizeof(SearchResults)));
}

void OctopusCUDAMiner::ThreadContext::InitPerEpoch(uint64_t blockHeight) {
  dagManager->reset(blockHeight);
  auto h_light = octopus_light_new(blockHeight);
  checkCudaErrors(cudaMemcpy(dagManager->h_light, h_light->cache,
                             dagManager->lightSize, cudaMemcpyHostToDevice));
  octopus_light_delete(h_light);

  const uint32_t work = dagManager->dagSize / 8;
  const uint32_t run = miner->settings.initGridSize * INIT_BLOCK_SIZE;

  uint32_t base;
  for (base = 0; base <= work - run; base += run) {
    InitDagItems<<<miner->settings.initGridSize, INIT_BLOCK_SIZE>>>(base);
  }
  if (base < work) {
    const uint32_t lastGrid =
        ((work - base) + INIT_BLOCK_SIZE - 1) / INIT_BLOCK_SIZE;
    InitDagItems<<<lastGrid, INIT_BLOCK_SIZE>>>(base);
  }
  checkCudaErrors(cudaDeviceSynchronize());
}

void OctopusCUDAMiner::ThreadContext::InitPerHeader(
    const octopus_h256_t headerHash, const octopus_h256_t boundary) {
  checkCudaErrors(
      cudaMemcpyToSymbol(d_header, headerHash.b, sizeof(headerHash)));
  {
    uint64_t buffer[4];
    for (int i = 0; i < 4; ++i) {
      const uint64_t b = reinterpret_cast<const uint64_t *>(boundary.b)[i];
      buffer[i] = ((b & 0xff00000000000000ULL) >> 56) |
                  ((b & 0x00ff000000000000ULL) >> 40) |
                  ((b & 0x0000ff0000000000ULL) >> 24) |
                  ((b & 0x000000ff00000000ULL) >> 8) |
                  ((b & 0x00000000ff000000ULL) << 8) |
                  ((b & 0x0000000000ff0000ULL) << 24) |
                  ((b & 0x000000000000ff00ULL) << 40) |
                  ((b & 0x00000000000000ffULL) << 56);
    }
    checkCudaErrors(cudaMemcpyToSymbol(d_boundary, buffer, sizeof(boundary)));
  }
  checkCudaErrors(cudaDeviceSynchronize());
}

void OctopusCUDAMiner::Work(ThreadContext *ctx) {
  ctx->InitCUDA();

  const uint32_t searchGridSize = settings.searchGridSize;
  const uint32_t batchSize = searchGridSize * SEARCH_BLOCK_SIZE;

  std::string jobId;
  uint64_t blockHeight = std::numeric_limits<uint64_t>::max();
  std::string headerHashString;
  octopus_h256_t headerHash;
  octopus_h256_t boundary;
  uint64_t nonce = startNonce + ctx->context_id * batchSize;

  while (is_running.load(std::memory_order_acquire)) {
    if (workJobId == MINER_NO_WORK) {
      boost::this_thread::sleep_for(boost::chrono::milliseconds(5000));
      continue;
    }
    if (0 != memcmp(headerHash.b, workHeaderHash.b, sizeof(headerHash))) {
      jobId = workJobId;
      headerHashString = workHeaderHashString;
      if (octopus_get_epoch(blockHeight) !=
          octopus_get_epoch(workBlockHeight)) {
        ctx->InitPerEpoch(workBlockHeight);
        blockHeight = workBlockHeight;
      }
      ctx->InitPerHeader(workHeaderHash, workBoundary);
      memcpy(headerHash.b, workHeaderHash.b, sizeof(headerHash));
      memcpy(boundary.b, workBoundary.b, sizeof(boundary));
      nonce = startNonce + ctx->context_id * batchSize;
    }

    volatile SearchResults &search_results =
        *reinterpret_cast<SearchResults *>(ctx->d_search_results);
    search_results.count = 0;
    Compute<<<settings.searchGridSize, SEARCH_BLOCK_SIZE>>>(
        nonce, reinterpret_cast<SearchResults *>(ctx->d_search_results));
    checkCudaErrors(cudaDeviceSynchronize());

    uint32_t found_count =
        std::min((uint32_t)search_results.count, MAX_SEARCH_RESULTS);
    for (uint32_t i = 0; i < found_count; i++) {
      uint64_t found_nonce = nonce + search_results.result[i].nonce_offset;
      std::vector<std::string> solutions;
      solutions.push_back(jobId);
      solutions.push_back("0x" + hex::to_hex_string(found_nonce));
      solutions.push_back(headerHashString);
      client->OnSolutionFound(solutions);
    }
    client->UpdateHashRate(batchSize);
    nonce += batchSize * device_ids.size();
  }

  checkCudaErrors(cudaDeviceSynchronize());
  ctx->dagManager->FreeCUDA();
  checkCudaErrors(cudaFreeHost(ctx->d_search_results));
}
