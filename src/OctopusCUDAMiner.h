#pragma once

#include <boost/thread.hpp>

#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

#include "AbstractMiner.h"
#include "octopus_params.h"

class StratumClient;

struct OctopusCUDAMinerSettings {
  std::vector<int> device_ids = {0};
  int initGridSize = 8192;
  int searchGridSize = 1024;
};

class CUDADagManager;

class OctopusCUDAMiner : public AbstractMiner {
protected:
  struct ThreadContext {
    OctopusCUDAMiner *miner;
    int device_id;
    int context_id;

    CUDADagManager *dagManager;

    void *d_search_results;

    ThreadContext(OctopusCUDAMiner *miner, int device_id, int context_id);

    void InitCUDA();
    void InitPerEpoch(uint64_t blockHeight);
    void InitPerHeader(const octopus_h256_t headerHash,
                       const octopus_h256_t bounadry);
  };

public:
  OctopusCUDAMiner(const OctopusCUDAMinerSettings &settings);

  ~OctopusCUDAMiner();

  void Start() override;

  void Join() override { workerThreads->join_all(); }

private:
  void Work(ThreadContext *ctx);

  std::unique_ptr<boost::thread_group> workerThreads;

  const OctopusCUDAMinerSettings settings;

protected:
  std::vector<int> device_ids;
  std::vector<ThreadContext> threadContexts;
};
