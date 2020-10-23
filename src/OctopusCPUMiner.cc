#include "OctopusCPUMiner.h"
#include "StratumClient.h"
#include "hex.h"
#include "light.h"
#include "octopus_params.h"
#include "octopus_structs.h"

void OctopusCPUMiner::Start() {
  workerThreads = std::make_unique<boost::thread_group>();
  for (uint32_t i = 0; i < numThreads; ++i) {
    workerThreads->create_thread(boost::bind(&OctopusCPUMiner::Work, this));
  }
}

void OctopusCPUMiner::Work() {
  std::string jobId;
  uint64_t blockHeight = std::numeric_limits<uint64_t>::max();
  std::string headerHashString;
  octopus_h256_t headerHash;
  octopus_h256_t boundary;
  octopus_light_t light = nullptr;
  uint64_t nonce = startNonce;

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
        if (light) {
          octopus_light_delete(light);
        }
      }
      blockHeight = workBlockHeight;
      if (!light) {
        light = octopus_light_new(blockHeight);
      }
      memcpy(headerHash.b, workHeaderHash.b, sizeof(headerHash));
      memcpy(boundary.b, workBoundary.b, sizeof(boundary));
      nonce = startNonce;
    }

#ifndef OCTOPUS_DEBUG
    octopus_return_value_t ret =
        octopus_light_compute(light, headerHash, nonce);

    if (ret.success) {
      if (octopus_check_difficulty(&ret.result, &boundary)) {
        std::vector<std::string> solutions;
        solutions.push_back(jobId);
        solutions.push_back("0x" + hex::to_hex_string(nonce));
        solutions.push_back(headerHashString);
        client->OnSolutionFound(solutions);
      }
    }

    ++nonce;
    client->UpdateHashRate(1);
#else
    octopus_light_compute(light, headerHash, nonce);
    break;
#endif
  }
}
