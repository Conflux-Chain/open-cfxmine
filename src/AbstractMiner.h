#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "hex.h"
#include "octopus_params.h"
#include "octopus_structs.h"

class StratumClient;

static const std::string MINER_NO_WORK = "__NO_WORK__";

class AbstractMiner {
public:
  AbstractMiner() : is_running(true), workJobId{MINER_NO_WORK} {
    std::random_device rd;
    startNonce = std::uniform_int_distribution<uint64_t>(
                     0, static_cast<uint64_t>(1)
                            << (64 - OCTOPUS_NONCE_SEGMENT_BITS))(rd)
                 << OCTOPUS_NONCE_SEGMENT_BITS;
  }

  virtual void Start() = 0;

  void Stop() { is_running.store(false, std::memory_order_release); }

  virtual void Join() = 0;

  void NotifyWork(const std::vector<std::string> &params) {
    using namespace hex;

    if (params.size() >= 4) {
      workJobId = params[0];
      workBlockHeight = std::stoull(params[1]);
      workHeaderHashString = params[2];
      workHeaderHash =
          byte_vector_to_h256(hex_to_byte_vector(workHeaderHashString, 32));
      workBoundary = byte_vector_to_h256(hex_to_byte_vector(params[3], 32));
    }
  }

  void AttachStratum(std::shared_ptr<StratumClient> client) {
    this->client = client;
  }

protected:
  std::atomic_bool is_running;

  std::string workJobId;
  uint64_t workBlockHeight;
  std::string workHeaderHashString;
  octopus_h256_t workHeaderHash;
  octopus_h256_t workBoundary;

  std::shared_ptr<StratumClient> client;

  uint64_t startNonce;
};
