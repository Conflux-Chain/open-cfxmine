#pragma once

#include <boost/thread.hpp>
#include <cstdint>
#include <iostream>

#include "AbstractMiner.h"

class StratumClient;

class OctopusCPUMiner : public AbstractMiner {
public:
  OctopusCPUMiner(uint32_t numThreads)
      : AbstractMiner(), numThreads{numThreads} {}

  ~OctopusCPUMiner() = default;

  void Start() override;

  void Join() override { workerThreads->join_all(); }

private:
  void Work();

  uint32_t numThreads;

  std::unique_ptr<boost::thread_group> workerThreads;
};
