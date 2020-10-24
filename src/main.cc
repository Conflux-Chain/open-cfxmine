#include "OctopusCPUMiner.h"
#include "OctopusCUDAMiner.h"
#include "StratumClient.h"
#include "cxxopts.hpp"
#include <chrono>
#include <memory>
#include <thread>

static void WatchDogThread(std::shared_ptr<StratumClient> client,
                           std::string address, int port) {
  while (true) {
    boost::this_thread::sleep_for(boost::chrono::milliseconds(10000));
    if (!client->IsRunning()) {
      bool succ = client->StartSubscribe(address, port);
      if (!succ) {
        exit(1);
      }
    }
  }
}

int main(int argc, char *argv[]) {
  cxxopts::Options options("cfxmine 0.0.1",
                           "A simple standalone miner for Conflux.");
  options.add_options()(
      "a,addr", "IP Address of Conflux full node to connect to",
      cxxopts::value<std::string>()->default_value("127.0.0.1"))(
      "p,port", "Conflux Stratum port number to connect to",
      cxxopts::value<int>()->default_value("32525"))(
      "n,name", "Worker name passed to the Conflux stratum",
      cxxopts::value<std::string>()->default_value("cfxmine"))(
      "r,retry",
      "How many times the miners repetitively try to connect the stratum if it "
      "fails. 0 means infinite.",
      cxxopts::value<int>()->default_value("10"))(
      "t,threads", "How many CPU mining threads we run in parallel.",
      cxxopts::value<int>()->default_value("1"))("h,help", "Print this help.")(
      "g,gpu", "Enable GPU mining",
      cxxopts::value<bool>()->default_value("false"))(
      "d,device_ids", "Specify gpu device ids",
      cxxopts::value<std::vector<int>>()->default_value("0"));

  std::string address;
  int port;
  std::string agent_name;
  int retry;
  int nthreads;
  bool use_gpu;
  OctopusCUDAMinerSettings cuda_miner_settings;
  try {
    cxxopts::ParseResult parsed_args = options.parse(argc, argv);
    address = parsed_args["addr"].as<std::string>();
    port = parsed_args["port"].as<int>();
    agent_name = parsed_args["name"].as<std::string>();
    retry = parsed_args["retry"].as<int>();
    nthreads = parsed_args["threads"].as<int>();
    use_gpu = parsed_args["gpu"].as<bool>();
    cuda_miner_settings.device_ids =
        parsed_args["device_ids"].as<std::vector<int>>();
    if (parsed_args.count("help") != 0) {
      std::cerr << options.help();
      return 0;
    }
  } catch (std::exception &ex) {
    std::cerr << "Cannot parse the arguments.\n" << ex.what() << "\n";
    std::cerr << options.help();
    return 1;
  }

  std::shared_ptr<AbstractMiner> miner;
  if (use_gpu) {
    std::cerr << "Using GPU." << std::endl;
    miner = std::make_shared<OctopusCUDAMiner>(cuda_miner_settings);
  } else {
    miner = std::make_shared<OctopusCPUMiner>(nthreads);
  }

  std::cout << "Start the miner for " << address << ":" << port << "\n";
  std::cout << "Press q and enter to quit the miner at any time.\n";

  if (nthreads > 32) {
    nthreads = 32;
  } else if (nthreads < 1) {
    nthreads = 1;
  }

#ifndef OCTOPUS_DEBUG
  std::shared_ptr<StratumClient> client =
      std::make_shared<StratumClient>(agent_name, retry, miner);
  miner->AttachStratum(client);
  bool succ = client->StartSubscribe(address, port);
  if (!succ) {
    return 1;
  }
#endif
  miner->Start();
#ifndef OCTOPUS_DEBUG
  boost::thread watchdog(std::bind(&WatchDogThread, client, address, port));
  while (true) {
    char c = std::cin.get();
    if (c == 'q') {
      miner->Stop();
      miner->Join();
      return 0;
    }
  }
#else
  std::cerr << "DEBUG" << std::endl;
  std::vector<std::string> params{
      "0xc34b3f2b5505303f65b5d666026af2e555465edacc47d0a0b4b98a05b7f95d63", "1",
      "0xc34b3f2b5505303f65b5d666026af2e555465edacc47d0a0b4b98a05b7f95d63",
      "0x10c6f7a0b5ed8d36b4c7f34938583621fafc8b0079a2834d26fa3fcc9ea9"};
  miner->NotifyWork(params);
  miner->Join();
#endif
}
