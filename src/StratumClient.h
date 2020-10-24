#pragma once
#include <atomic>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <chrono>
#include <memory>
#include <string>

class AbstractMiner;

class StratumClient {
public:
  enum ConnectionStatus {
    StratumNotConnected,
    StratumConnecting,
    StratumConnected,
  };

  explicit StratumClient(const std::string &name, int retry,
                         std::shared_ptr<AbstractMiner> miner)
      : kRetryCnt(retry), name(name), miner(std::move(miner)), stream_buf(),
        running(StratumNotConnected) {}

  ~StratumClient() = default;

  bool StartSubscribe(const std::string &address, const int port);

  void OnSolutionFound(const std::vector<std::string> &solution);

  void UpdateHashRate(size_t nonce_count);

  bool IsRunning();

  void Stop();

private:
  const int kRetryCnt;

  std::string name;
  std::shared_ptr<AbstractMiner> miner;
  std::unique_ptr<boost::asio::io_service> ioService;
  boost::asio::streambuf stream_buf;
  std::unique_ptr<boost::asio::io_service::work> ioWork;
  ConnectionStatus running;
  std::atomic<uint64_t> jsonId;
  std::unique_ptr<boost::thread> workerThread;
  std::unique_ptr<boost::asio::ip::tcp::socket> client_socket;

  size_t total_nonce_count = 0;
  size_t total_accepted_count = 0;
  std::chrono::high_resolution_clock::time_point hashrate_last_report_time;
  std::chrono::high_resolution_clock::time_point hashrate_start_time;

  void HandleDisconnect();

  void DummyWriteHandler(const boost::system::error_code &ec,
                         std::size_t bytes_transferred);

  void AsyncReadUntilHandler(const boost::system::error_code &ec,
                             std::size_t bytes_transferred);

  void SubmitJobAsync(const std::vector<std::string> solutions);

  void UpdateHashRateAsync(size_t nonce_count);

  void WorkerThread();
};