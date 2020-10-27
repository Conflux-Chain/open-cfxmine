#include "StratumClient.h"
#include "AbstractMiner.h"
#include <iostream>
#include <json/json.h>
#include <sstream>

#ifdef WIN32
#define UNUSED
#else
#define UNUSED __attribute__((unused))
#endif

using namespace boost::asio::ip;

namespace {

Json::Value ReadJsonFromSocket(tcp::socket &client_socket) {
  boost::asio::streambuf sbuf;
  size_t bytes_read = boost::asio::read_until(client_socket, sbuf, "\n");
  boost::asio::streambuf::const_buffers_type cbuf = sbuf.data();
  std::string data(buffers_begin(cbuf), buffers_begin(cbuf) + bytes_read);

  std::unique_ptr<Json::CharReader> reader(
      Json::CharReaderBuilder().newCharReader());
  Json::Value root;
  std::string errors;
  bool succ =
      reader->parse(data.c_str(), data.c_str() + data.size(), &root, &errors);
  if (!succ) {
    std::cout << "Unable to parse " << data << ".\nError: " << errors << "\n";
  }
  return root;
}

void ProcessErrorMessage(UNUSED tcp::socket &client_socket,
                         const Json::Value &msg) {
  std::cout << "Got error from Server: " << msg << "\n";
}

void ProcessUnknownRPCMessage(UNUSED tcp::socket &client_socket,
                              const Json::Value &msg) {
  std::cout << "Got unknown RPC request from Server: " << msg << "\n";
}

void ProcessResponseMessage(UNUSED tcp::socket &client_socket,
                            const Json::Value &msg) {
  try {
    Json::Value res = msg["result"];
    if (res.isBool()) {
      if (res.asBool()) {
        std::cout << "Accepted solution (" << msg["id"].asInt() << ").\n";
      } else {
        std::cout << "Rejected solution (" << msg["id"].asInt()
                  << ", no reason specified).\n";
      }
    } else if (res.isArray()) {
      if (res[0].asBool()) {
        std::cout << "Accepted solution (" << msg["id"].asInt() << ").\n";
      } else {
        std::cout << "Rejected solution (" << msg["id"].asInt() << ", " << res
                  << ").\n";
      }
    } else {
      std::cout << "Got unknown response from Server: " << msg << "\n";
    }
  } catch (std::exception &ex) {
    std::cout << "Got unknown response from Server: " << msg << "\n";
  }
}

std::string BuildJsonString(const Json::Value &value) {
  Json::StreamWriterBuilder builder;
  builder["commentStyle"] = "None";
  builder["indentation"] = "";
  std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
  std::ostringstream sout;
  writer->write(value, &sout);
  return sout.str() + "\n";
}

} // namespace

void StratumClient::HandleDisconnect() {
  std::cout << "Disconnected!\n";
  this->ioWork.reset();
}

void StratumClient::DummyWriteHandler(const boost::system::error_code &ec,
                                      UNUSED std::size_t bytes_transferred) {
  if (boost::asio::error::eof == ec ||
      boost::asio::error::connection_reset == ec) {
    this->HandleDisconnect();
  }
}

void StratumClient::AsyncReadUntilHandler(const boost::system::error_code &ec,
                                          std::size_t bytes_transferred) {
  if (boost::asio::error::eof == ec ||
      boost::asio::error::connection_reset == ec) {
    this->HandleDisconnect();
    return;
  }
  boost::asio::streambuf::const_buffers_type cbuf = this->stream_buf.data();
  std::string data(buffers_begin(cbuf),
                   buffers_begin(cbuf) + bytes_transferred);
  this->stream_buf.consume(bytes_transferred);
  std::unique_ptr<Json::CharReader> reader(
      Json::CharReaderBuilder().newCharReader());
  Json::Value root;
  std::string errors;
  bool succ =
      reader->parse(data.c_str(), data.c_str() + data.size(), &root, &errors);

  if (!succ) {
    std::cout << "Unable to parse " << data << ". Reported error: " << errors
              << "\n";
  } else if (root.isMember("error")) {
    ProcessErrorMessage(*this->client_socket, root);
  } else if (root.isMember("result")) {
    ProcessResponseMessage(*this->client_socket, root);
  } else if (root.isMember("method")) {
    if (root["method"] == "mining.notify") {
      Json::Value params = root["params"];
      try {
        std::vector<std::string> params_vec;
        if (params.size() > 0) {
          std::cout << "Get a new job to work on (" << params[0];
          for (int i = 1; i < static_cast<int>(params.size()); ++i) {
            std::cout << "," << params[i];
          }
          std::cout << ")\n";
        }
        for (int i = 0; i < static_cast<int>(params.size()); i++)
          params_vec.push_back(params[i].asString());
        miner->NotifyWork(params_vec);
      } catch (std::exception &ex) {
        ProcessUnknownRPCMessage(*this->client_socket, root);
      }
    } else {
      ProcessUnknownRPCMessage(*this->client_socket, root);
    }
  }

  boost::asio::async_read_until(*this->client_socket, this->stream_buf, "\n",
                                std::bind(&StratumClient::AsyncReadUntilHandler,
                                          this, std::placeholders::_1,
                                          std::placeholders::_2));
}

void StratumClient::SubmitJobAsync(const std::vector<std::string> solutions) {
  Json::Value submitJson;
  submitJson["jsonrpc"] = "2.0";
  submitJson["method"] = "mining.submit";
  submitJson["id"] = (Json::UInt64)this->jsonId++;
  Json::Value params;
  params.append(this->name);
  for (size_t i = 0; i < solutions.size(); i++) {
    params.append(solutions[i]);
  }
  submitJson["params"] = params;
  std::cout << "Found a solution: {";
  for (size_t i = 0; i < solutions.size(); i++) {
    if (i != 0)
      std::cout << ",";
    std::cout << solutions[i];
  }
  std::cout << "}\n";
  total_accepted_count++;
  // std::cout << "Submit: " << message;
  boost::asio::async_write(
      *this->client_socket, boost::asio::buffer(BuildJsonString(submitJson)),
      std::bind(&StratumClient::DummyWriteHandler, this, std::placeholders::_1,
                std::placeholders::_2));
}

void StratumClient::UpdateHashRateAsync(size_t nonce_count) {
  auto now = std::chrono::high_resolution_clock::now();
  if (total_nonce_count == 0) {
    hashrate_start_time = hashrate_last_report_time = now;
    total_nonce_count += nonce_count;
  } else {
    total_nonce_count += nonce_count;
    if ((now - hashrate_last_report_time) > std::chrono::seconds(6)) {
      std::cout << "Hashrate: "
                << 1.0 * total_nonce_count /
                       std::chrono::duration_cast<std::chrono::seconds>(
                           now - hashrate_start_time)
                           .count()
                << "/s" << std::endl;
      hashrate_last_report_time = now;
    }
  }
}

void StratumClient::WorkerThread() {
  boost::asio::async_read_until(*this->client_socket, this->stream_buf, "\n",
                                std::bind(&StratumClient::AsyncReadUntilHandler,
                                          this, std::placeholders::_1,
                                          std::placeholders::_2));

  ioService->run();
  this->running = StratumNotConnected;
}

bool StratumClient::StartSubscribe(const std::string &address, const int port) {
  this->running = StratumConnecting;
  for (int i = 0; i < kRetryCnt || kRetryCnt == 0; i++) {
    std::cout << "Trying to connect to the server " << address << ":" << port
              << "\n";
    try {
      this->ioService = std::make_unique<boost::asio::io_service>();
      this->client_socket = std::make_unique<tcp::socket>(*this->ioService);
      this->client_socket->connect(
          tcp::endpoint(address::from_string(address), port));
      Json::Value subJson;
      subJson["jsonrpc"] = "2.0";
      subJson["method"] = "mining.subscribe";
      subJson["id"] = (Json::UInt64)1;
      Json::Value params;
      params.append(this->name);
      params.append("");
      subJson["params"] = params;
      boost::asio::write(*this->client_socket,
                         boost::asio::buffer(BuildJsonString(subJson)));
      // Parse the response to make sure it returns OK
      {
        Json::Value root = ReadJsonFromSocket(*this->client_socket);
        if (!root["result"].asBool()) {
          return false;
        }
      }
      this->ioWork =
          std::make_unique<boost::asio::io_service::work>(*this->ioService);
      this->jsonId.store(2, std::memory_order_relaxed);

      this->workerThread = std::make_unique<boost::thread>(
          std::bind(&StratumClient::WorkerThread, this));
      std::cout << "Connected to the server " << address << ":" << port
                << "!\n";
      this->running = StratumConnected;
      return true;
    } catch (std::exception &ex) {
      std::cout << "Unable to connect and subscribe to the server " << address
                << ":" << port << "!\n";
      boost::this_thread::sleep_for(boost::chrono::milliseconds(3000));
    }
  }
  return false;
}

void StratumClient::OnSolutionFound(const std::vector<std::string> &solutions) {
  if (this->running == StratumConnected) {
    this->ioService->post(
        std::bind(&StratumClient::SubmitJobAsync, this, solutions));
  } else {
    std::cout << "Stratum is not connected! One solution discarded!\n";
  }
}

void StratumClient::UpdateHashRate(size_t nonce_count) {
  this->ioService->post(
      std::bind(&StratumClient::UpdateHashRateAsync, this, nonce_count));
}

void StratumClient::Stop() {
  this->ioWork.reset();
  this->workerThread->join();
}

bool StratumClient::IsRunning() { return this->running; }
