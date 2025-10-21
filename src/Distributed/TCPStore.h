/*
 * TinyTorch
 * @author  : keith@robot9.me
 *
 */

#pragma once

#include <memory>

#include "Store.h"

namespace tinytorch::distributed {

class TCPStore : public Store {
 public:
  static constexpr std::uint16_t kDefaultPort = 29500;

  explicit TCPStore(const std::string& host, uint16_t port = kDefaultPort, bool isServer = false,
                    bool waitWorkers = true, int numWorkers = -1,
                    const std::chrono::milliseconds& timeout = kDefaultTimeout);

  ~TCPStore() override;

  TCPStore(const TCPStore&) = delete;
  TCPStore& operator=(const TCPStore&) = delete;

  void set(const std::string& key, const std::vector<uint8_t>& value) override;
  std::vector<uint8_t> get(const std::string& key) override;
  int64_t add(const std::string& key, int64_t value) override;
  bool deleteKey(const std::string& key) override;
  bool check(const std::vector<std::string>& keys) override;
  int64_t getNumKeys() override;
  void wait(const std::vector<std::string>& keys) override;

  const std::string& getHost() const noexcept { return host_; }
  std::uint16_t getPort() const noexcept { return port_; }

 private:
  class ServerImpl;
  class ClientImpl;

  std::unique_ptr<ServerImpl> server_;
  std::unique_ptr<ClientImpl> client_;

  std::string host_;
  uint16_t port_;
  bool isServer_;
};

}  // namespace tinytorch::distributed