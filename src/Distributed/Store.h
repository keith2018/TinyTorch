/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace tinytorch::distributed {

// Ref: https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/Store.hpp
class Store {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout = std::chrono::seconds(300);
  static constexpr std::chrono::milliseconds kNoTimeout = std::chrono::milliseconds::zero();

  Store() : timeout_(kDefaultTimeout) {}

  explicit Store(const std::chrono::milliseconds& timeout) : timeout_(timeout) {}

  Store(const Store&) = default;
  Store(Store&&) noexcept = default;

  virtual ~Store() = default;

  void set(const std::string& key, const std::string& value);
  virtual void set(const std::string& key, const std::vector<uint8_t>& value) = 0;
  virtual std::vector<uint8_t> get(const std::string& key) = 0;
  std::string getString(const std::string& key);
  virtual int64_t add(const std::string& key, int64_t value) = 0;
  virtual bool deleteKey(const std::string& key) = 0;
  virtual bool check(const std::vector<std::string>& keys) = 0;
  virtual int64_t getNumKeys() = 0;
  virtual void wait(const std::vector<std::string>& keys) = 0;

 protected:
  std::chrono::milliseconds timeout_;
};

}  // namespace tinytorch::distributed
