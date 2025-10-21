/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include "Store.h"

namespace tinytorch::distributed {

class FileStore : public Store {
 public:
  explicit FileStore(std::string path);
  ~FileStore() override = default;

  FileStore(const FileStore&) = delete;
  FileStore& operator=(const FileStore&) = delete;

  void set(const std::string& key, const std::vector<uint8_t>& value) override;
  std::vector<uint8_t> get(const std::string& key) override;
  int64_t add(const std::string& key, int64_t value) override;
  bool deleteKey(const std::string& key) override;
  bool check(const std::vector<std::string>& keys) override;
  int64_t getNumKeys() override;
  void wait(const std::vector<std::string>& keys) override;

 private:
  std::string getKeyPath(const std::string& key) const;
  bool checkKey(const std::string& key) const;

  std::string path_;
};

}  // namespace tinytorch::distributed
