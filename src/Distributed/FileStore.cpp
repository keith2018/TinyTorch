/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "FileStore.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <thread>

#include "Utils/Logger.h"

namespace tinytorch::distributed {

class FileLock {
 public:
  explicit FileLock(const std::string& path) {
    fd_ = open(path.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd_ == -1) {
      LOGE("FileStore: Open failed for path: %s", path.c_str());
      return;
    }

    struct flock fl {};
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0;
    if (fcntl(fd_, F_SETLKW, &fl) == -1) {
      LOGE("FileStore: fcntl lock failed for path: %s", path.c_str());
      close(fd_);
      fd_ = -1;
    }
  }

  ~FileLock() {
    if (fd_ != -1) {
      struct flock fl {};
      fl.l_type = F_UNLCK;
      fl.l_whence = SEEK_SET;
      fl.l_start = 0;
      fl.l_len = 0;
      fcntl(fd_, F_SETLK, &fl);
      close(fd_);
    }
  }

  bool isValid() const { return fd_ != -1; }

 private:
  int fd_ = -1;
};

FileStore::FileStore(std::string path) : path_(std::move(path)) {
  int result = 0;
  result = mkdir(path_.c_str(), 0755);
  if (result != 0 && errno != EEXIST) {
    LOGE("FileStore: Error creating directory: %s, errno: %d", path_.c_str(), errno);
  }
}

std::string FileStore::getKeyPath(const std::string& key) const { return path_ + "/" + key; }

void FileStore::set(const std::string& key, const std::vector<uint8_t>& value) {
  const auto path = getKeyPath(key);
  FileLock lock(path);
  if (!lock.isValid()) {
    LOGE("FileStore: lock failed for path: %s", path.c_str());
    return;
  }

  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    LOGE("FileStore: fopen for writing failed for path: %s", path.c_str());
    return;
  }

  if (!value.empty()) {
    size_t written = fwrite(value.data(), 1, value.size(), f);
    if (written != value.size()) {
      LOGE("FileStore: fwrite failed for path: %s", path.c_str());
    }
  }
  fclose(f);
}

std::vector<uint8_t> FileStore::get(const std::string& key) {
  const auto path = getKeyPath(key);

  auto timeout = timeout_;
  auto start = std::chrono::steady_clock::now();
  bool hasTimeout = (timeout != kNoTimeout);

  while (!checkKey(key)) {
    if (hasTimeout) {
      auto now = std::chrono::steady_clock::now();
      if (now - start > timeout) {
        LOGE("FileStore: Get timeout after %lld ms for key: %s", timeout.count(), key.c_str());
        return {};
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  FileLock lock(path);
  if (!lock.isValid()) {
    LOGE("FileStore: lock failed for path: %s", path.c_str());
    return {};
  }

  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    LOGE("FileStore: fopen for reading failed for path: %s", path.c_str());
    return {};
  }

  using FileCloserFn = int (*)(FILE*);
  std::unique_ptr<FILE, FileCloserFn> fileGuard(f, &fclose);

  if (fseek(f, 0, SEEK_END) != 0) {
    LOGE("FileStore: fseek failed for path: %s", path.c_str());
    return {};
  }

  auto fileSize = ftell(f);
  if (fileSize < 0) {
    LOGE("FileStore: ftell failed for path: %s", path.c_str());
    return {};
  }

  if (fseek(f, 0, SEEK_SET) != 0) {
    LOGE("FileStore: fseek to beginning failed for path: %s", path.c_str());
    return {};
  }

  auto size = static_cast<size_t>(fileSize);
  std::vector<uint8_t> value(size);

  if (size > 0) {
    auto read = fread(value.data(), 1, size, f);
    if (read != size) {
      LOGE("FileStore: fread failed for path: %s", path.c_str());
      return {};
    }
  }

  return value;
}

int64_t FileStore::add(const std::string& key, int64_t value) {
  const auto path = getKeyPath(key);
  FileLock lock(path);
  if (!lock.isValid()) {
    LOGE("FileStore: lock failed for path: %s", path.c_str());
    return 0;
  }

  int64_t currentValue = 0;
  FILE* fRead = fopen(path.c_str(), "rb");
  if (fRead) {
    char buf[64] = {0};
    if (fread(buf, 1, sizeof(buf) - 1, fRead) > 0) {
      currentValue = std::stoll(buf);
    }
    fclose(fRead);
  }

  currentValue += value;

  FILE* fWrite = fopen(path.c_str(), "wb");
  if (!fWrite) {
    LOGE("FileStore: fopen for writing failed for path: %s", path.c_str());
    return currentValue;
  }

  using FileCloserFn = int (*)(FILE*);
  std::unique_ptr<FILE, FileCloserFn> fileGuard(fWrite, &fclose);
  std::string s = std::to_string(currentValue);
  if (fwrite(s.c_str(), 1, s.length(), fWrite) != s.length()) {
    LOGE("FileStore: fwrite failed for path: %s", path.c_str());
  }

  return currentValue;
}

bool FileStore::deleteKey(const std::string& key) {
  const auto path = getKeyPath(key);
  FileLock lock(path);
  if (!lock.isValid()) {
    LOGE("FileStore: lock failed for path: %s", path.c_str());
    return false;
  }
  return unlink(path.c_str()) == 0;
}

bool FileStore::checkKey(const std::string& key) const {
  const auto path = getKeyPath(key);
  return access(path.c_str(), F_OK) == 0;
}

bool FileStore::check(const std::vector<std::string>& keys) {
  return std::all_of(keys.begin(), keys.end(), [this](const auto& key) { return checkKey(key); });
}

int64_t FileStore::getNumKeys() {
  int64_t count = 0;
  DIR* dir = opendir(path_.c_str());
  if (!dir) {
    LOGE("FileStore: opendir failed for path: %s", path_.c_str());
    return 0;
  }

  dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    const std::string name = entry->d_name;
    if (name != "." && name != "..") {
      count++;
    }
  }

  closedir(dir);
  return count;
}

void FileStore::wait(const std::vector<std::string>& keys) {
  auto start = std::chrono::steady_clock::now();
  while (!check(keys)) {
    auto now = std::chrono::steady_clock::now();
    if (now - start > timeout_) {
      LOGE("FileStore: Wait timeout after %lld ms", timeout_.count());
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace tinytorch::distributed