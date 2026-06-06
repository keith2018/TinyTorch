/*
 * TinyTorch
 * @author  : keith@robot9.me
 *
 */

#include "TCPStore.h"

#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Utils/Logger.h"
#include "Utils/Macros.h"

#ifdef __APPLE__
#include <sys/types.h>
#define MSG_NOSIGNAL 0
#endif

typedef int socket_t;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define closesocket close

namespace tinytorch::distributed {

namespace detail {

enum class TCPStoreCommandType : uint8_t {
  CMD_SET,
  CMD_GET,
  CMD_ADD,
  CMD_CHECK,
  CMD_WAIT,
  CMD_DELETE_KEY,
  CMD_NUM_KEYS,
  CMD_WORKER_REGISTER,
  CMD_WORKER_UNREGISTER
};

static void setupSignalHandlers() { signal(SIGPIPE, SIG_IGN); }

static bool sendBytes(socket_t socket, const void* data, size_t size) {
  const char* ptr = static_cast<const char*>(data);
  size_t remaining = size;

  while (remaining > 0) {
    ssize_t sent = send(socket, ptr, remaining, MSG_NOSIGNAL);
    if (sent < 0) {
      int error = errno;
      if (error == EINTR) {
        continue;
      }
      LOGE("TCPStore: Send error: %d", error);
      return false;
    } else if (sent == 0) {
      return false;
    }
    ptr += sent;
    remaining -= sent;
  }
  return true;
}

static bool recvBytes(socket_t socket, void* data, size_t size) {
  char* ptr = static_cast<char*>(data);
  size_t remaining = size;

  while (remaining > 0) {
    ssize_t received = recv(socket, ptr, remaining, 0);
    if (received < 0) {
      int error = errno;
      if (error == EINTR) {
        continue;
      }
      if (error != EAGAIN && error != EWOULDBLOCK && error != ETIMEDOUT) {
        LOGE("TCPStore: Recv error: %d", error);
      }
      return false;
    } else if (received == 0) {
      return false;
    }
    ptr += received;
    remaining -= received;
  }
  return true;
}

static bool sendString(socket_t sock, const std::string& s) {
  uint64_t size = s.size();
  return sendBytes(sock, &size, sizeof(size)) && (size == 0 || sendBytes(sock, s.data(), size));
}

static bool recvString(socket_t sock, std::string& s) {
  uint64_t size;
  if (!recvBytes(sock, &size, sizeof(size))) return false;
  s.resize(size);
  return size == 0 || recvBytes(sock, &s[0], size);
}

static bool sendBlob(socket_t sock, const std::vector<uint8_t>& v) {
  uint64_t size = v.size();
  return sendBytes(sock, &size, sizeof(size)) && (size == 0 || sendBytes(sock, v.data(), size));
}

static bool recvBlob(socket_t sock, std::vector<uint8_t>& v) {
  uint64_t size;
  if (!recvBytes(sock, &size, sizeof(size))) return false;
  v.resize(size);
  return size == 0 || recvBytes(sock, v.data(), size);
}

static bool configureSocket(socket_t socket, bool isServer) {
  int optval = 1;

  if (isServer) {
    if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0) {
      LOGE("TCPStore: Failed to set SO_REUSEADDR");
      return false;
    }
  }

  if (setsockopt(socket, SOL_SOCKET, SO_KEEPALIVE, &optval, sizeof(optval)) < 0) {
    LOGE("TCPStore: Failed to set SO_KEEPALIVE");
    return false;
  }

#ifdef __APPLE__
  if (setsockopt(socket, SOL_SOCKET, SO_NOSIGPIPE, &optval, sizeof(optval)) < 0) {
    LOGE("TCPStore: Failed to set SO_NOSIGPIPE");
    return false;
  }
#endif

  return true;
}

static bool setSocketTimeout(socket_t socket, int timeoutSec) {
  timeval timeout{};
  timeout.tv_sec = timeoutSec;
  timeout.tv_usec = 0;
  if (setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
    return false;
  }
  if (setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0) {
    return false;
  }
  return true;
}

}  // namespace detail

class TCPStore::ClientImpl {
 public:
  ClientImpl(std::string host, uint16_t port, const std::chrono::milliseconds& timeout);
  ~ClientImpl();

  ClientImpl(const ClientImpl&) = delete;
  ClientImpl& operator=(const ClientImpl&) = delete;

  bool connect();
  bool set(const std::string& key, const std::vector<uint8_t>& value) const;
  bool get(const std::string& key, std::vector<uint8_t>& value) const;
  bool add(const std::string& key, int64_t addValue, int64_t& newValue) const;
  bool deleteKey(const std::string& key) const;
  bool check(const std::vector<std::string>& keys) const;
  bool getNumKeys(int64_t& numKeys) const;
  bool wait(const std::vector<std::string>& keys) const;
  bool registerWorker() const;
  bool unregisterWorker() const;

  const std::string& getHost() const noexcept { return host_; }
  std::uint16_t getPort() const noexcept { return port_; }

 private:
  bool sendCommand(detail::TCPStoreCommandType cmd) const;
  bool sendKeys(const std::vector<std::string>& keys) const;

  std::string host_;
  uint16_t port_;
  socket_t clientSock_;
  int maxRetries_ = 10;
  int retryDelayMs_ = 1000;
  mutable std::atomic<bool> isRegistered_{false};
};

class TCPStore::ServerImpl {
 public:
  ServerImpl(uint16_t port, int numWorkers, const std::chrono::milliseconds& timeout);
  ~ServerImpl();

  bool start();
  bool waitForWorkers();
  int getRegisteredWorkers() const;

 private:
  void serverLoop();
  void handleClientInThread(socket_t socket);
  bool handleClient(socket_t socket);
  bool handleSetCommand(socket_t socket);
  bool handleGetCommand(socket_t socket);
  bool handleAddCommand(socket_t socket);
  bool handleCheckCommand(socket_t socket);
  bool handleWaitCommand(socket_t socket);
  bool handleDeleteCommand(socket_t socket);
  bool handleNumKeysCommand(socket_t socket);
  bool handleWorkerRegister(socket_t socket);
  bool handleWorkerUnregister(socket_t socket);

  void notifyKeyWaiters(const std::string& key);
  bool waitForCondition(std::unique_lock<std::mutex>& lock, std::shared_ptr<std::condition_variable>& cv,
                        const std::function<bool()>& pred);

  uint16_t port_;
  int numWorkers_;
  std::chrono::milliseconds timeout_;
  std::atomic<bool> shutdownServer_;

  std::mutex mutex_;
  std::unordered_map<std::string, std::vector<uint8_t>> keyValueStore_;

  std::condition_variable workerCV_;
  std::atomic<int> registeredWorkers_;

  std::unordered_map<std::string, std::vector<std::shared_ptr<std::condition_variable>>> keyCVs_;

  socket_t serverSock_;
  std::thread serverThread_;

  std::mutex clientThreadsMutex_;
  std::vector<std::thread> clientThreads_;
};

// ============================================================
// TCPStore
// ============================================================

TCPStore::TCPStore(const std::string& host, uint16_t port, bool isServer, bool waitWorkers, int numWorkers,
                   const std::chrono::milliseconds& timeout)
    : Store(timeout), host_(host), port_(port), isServer_(isServer) {
  detail::setupSignalHandlers();

  if (isServer_) {
    server_ = std::make_unique<ServerImpl>(port, numWorkers, timeout);
    if (!server_->start()) {
      LOGE("TCPStore: Failed to start server");
      return;
    }
  }

  client_ = std::make_unique<ClientImpl>(host, port, timeout);
  if (!client_->connect()) {
    LOGE("TCPStore: Failed to connect client");
    return;
  }
  if (!client_->registerWorker()) {
    LOGE("TCPStore: Failed to register worker");
    return;
  }

  if (isServer_ && waitWorkers && numWorkers > 0) {
    if (!server_->waitForWorkers()) {
      LOGE("TCPStore: Timeout waiting for workers (expected: %d, registered: %d)", numWorkers,
           server_->getRegisteredWorkers());
      return;
    }
  }
}

TCPStore::~TCPStore() {
  if (client_) {
    client_->unregisterWorker();
  }
  client_.reset();
  server_.reset();
}

void TCPStore::set(const std::string& key, const std::vector<uint8_t>& value) {
  if (!client_->set(key, value)) {
    LOGE("TCPStore: set failed for key: %s", key.c_str());
  }
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
  std::vector<uint8_t> value;
  if (!client_->get(key, value)) {
    LOGE("TCPStore: get failed for key: %s", key.c_str());
  }
  return value;
}

int64_t TCPStore::add(const std::string& key, int64_t value) {
  int64_t newValue = -1;
  if (!client_->add(key, value, newValue)) {
    LOGE("TCPStore: add failed for key: %s", key.c_str());
  }
  return newValue;
}

bool TCPStore::deleteKey(const std::string& key) { return client_->deleteKey(key); }

bool TCPStore::check(const std::vector<std::string>& keys) { return client_->check(keys); }

int64_t TCPStore::getNumKeys() {
  int64_t numKeys = -1;
  if (!client_->getNumKeys(numKeys)) {
    LOGE("TCPStore: getNumKeys failed");
  }
  return numKeys;
}

void TCPStore::wait(const std::vector<std::string>& keys) {
  if (!client_->wait(keys)) {
    LOGE("TCPStore: wait failed or timed out");
  }
}

// ============================================================
// ClientImpl
// ============================================================

TCPStore::ClientImpl::ClientImpl(std::string host, uint16_t port, const std::chrono::milliseconds& timeout)
    : host_(std::move(host)), port_(port), clientSock_(INVALID_SOCKET) {}

TCPStore::ClientImpl::~ClientImpl() {
  if (clientSock_ != INVALID_SOCKET) {
    closesocket(clientSock_);
    clientSock_ = INVALID_SOCKET;
  }
}

bool TCPStore::ClientImpl::connect() {
  clientSock_ = socket(AF_INET, SOCK_STREAM, 0);
  if (clientSock_ == INVALID_SOCKET) {
    LOGE("TCPStore: Failed to create client socket");
    return false;
  }

  if (!detail::configureSocket(clientSock_, false)) {
    closesocket(clientSock_);
    clientSock_ = INVALID_SOCKET;
    return false;
  }

  detail::setSocketTimeout(clientSock_, 30);

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port_);

  addrinfo hints{}, *result;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  if (getaddrinfo(host_.c_str(), nullptr, &hints, &result) != 0) {
    LOGE("TCPStore: Failed to resolve hostname %s", host_.c_str());
    closesocket(clientSock_);
    clientSock_ = INVALID_SOCKET;
    return false;
  }

  std::memcpy(&addr.sin_addr, &reinterpret_cast<sockaddr_in*>(result->ai_addr)->sin_addr, sizeof(in_addr));
  freeaddrinfo(result);

  bool connected = false;
  for (int i = 0; i < maxRetries_; i++) {
    if (::connect(clientSock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      connected = true;
      break;
    }
    if (i < maxRetries_ - 1) {
      LOGE("TCPStore: Connection attempt %d failed, retrying...", i + 1);
      std::this_thread::sleep_for(std::chrono::milliseconds(retryDelayMs_));
    }
  }

  if (!connected) {
    LOGE("TCPStore: Failed to connect after %d attempts", maxRetries_);
    closesocket(clientSock_);
    clientSock_ = INVALID_SOCKET;
    return false;
  }

  return true;
}

bool TCPStore::ClientImpl::sendCommand(detail::TCPStoreCommandType cmd) const {
  return detail::sendBytes(clientSock_, &cmd, sizeof(cmd));
}

bool TCPStore::ClientImpl::sendKeys(const std::vector<std::string>& keys) const {
  uint64_t numKeys = keys.size();
  if (!detail::sendBytes(clientSock_, &numKeys, sizeof(numKeys))) return false;
  for (const auto& key : keys) {
    if (!detail::sendString(clientSock_, key)) return false;
  }
  return true;
}

bool TCPStore::ClientImpl::set(const std::string& key, const std::vector<uint8_t>& value) const {
  if (!sendCommand(detail::TCPStoreCommandType::CMD_SET)) return false;
  if (!detail::sendString(clientSock_, key)) return false;
  if (!detail::sendBlob(clientSock_, value)) return false;

  uint8_t confirmation;
  return detail::recvBytes(clientSock_, &confirmation, sizeof(confirmation)) && confirmation == 1;
}

bool TCPStore::ClientImpl::get(const std::string& key, std::vector<uint8_t>& value) const {
  if (!sendCommand(detail::TCPStoreCommandType::CMD_GET)) return false;
  if (!detail::sendString(clientSock_, key)) return false;
  return detail::recvBlob(clientSock_, value);
}

bool TCPStore::ClientImpl::add(const std::string& key, int64_t addValue, int64_t& newValue) const {
  if (!sendCommand(detail::TCPStoreCommandType::CMD_ADD)) return false;
  if (!detail::sendString(clientSock_, key)) return false;
  if (!detail::sendBytes(clientSock_, &addValue, sizeof(addValue))) return false;
  return detail::recvBytes(clientSock_, &newValue, sizeof(newValue));
}

bool TCPStore::ClientImpl::deleteKey(const std::string& key) const {
  if (!sendCommand(detail::TCPStoreCommandType::CMD_DELETE_KEY)) return false;
  if (!detail::sendString(clientSock_, key)) return false;

  uint8_t result;
  return detail::recvBytes(clientSock_, &result, sizeof(result)) && result == 1;
}

bool TCPStore::ClientImpl::check(const std::vector<std::string>& keys) const {
  if (!sendCommand(detail::TCPStoreCommandType::CMD_CHECK)) return false;
  if (!sendKeys(keys)) return false;

  uint8_t result;
  return detail::recvBytes(clientSock_, &result, sizeof(result)) && result == 1;
}

bool TCPStore::ClientImpl::getNumKeys(int64_t& numKeys) const {
  if (!sendCommand(detail::TCPStoreCommandType::CMD_NUM_KEYS)) return false;
  return detail::recvBytes(clientSock_, &numKeys, sizeof(numKeys));
}

bool TCPStore::ClientImpl::wait(const std::vector<std::string>& keys) const {
  if (!sendCommand(detail::TCPStoreCommandType::CMD_WAIT)) return false;
  if (!sendKeys(keys)) return false;

  uint8_t result;
  return detail::recvBytes(clientSock_, &result, sizeof(result)) && result == 1;
}

bool TCPStore::ClientImpl::registerWorker() const {
  if (!sendCommand(detail::TCPStoreCommandType::CMD_WORKER_REGISTER)) return false;

  uint8_t confirmation;
  if (!detail::recvBytes(clientSock_, &confirmation, sizeof(confirmation)) || confirmation != 1) {
    return false;
  }
  isRegistered_.store(true);
  return true;
}

bool TCPStore::ClientImpl::unregisterWorker() const {
  if (!isRegistered_.load()) return true;
  if (!sendCommand(detail::TCPStoreCommandType::CMD_WORKER_UNREGISTER)) return false;
  isRegistered_.store(false);
  return true;
}

// ============================================================
// ServerImpl
// ============================================================

TCPStore::ServerImpl::ServerImpl(uint16_t port, int numWorkers, const std::chrono::milliseconds& timeout)
    : port_(port),
      numWorkers_(numWorkers),
      timeout_(timeout),
      shutdownServer_(false),
      registeredWorkers_(0),
      serverSock_(INVALID_SOCKET) {}

TCPStore::ServerImpl::~ServerImpl() {
  shutdownServer_.store(true);

  if (serverSock_ != INVALID_SOCKET) {
    closesocket(serverSock_);
    serverSock_ = INVALID_SOCKET;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [key, cvs] : keyCVs_) {
      for (auto& cv : cvs) {
        cv->notify_all();
      }
    }
    keyCVs_.clear();
    workerCV_.notify_all();
  }

  if (serverThread_.joinable()) {
    serverThread_.join();
  }

  {
    std::lock_guard<std::mutex> lock(clientThreadsMutex_);
    for (auto& t : clientThreads_) {
      if (t.joinable()) {
        t.join();
      }
    }
    clientThreads_.clear();
  }
}

bool TCPStore::ServerImpl::start() {
  serverSock_ = socket(AF_INET, SOCK_STREAM, 0);
  if (serverSock_ == INVALID_SOCKET) {
    LOGE("TCPStore: Failed to create server socket");
    return false;
  }

  if (!detail::configureSocket(serverSock_, true)) {
    closesocket(serverSock_);
    serverSock_ = INVALID_SOCKET;
    return false;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port_);
  addr.sin_addr.s_addr = INADDR_ANY;

  if (bind(serverSock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    LOGE("TCPStore: Failed to bind server socket on port %d", port_);
    closesocket(serverSock_);
    serverSock_ = INVALID_SOCKET;
    return false;
  }

  if (listen(serverSock_, 128) != 0) {
    LOGE("TCPStore: Failed to listen on server socket");
    closesocket(serverSock_);
    serverSock_ = INVALID_SOCKET;
    return false;
  }

  serverThread_ = std::thread(&ServerImpl::serverLoop, this);
  return true;
}

bool TCPStore::ServerImpl::waitForWorkers() {
  if (numWorkers_ <= 0) return true;

  std::unique_lock<std::mutex> lock(mutex_);
  bool success = workerCV_.wait_for(
      lock, timeout_, [this] { return shutdownServer_.load() || registeredWorkers_.load() >= numWorkers_; });

  if (!success) {
    LOGE("TCPStore: Timeout waiting for workers. Expected: %d, Registered: %d", numWorkers_, registeredWorkers_.load());
  }
  return success && !shutdownServer_.load();
}

int TCPStore::ServerImpl::getRegisteredWorkers() const { return registeredWorkers_.load(); }

void TCPStore::ServerImpl::serverLoop() {
  detail::setSocketTimeout(serverSock_, 1);

  while (!shutdownServer_.load()) {
    sockaddr_in clientAddr{};
    socklen_t addrLen = sizeof(clientAddr);
    socket_t clientSocket = accept(serverSock_, reinterpret_cast<sockaddr*>(&clientAddr), &addrLen);

    if (clientSocket == INVALID_SOCKET) {
      continue;
    }

    std::lock_guard<std::mutex> lock(clientThreadsMutex_);
    clientThreads_.emplace_back(&ServerImpl::handleClientInThread, this, clientSocket);
  }
}

void TCPStore::ServerImpl::handleClientInThread(socket_t socket) {
  detail::configureSocket(socket, false);
  // SO_RCVTIMEO acts as a mid-message safety net only. Inter-message idleness
  // is governed by poll() below — a long quiet period (e.g. while a client is
  // loading a large model between registerWorker() and the first NCCL
  // collective) must NOT tear down the connection.
  detail::setSocketTimeout(socket, 30);

  while (!shutdownServer_.load()) {
    // wait for either readable data or shutdown, in 1s ticks.
    struct pollfd pfd {};
    pfd.fd = socket;
    pfd.events = POLLIN;
    int rc = poll(&pfd, 1, 1000);
    if (rc < 0) {
      if (errno == EINTR) continue;
      break;
    }
    if (rc == 0) {
      // idle; loop and re-check shutdownServer_.
      continue;
    }
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
      break;
    }
    if (!(pfd.revents & POLLIN)) {
      continue;
    }
    if (!handleClient(socket)) {
      break;
    }
  }

  closesocket(socket);
}

bool TCPStore::ServerImpl::handleClient(socket_t socket) {
  if (shutdownServer_.load()) return false;

  detail::TCPStoreCommandType cmd;
  if (!detail::recvBytes(socket, &cmd, sizeof(cmd))) {
    return false;
  }

  if (shutdownServer_.load()) return false;

  switch (cmd) {
    case detail::TCPStoreCommandType::CMD_SET:
      return handleSetCommand(socket);
    case detail::TCPStoreCommandType::CMD_GET:
      return handleGetCommand(socket);
    case detail::TCPStoreCommandType::CMD_ADD:
      return handleAddCommand(socket);
    case detail::TCPStoreCommandType::CMD_CHECK:
      return handleCheckCommand(socket);
    case detail::TCPStoreCommandType::CMD_WAIT:
      return handleWaitCommand(socket);
    case detail::TCPStoreCommandType::CMD_DELETE_KEY:
      return handleDeleteCommand(socket);
    case detail::TCPStoreCommandType::CMD_NUM_KEYS:
      return handleNumKeysCommand(socket);
    case detail::TCPStoreCommandType::CMD_WORKER_REGISTER:
      return handleWorkerRegister(socket);
    case detail::TCPStoreCommandType::CMD_WORKER_UNREGISTER:
      return handleWorkerUnregister(socket);
    default:
      LOGE("TCPStore: Unknown command: %d", static_cast<int>(cmd));
      return false;
  }
}

void TCPStore::ServerImpl::notifyKeyWaiters(const std::string& key) {
  auto it = keyCVs_.find(key);
  if (it != keyCVs_.end()) {
    for (auto& cv : it->second) {
      cv->notify_all();
    }
    keyCVs_.erase(it);
  }
}

bool TCPStore::ServerImpl::waitForCondition(std::unique_lock<std::mutex>& lock,
                                            std::shared_ptr<std::condition_variable>& cv,
                                            const std::function<bool()>& pred) {
  if (timeout_ != kNoTimeout) {
    return cv->wait_for(lock, timeout_, [&] { return shutdownServer_.load() || pred(); }) && !shutdownServer_.load();
  }
  cv->wait(lock, [&] { return shutdownServer_.load() || pred(); });
  return !shutdownServer_.load();
}

bool TCPStore::ServerImpl::handleSetCommand(socket_t socket) {
  std::string key;
  std::vector<uint8_t> value;
  if (!detail::recvString(socket, key) || !detail::recvBlob(socket, value)) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    keyValueStore_[key] = std::move(value);
    notifyKeyWaiters(key);
  }

  uint8_t confirmation = 1;
  return detail::sendBytes(socket, &confirmation, sizeof(confirmation));
}

bool TCPStore::ServerImpl::handleGetCommand(socket_t socket) {
  std::string key;
  if (!detail::recvString(socket, key)) return false;

  std::vector<uint8_t> value;
  std::shared_ptr<std::condition_variable> cv;

  {
    std::unique_lock<std::mutex> lock(mutex_);

    auto it = keyValueStore_.find(key);
    if (it != keyValueStore_.end()) {
      value = it->second;
      return detail::sendBlob(socket, value);
    }

    cv = std::make_shared<std::condition_variable>();
    keyCVs_[key].push_back(cv);

    bool success = waitForCondition(lock, cv, [&] { return keyValueStore_.find(key) != keyValueStore_.end(); });

    auto cvIt = keyCVs_.find(key);
    if (cvIt != keyCVs_.end()) {
      auto& cvs = cvIt->second;
      cvs.erase(std::remove(cvs.begin(), cvs.end(), cv), cvs.end());
      if (cvs.empty()) keyCVs_.erase(cvIt);
    }

    if (success) {
      auto valueIt = keyValueStore_.find(key);
      if (valueIt != keyValueStore_.end()) {
        value = valueIt->second;
      } else {
        success = false;
      }
    }

    if (!success) {
      LOGE("TCPStore: Get timeout for key: %s", key.c_str());
      return false;
    }
  }

  return detail::sendBlob(socket, value);
}

bool TCPStore::ServerImpl::handleAddCommand(socket_t socket) {
  std::string key;
  if (!detail::recvString(socket, key)) return false;

  int64_t addValue;
  if (!detail::recvBytes(socket, &addValue, sizeof(addValue))) return false;

  int64_t newValue = addValue;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& existing = keyValueStore_[key];
    if (!existing.empty()) {
      newValue += std::stoll(std::string(reinterpret_cast<const char*>(existing.data()), existing.size()));
    }
    auto str = std::to_string(newValue);
    keyValueStore_[key] = std::vector<uint8_t>(str.begin(), str.end());
    notifyKeyWaiters(key);
  }

  return detail::sendBytes(socket, &newValue, sizeof(newValue));
}

bool TCPStore::ServerImpl::handleCheckCommand(socket_t socket) {
  uint64_t numKeys;
  if (!detail::recvBytes(socket, &numKeys, sizeof(numKeys))) return false;

  std::vector<std::string> keys(numKeys);
  for (size_t i = 0; i < numKeys; i++) {
    if (!detail::recvString(socket, keys[i])) return false;
  }

  bool allExist = true;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& key : keys) {
      if (keyValueStore_.find(key) == keyValueStore_.end()) {
        allExist = false;
        break;
      }
    }
  }

  uint8_t result = allExist ? 1 : 0;
  return detail::sendBytes(socket, &result, sizeof(result));
}

bool TCPStore::ServerImpl::handleWaitCommand(socket_t socket) {
  uint64_t numKeys;
  if (!detail::recvBytes(socket, &numKeys, sizeof(numKeys))) return false;

  std::vector<std::string> keys(numKeys);
  for (size_t i = 0; i < numKeys; i++) {
    if (!detail::recvString(socket, keys[i])) return false;
  }

  auto allKeysExist = [&] {
    return std::all_of(keys.begin(), keys.end(),
                       [&](const auto& k) { return keyValueStore_.find(k) != keyValueStore_.end(); });
  };

  bool success;
  {
    std::unique_lock<std::mutex> lock(mutex_);

    if (allKeysExist()) {
      uint8_t result = 1;
      return detail::sendBytes(socket, &result, sizeof(result));
    }

    auto cv = std::make_shared<std::condition_variable>();
    std::vector<std::string> missingKeys;
    for (const auto& key : keys) {
      if (keyValueStore_.find(key) == keyValueStore_.end()) {
        keyCVs_[key].push_back(cv);
        missingKeys.push_back(key);
      }
    }

    success = waitForCondition(lock, cv, allKeysExist);

    for (const auto& key : missingKeys) {
      auto it = keyCVs_.find(key);
      if (it != keyCVs_.end()) {
        auto& cvs = it->second;
        cvs.erase(std::remove(cvs.begin(), cvs.end(), cv), cvs.end());
        if (cvs.empty()) keyCVs_.erase(it);
      }
    }
  }

  uint8_t result = success ? 1 : 0;
  return detail::sendBytes(socket, &result, sizeof(result));
}

bool TCPStore::ServerImpl::handleDeleteCommand(socket_t socket) {
  std::string key;
  if (!detail::recvString(socket, key)) return false;

  bool found = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = keyValueStore_.find(key);
    if (it != keyValueStore_.end()) {
      keyValueStore_.erase(it);
      found = true;
    }
  }

  uint8_t result = found ? 1 : 0;
  return detail::sendBytes(socket, &result, sizeof(result));
}

bool TCPStore::ServerImpl::handleNumKeysCommand(socket_t socket) {
  int64_t numKeys;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    numKeys = static_cast<int64_t>(keyValueStore_.size());
  }
  return detail::sendBytes(socket, &numKeys, sizeof(numKeys));
}

bool TCPStore::ServerImpl::handleWorkerRegister(socket_t socket) {
  int currentWorkers = registeredWorkers_.fetch_add(1) + 1;

  uint8_t confirmation = 1;
  if (!detail::sendBytes(socket, &confirmation, sizeof(confirmation))) {
    registeredWorkers_.fetch_sub(1);
    return false;
  }

  if (currentWorkers >= numWorkers_) {
    std::lock_guard<std::mutex> lock(mutex_);
    workerCV_.notify_all();
  }

  LOGI("TCPStore: Worker registered (%d/%d)", currentWorkers, numWorkers_);
  return true;
}

bool TCPStore::ServerImpl::handleWorkerUnregister(socket_t socket) {
  UNUSED(socket);
  registeredWorkers_.fetch_sub(1);
  return true;
}

}  // namespace tinytorch::distributed
