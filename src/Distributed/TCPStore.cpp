/*
 * TinyTorch
 * @author  : keith@robot9.me
 *
 */

#include "TCPStore.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "Utils/Logger.h"
#include "Utils/Macros.h"

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

static void setupSignalHandlers() {
  // ignore SIGPIPE
  signal(SIGPIPE, SIG_IGN);
}

static bool sendBytes(socket_t socket, const void* data, size_t size) {
  const char* ptr = static_cast<const char*>(data);
  size_t remaining = size;

  while (remaining > 0) {
    ssize_t sent = send(socket, ptr, remaining, MSG_NOSIGNAL);

    if (sent < 0) {
      int error = errno;
      if (error == EPIPE || error == ECONNRESET || error == ECONNABORTED) {
        LOGE("TCPStore: Client disconnected (EPIPE/ECONNRESET/ECONNABORTED)");
        return false;
      } else if (error == EAGAIN) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      } else {
        LOGE("TCPStore: Send error: %d", error);
        return false;
      }
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
      if (error == ECONNRESET || error == EPIPE || error == ECONNABORTED) {
        LOGE("TCPStore: Client disconnected during recv");
        return false;
      } else if (error == EAGAIN) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      } else if (error == ETIMEDOUT) {
        LOGE("TCPStore: Recv timeout");
        return false;
      } else {
        LOGE("TCPStore: Recv error: %d", error);
        return false;
      }
    } else if (received == 0) {
      return false;
    }

    ptr += received;
    remaining -= received;
  }
  return true;
}

bool configureSocket(socket_t socket) {
  int optval = 1;

  if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0) {
    LOGE("TCPStore: Failed to set SO_REUSEADDR");
    return false;
  }

  if (setsockopt(socket, SOL_SOCKET, SO_KEEPALIVE, &optval, sizeof(optval)) < 0) {
    LOGE("TCPStore: Failed to set SO_KEEPALIVE");
    return false;
  }

  timeval timeout{};
  timeout.tv_sec = 30;  // 30s
  timeout.tv_usec = 0;
  if (setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
    LOGE("TCPStore: Failed to set SO_RCVTIMEO");
    return false;
  }

  if (setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0) {
    LOGE("TCPStore: Failed to set SO_SNDTIMEO");
    return false;
  }

  return true;
}

static bool isSocketConnected(socket_t socket) {
  int error = 0;
  socklen_t len = sizeof(error);
  int retval = getsockopt(socket, SOL_SOCKET, SO_ERROR, &error, &len);

  if (retval != 0) {
    return false;
  }

  if (error != 0) {
    return false;
  }

  char dummy;
  ssize_t result = send(socket, &dummy, 0, MSG_NOSIGNAL);
  if (result < 0) {
    int sendError = errno;
    if (sendError == EPIPE || sendError == ECONNRESET || sendError == ENOTCONN) {
      return false;
    }
  }

  return true;
}

}  // namespace detail

class TCPStore::ClientImpl {
 public:
  ClientImpl(std::string host, uint16_t port, const std::chrono::milliseconds& timeout);
  ~ClientImpl();

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
  std::string host_;
  uint16_t port_;
  std::chrono::milliseconds timeout_;
  socket_t clientSock_;

  int maxRetries = 10;
  int retryDelayMs = 1000;

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

  void cleanupFinishedThreads();
  void shutdownAllThreads();
  void cleanupDisconnectedClients();
  void startClientMonitoring();
  void stopClientMonitoring();

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

  std::vector<std::unique_ptr<std::thread>> clientThreads_;
  std::mutex clientThreadsMutex_;
  std::atomic<size_t> activeThreadCount_{0};

  std::unordered_set<socket_t> activeClients_;
  std::mutex activeClientsMutex_;
  std::thread clientMonitorThread_;
  std::atomic<bool> monitoringRunning_{false};
};

TCPStore::TCPStore(const std::string& host, uint16_t port, bool isServer, bool waitWorkers, int numWorkers,
                   const std::chrono::milliseconds& timeout)
    : Store(timeout), host_(host), port_(port), isServer_(isServer) {
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
      LOGE("TCPStore: Timeout while waiting for workers (expected: %d, registered: %d)", numWorkers,
           server_->getRegisteredWorkers());
      return;
    }
  }
}

TCPStore::~TCPStore() {
  if (client_) {
    bool ret = client_->unregisterWorker();
    if (!ret) {
      LOGE("TCPStore: unregisterWorker failed");
    }
  }
  client_.reset();
  server_.reset();
}

void TCPStore::set(const std::string& key, const std::vector<uint8_t>& value) {
  if (!client_->set(key, value)) {
    LOGE("TCPStore: set operation failed for key: %s", key.c_str());
  }
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
  std::vector<uint8_t> value;
  if (!client_->get(key, value)) {
    LOGE("TCPStore: get operation failed for key: %s", key.c_str());
  }
  return value;
}

int64_t TCPStore::add(const std::string& key, int64_t value) {
  int64_t newValue = -1;
  if (!client_->add(key, value, newValue)) {
    LOGE("TCPStore: add operation failed for key: %s", key.c_str());
  }
  return newValue;
}

bool TCPStore::deleteKey(const std::string& key) { return client_->deleteKey(key); }

bool TCPStore::check(const std::vector<std::string>& keys) { return client_->check(keys); }

int64_t TCPStore::getNumKeys() {
  int64_t numKeys = -1;
  if (!client_->getNumKeys(numKeys)) {
    LOGE("TCPStore: getNumKeys operation failed");
  }
  return numKeys;
}

void TCPStore::wait(const std::vector<std::string>& keys) {
  if (!client_->wait(keys)) {
    LOGE("TCPStore: wait operation failed or timed out");
  }
}

TCPStore::ClientImpl::ClientImpl(std::string host, uint16_t port, const std::chrono::milliseconds& timeout)
    : host_(std::move(host)), port_(port), timeout_(timeout), clientSock_(INVALID_SOCKET) {}

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

  if (!detail::configureSocket(clientSock_)) {
    closesocket(clientSock_);
    clientSock_ = INVALID_SOCKET;
    return false;
  }

  sockaddr_in addr{};
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port_);

  addrinfo hints{}, *result;
  std::memset(&hints, 0, sizeof(hints));
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

  // connect & retry
  bool connected = false;
  for (int i = 0; i < maxRetries && !connected; i++) {
    if (::connect(clientSock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      connected = true;
      break;
    }

    if (i < maxRetries - 1) {
      LOGE("TCPStore: Connection attempt failed, retrying in %d ms", retryDelayMs);
      std::this_thread::sleep_for(std::chrono::milliseconds(retryDelayMs));
    }
  }

  if (!connected) {
    LOGE("TCPStore: Failed to connect to server after %d attempts", maxRetries);
    closesocket(clientSock_);
    clientSock_ = INVALID_SOCKET;
    return false;
  }

  return true;
}

bool TCPStore::ClientImpl::set(const std::string& key, const std::vector<uint8_t>& value) const {
  detail::TCPStoreCommandType cmd = detail::TCPStoreCommandType::CMD_SET;

  if (!detail::sendBytes(clientSock_, &cmd, sizeof(cmd))) {
    LOGE("TCPStore: Failed to send SET command");
    return false;
  }

  uint64_t keySize = key.size();
  if (!detail::sendBytes(clientSock_, &keySize, sizeof(keySize)) ||
      !detail::sendBytes(clientSock_, key.data(), keySize)) {
    LOGE("TCPStore: Failed to send key for SET command");
    return false;
  }

  uint64_t valueSize = value.size();
  if (!detail::sendBytes(clientSock_, &valueSize, sizeof(valueSize)) ||
      !detail::sendBytes(clientSock_, value.data(), valueSize)) {
    LOGE("TCPStore: Failed to send value for SET command");
    return false;
  }

  uint8_t confirmation;
  if (!detail::recvBytes(clientSock_, &confirmation, sizeof(confirmation)) || confirmation != 1) {
    LOGE("TCPStore: Failed to receive confirmation for SET command");
    return false;
  }

  return true;
}

bool TCPStore::ClientImpl::get(const std::string& key, std::vector<uint8_t>& value) const {
  detail::TCPStoreCommandType cmd = detail::TCPStoreCommandType::CMD_GET;

  if (!detail::sendBytes(clientSock_, &cmd, sizeof(cmd))) {
    LOGE("TCPStore: Failed to send GET command");
    return false;
  }

  uint64_t keySize = key.size();
  if (!detail::sendBytes(clientSock_, &keySize, sizeof(keySize)) ||
      !detail::sendBytes(clientSock_, key.data(), keySize)) {
    LOGE("TCPStore: Failed to send key for GET command");
    return false;
  }

  uint64_t valueSize;
  if (!detail::recvBytes(clientSock_, &valueSize, sizeof(valueSize))) {
    LOGE("TCPStore: Failed to receive value size for GET command");
    return false;
  }

  value.resize(valueSize);
  if (valueSize > 0 && !detail::recvBytes(clientSock_, value.data(), valueSize)) {
    LOGE("TCPStore: Failed to receive value for GET command");
    return false;
  }

  return true;
}

bool TCPStore::ClientImpl::add(const std::string& key, int64_t addValue, int64_t& newValue) const {
  detail::TCPStoreCommandType cmd = detail::TCPStoreCommandType::CMD_ADD;

  if (!detail::sendBytes(clientSock_, &cmd, sizeof(cmd))) {
    LOGE("TCPStore: Failed to send ADD command");
    return false;
  }

  uint64_t keySize = key.size();
  if (!detail::sendBytes(clientSock_, &keySize, sizeof(keySize)) ||
      !detail::sendBytes(clientSock_, key.data(), keySize)) {
    LOGE("TCPStore: Failed to send key for ADD command");
    return false;
  }

  if (!detail::sendBytes(clientSock_, &addValue, sizeof(addValue))) {
    LOGE("TCPStore: Failed to send value for ADD command");
    return false;
  }

  if (!detail::recvBytes(clientSock_, &newValue, sizeof(newValue))) {
    LOGE("TCPStore: Failed to receive new value for ADD command");
    return false;
  }

  return true;
}

bool TCPStore::ClientImpl::deleteKey(const std::string& key) const {
  detail::TCPStoreCommandType cmd = detail::TCPStoreCommandType::CMD_DELETE_KEY;

  if (!detail::sendBytes(clientSock_, &cmd, sizeof(cmd))) {
    LOGE("TCPStore: Failed to send DELETE command");
    return false;
  }

  uint64_t keySize = key.size();
  if (!detail::sendBytes(clientSock_, &keySize, sizeof(keySize)) ||
      !detail::sendBytes(clientSock_, key.data(), keySize)) {
    LOGE("TCPStore: Failed to send key for DELETE command");
    return false;
  }

  uint8_t result;
  if (!detail::recvBytes(clientSock_, &result, sizeof(result))) {
    LOGE("TCPStore: Failed to receive result for DELETE command");
    return false;
  }

  return result == 1;
}

bool TCPStore::ClientImpl::check(const std::vector<std::string>& keys) const {
  detail::TCPStoreCommandType cmd = detail::TCPStoreCommandType::CMD_CHECK;

  if (!detail::sendBytes(clientSock_, &cmd, sizeof(cmd))) {
    LOGE("TCPStore: Failed to send CHECK command");
    return false;
  }

  uint64_t numKeys = keys.size();
  if (!detail::sendBytes(clientSock_, &numKeys, sizeof(numKeys))) {
    LOGE("TCPStore: Failed to send number of keys for CHECK command");
    return false;
  }

  for (const auto& key : keys) {
    uint64_t keySize = key.size();
    if (!detail::sendBytes(clientSock_, &keySize, sizeof(keySize)) ||
        !detail::sendBytes(clientSock_, key.data(), keySize)) {
      LOGE("TCPStore: Failed to send key for CHECK command");
      return false;
    }
  }

  uint8_t result;
  if (!detail::recvBytes(clientSock_, &result, sizeof(result))) {
    LOGE("TCPStore: Failed to receive result for CHECK command");
    return false;
  }

  return result == 1;
}

bool TCPStore::ClientImpl::getNumKeys(int64_t& numKeys) const {
  detail::TCPStoreCommandType cmd = detail::TCPStoreCommandType::CMD_NUM_KEYS;

  if (!detail::sendBytes(clientSock_, &cmd, sizeof(cmd))) {
    LOGE("TCPStore: Failed to send NUM_KEYS command");
    return false;
  }

  if (!detail::recvBytes(clientSock_, &numKeys, sizeof(numKeys))) {
    LOGE("TCPStore: Failed to receive number of keys");
    return false;
  }

  return true;
}

bool TCPStore::ClientImpl::wait(const std::vector<std::string>& keys) const {
  detail::TCPStoreCommandType cmd = detail::TCPStoreCommandType::CMD_WAIT;

  if (!detail::sendBytes(clientSock_, &cmd, sizeof(cmd))) {
    LOGE("TCPStore: Failed to send WAIT command");
    return false;
  }

  uint64_t numKeys = keys.size();
  if (!detail::sendBytes(clientSock_, &numKeys, sizeof(numKeys))) {
    LOGE("TCPStore: Failed to send number of keys for WAIT command");
    return false;
  }

  for (const auto& key : keys) {
    uint64_t keySize = key.size();
    if (!detail::sendBytes(clientSock_, &keySize, sizeof(keySize)) ||
        !detail::sendBytes(clientSock_, key.data(), keySize)) {
      LOGE("TCPStore: Failed to send key for WAIT command");
      return false;
    }
  }

  uint8_t result;
  if (!detail::recvBytes(clientSock_, &result, sizeof(result))) {
    LOGE("TCPStore: Failed to receive result for WAIT command");
    return false;
  }

  return result == 1;
}

bool TCPStore::ClientImpl::registerWorker() const {
  detail::TCPStoreCommandType cmd = detail::TCPStoreCommandType::CMD_WORKER_REGISTER;
  if (!detail::sendBytes(clientSock_, &cmd, sizeof(cmd))) {
    LOGE("TCPStore: Failed to send worker registration command");
    return false;
  }

  uint8_t confirmation;
  if (!detail::recvBytes(clientSock_, &confirmation, sizeof(confirmation)) || confirmation != 1) {
    LOGE("TCPStore: Failed to receive worker registration confirmation");
    return false;
  }

  isRegistered_.store(true);
  return true;
}

bool TCPStore::ClientImpl::unregisterWorker() const {
  if (!isRegistered_.load()) {
    return true;
  }

  detail::TCPStoreCommandType cmd = detail::TCPStoreCommandType::CMD_WORKER_UNREGISTER;
  if (!detail::sendBytes(clientSock_, &cmd, sizeof(cmd))) {
    LOGE("TCPStore: Failed to send worker unregistration command");
    return false;
  }

  isRegistered_.store(false);
  return true;
}

TCPStore::ServerImpl::ServerImpl(uint16_t port, int numWorkers, const std::chrono::milliseconds& timeout)
    : port_(port),
      numWorkers_(numWorkers),
      timeout_(timeout),
      shutdownServer_(false),
      registeredWorkers_(0),
      serverSock_(INVALID_SOCKET) {
  detail::setupSignalHandlers();
}

TCPStore::ServerImpl::~ServerImpl() {
  shutdownServer_.store(true);
  stopClientMonitoring();

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

  serverThread_.detach();
  shutdownAllThreads();
}

bool TCPStore::ServerImpl::start() {
  serverSock_ = socket(AF_INET, SOCK_STREAM, 0);
  if (serverSock_ == INVALID_SOCKET) {
    LOGE("TCPStore: Failed to create server socket");
    return false;
  }

  if (!detail::configureSocket(serverSock_)) {
    closesocket(serverSock_);
    serverSock_ = INVALID_SOCKET;
    return false;
  }

  sockaddr_in addr{};
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port_);
  addr.sin_addr.s_addr = INADDR_ANY;

  if (bind(serverSock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    LOGE("TCPStore: Failed to bind server socket");
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
  startClientMonitoring();
  return true;
}

bool TCPStore::ServerImpl::waitForWorkers() {
  if (numWorkers_ <= 0) {
    return true;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  bool success = workerCV_.wait_for(
      lock, timeout_, [this] { return shutdownServer_.load() || registeredWorkers_.load() >= numWorkers_; });

  if (!success) {
    LOGE("TCPStore: Timeout waiting for workers. Expected: %d, Registered: %d", numWorkers_, registeredWorkers_.load());
  } else if (!shutdownServer_.load()) {
    LOGI("TCPStore: All %d workers registered successfully", numWorkers_);
  }
  return success && !shutdownServer_.load();
}

int TCPStore::ServerImpl::getRegisteredWorkers() const { return registeredWorkers_.load(); }

void TCPStore::ServerImpl::startClientMonitoring() {
  monitoringRunning_.store(true);
  clientMonitorThread_ = std::thread([this]() {
    while (monitoringRunning_.load()) {
      cleanupDisconnectedClients();
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  });
}

void TCPStore::ServerImpl::stopClientMonitoring() {
  monitoringRunning_.store(false);
  clientMonitorThread_.detach();
}

void TCPStore::ServerImpl::cleanupDisconnectedClients() {
  std::lock_guard<std::mutex> lock(activeClientsMutex_);

  auto it = activeClients_.begin();
  while (it != activeClients_.end()) {
    if (!detail::isSocketConnected(*it)) {
      LOGI("TCPStore: Detected disconnected client, cleaning up");
      closesocket(*it);
      it = activeClients_.erase(it);
    } else {
      ++it;
    }
  }
}

void TCPStore::ServerImpl::cleanupFinishedThreads() {
  std::lock_guard<std::mutex> lock(clientThreadsMutex_);

  auto it = clientThreads_.begin();
  while (it != clientThreads_.end()) {
    if ((*it)->joinable()) {
      ++it;
    } else {
      it = clientThreads_.erase(it);
      activeThreadCount_.fetch_sub(1);
    }
  }
}

void TCPStore::ServerImpl::shutdownAllThreads() {
  std::vector<std::unique_ptr<std::thread>> threads;
  {
    std::lock_guard<std::mutex> lock(clientThreadsMutex_);
    threads = std::move(clientThreads_);
    clientThreads_.clear();
  }

  for (auto& thread : threads) {
    if (thread) {
      thread->detach();
    }
  }

  activeThreadCount_.store(0);
}

void TCPStore::ServerImpl::serverLoop() {
  while (!shutdownServer_.load()) {
    sockaddr_in clientAddr{};
    socklen_t addrLen = sizeof(clientAddr);
    socket_t clientSocket = accept(serverSock_, reinterpret_cast<sockaddr*>(&clientAddr), &addrLen);

    if (clientSocket == INVALID_SOCKET) {
      if (!shutdownServer_.load()) {
        LOGE("TCPStore: Failed to accept client connection");
      }
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(activeClientsMutex_);
      activeClients_.insert(clientSocket);
    }

    {
      std::lock_guard<std::mutex> lock(clientThreadsMutex_);

      if (clientThreads_.size() > 128) {
        cleanupFinishedThreads();
      }

      auto thread = std::make_unique<std::thread>(&ServerImpl::handleClientInThread, this, clientSocket);
      clientThreads_.push_back(std::move(thread));
      activeThreadCount_.fetch_add(1);
    }
  }
}

void TCPStore::ServerImpl::handleClientInThread(socket_t socket) {
  struct ThreadGuard {
    std::atomic<size_t>& counter;
    socket_t socket_;
    std::unordered_set<socket_t>& activeClients_;
    std::mutex& activeClientsMutex_;

    explicit ThreadGuard(std::atomic<size_t>& c, socket_t s, std::unordered_set<socket_t>& clients, std::mutex& mutex)
        : counter(c), socket_(s), activeClients_(clients), activeClientsMutex_(mutex) {}

    ~ThreadGuard() {
      counter.fetch_sub(1);
      {
        std::lock_guard<std::mutex> lock(activeClientsMutex_);
        activeClients_.erase(socket_);
      }
      closesocket(socket_);
    }
  };

  ThreadGuard guard(activeThreadCount_, socket, activeClients_, activeClientsMutex_);

  timeval timeout{};
  timeout.tv_sec = 1;  // 1s
  timeout.tv_usec = 0;
  setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));

  while (!shutdownServer_.load()) {
    if (!handleClient(socket)) {
      break;
    }
  }
}

bool TCPStore::ServerImpl::handleClient(socket_t socket) {
  if (shutdownServer_.load()) {
    return false;
  }

  detail::TCPStoreCommandType cmd;
  if (!detail::recvBytes(socket, &cmd, sizeof(cmd))) {
    return false;
  }

  if (shutdownServer_.load()) {
    return false;
  }

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
      LOGE("TCPStore: Unknown command type: %d", static_cast<int>(cmd));
      return false;
  }
}

bool TCPStore::ServerImpl::handleSetCommand(socket_t socket) {
  uint64_t keySize;
  if (!detail::recvBytes(socket, &keySize, sizeof(keySize))) {
    return false;
  }

  std::string key(keySize, '\0');
  if (!detail::recvBytes(socket, &key[0], keySize)) {
    return false;
  }

  uint64_t valueSize;
  if (!detail::recvBytes(socket, &valueSize, sizeof(valueSize))) {
    return false;
  }

  std::vector<uint8_t> value(valueSize);
  if (!detail::recvBytes(socket, value.data(), valueSize)) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    keyValueStore_[key] = std::move(value);

    auto it = keyCVs_.find(key);
    if (it != keyCVs_.end()) {
      for (auto& cv : it->second) {
        cv->notify_all();
      }
      keyCVs_.erase(it);
    }
  }

  uint8_t confirmation = 1;
  return detail::sendBytes(socket, &confirmation, sizeof(confirmation));
}

bool TCPStore::ServerImpl::handleGetCommand(socket_t socket) {
  uint64_t keySize;
  if (!detail::recvBytes(socket, &keySize, sizeof(keySize))) {
    return false;
  }

  std::string key(keySize, '\0');
  if (!detail::recvBytes(socket, &key[0], keySize)) {
    return false;
  }

  std::vector<uint8_t> value;
  std::shared_ptr<std::condition_variable> cv;
  bool success = false;

  {
    std::unique_lock<std::mutex> lock(mutex_);

    auto it = keyValueStore_.find(key);
    if (it != keyValueStore_.end()) {
      value = it->second;
      success = true;
    } else {
      cv = std::make_shared<std::condition_variable>();
      keyCVs_[key].push_back(cv);
    }
  }

  if (success) {
    uint64_t valueSize = value.size();
    return detail::sendBytes(socket, &valueSize, sizeof(valueSize)) &&
           (valueSize == 0 || detail::sendBytes(socket, value.data(), valueSize));
  }

  {
    std::unique_lock<std::mutex> lock(mutex_);

    if (timeout_ != kNoTimeout) {
      auto waitUntil = std::chrono::steady_clock::now() + timeout_;
      success = cv->wait_until(lock, waitUntil, [this, &key] {
        return shutdownServer_.load() || keyValueStore_.find(key) != keyValueStore_.end();
      });
    } else {
      cv->wait(lock,
               [this, &key] { return shutdownServer_.load() || keyValueStore_.find(key) != keyValueStore_.end(); });
      success = true;
    }

    if (shutdownServer_.load()) {
      success = false;
    }

    auto cvIt = keyCVs_.find(key);
    if (cvIt != keyCVs_.end()) {
      auto& cvs = cvIt->second;
      cvs.erase(std::remove(cvs.begin(), cvs.end(), cv), cvs.end());
      if (cvs.empty()) {
        keyCVs_.erase(cvIt);
      }
    }

    if (success) {
      auto valueIt = keyValueStore_.find(key);
      if (valueIt != keyValueStore_.end()) {
        value = valueIt->second;
      } else {
        success = false;
      }
    }
  }

  if (success) {
    uint64_t valueSize = value.size();
    return detail::sendBytes(socket, &valueSize, sizeof(valueSize)) &&
           (valueSize == 0 || detail::sendBytes(socket, value.data(), valueSize));
  }

  LOGE("TCPStore: Get timeout after %lld ms for key: %s", timeout_.count(), key.c_str());
  return false;
}

bool TCPStore::ServerImpl::handleAddCommand(socket_t socket) {
  uint64_t keySize;
  if (!detail::recvBytes(socket, &keySize, sizeof(keySize))) {
    return false;
  }

  std::string key(keySize, '\0');
  if (!detail::recvBytes(socket, &key[0], keySize)) {
    return false;
  }

  int64_t addValue;
  if (!detail::recvBytes(socket, &addValue, sizeof(addValue))) {
    return false;
  }

  int64_t newValue = addValue;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& value = keyValueStore_[key];
    if (!value.empty()) {
      auto buf = reinterpret_cast<const char*>(value.data());
      auto len = value.size();
      newValue += std::stoll(std::string(buf, len));
    }
    auto newValStr = std::to_string(newValue);
    keyValueStore_[key] = std::vector<uint8_t>(newValStr.begin(), newValStr.end());

    auto it = keyCVs_.find(key);
    if (it != keyCVs_.end()) {
      for (auto& cv : it->second) {
        cv->notify_all();
      }
      keyCVs_.erase(it);
    }
  }

  return detail::sendBytes(socket, &newValue, sizeof(newValue));
}

bool TCPStore::ServerImpl::handleCheckCommand(socket_t socket) {
  uint64_t numKeys;
  if (!detail::recvBytes(socket, &numKeys, sizeof(numKeys))) {
    return false;
  }

  std::vector<std::string> keys(numKeys);

  for (size_t i = 0; i < numKeys; i++) {
    uint64_t keySize;
    if (!detail::recvBytes(socket, &keySize, sizeof(keySize))) {
      return false;
    }

    keys[i].resize(keySize);
    if (!detail::recvBytes(socket, &keys[i][0], keySize)) {
      return false;
    }
  }

  bool allKeysExist = true;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& key : keys) {
      if (keyValueStore_.find(key) == keyValueStore_.end()) {
        allKeysExist = false;
        break;
      }
    }
  }

  uint8_t result = allKeysExist ? 1 : 0;
  return detail::sendBytes(socket, &result, sizeof(result));
}

bool TCPStore::ServerImpl::handleWaitCommand(socket_t socket) {
  uint64_t numKeys;
  if (!detail::recvBytes(socket, &numKeys, sizeof(numKeys))) {
    return false;
  }

  std::vector<std::string> keys(numKeys);
  for (size_t i = 0; i < numKeys; i++) {
    uint64_t keySize;
    if (!detail::recvBytes(socket, &keySize, sizeof(keySize))) {
      return false;
    }
    keys[i].resize(keySize);
    if (!detail::recvBytes(socket, &keys[i][0], keySize)) {
      return false;
    }
  }

  bool success = true;
  std::shared_ptr<std::condition_variable> cv;
  std::vector<std::string> missingKeys;

  {
    std::unique_lock<std::mutex> lock(mutex_);

    for (const auto& key : keys) {
      if (keyValueStore_.find(key) == keyValueStore_.end()) {
        missingKeys.push_back(key);
      }
    }

    if (missingKeys.empty()) {
      uint8_t result = 1;
      return detail::sendBytes(socket, &result, sizeof(result));
    }

    cv = std::make_shared<std::condition_variable>();
    for (const auto& key : missingKeys) {
      keyCVs_[key].push_back(cv);
    }
  }

  {
    std::unique_lock<std::mutex> lock(mutex_);

    if (timeout_ != kNoTimeout) {
      auto waitUntil = std::chrono::steady_clock::now() + timeout_;
      success = cv->wait_until(lock, waitUntil, [this, &keys] {
        return shutdownServer_.load() || std::all_of(keys.begin(), keys.end(), [this](const auto& key) {
                 return keyValueStore_.find(key) != keyValueStore_.end();
               });
      });
    } else {
      cv->wait(lock, [this, &keys] {
        return shutdownServer_.load() || std::all_of(keys.begin(), keys.end(), [this](const auto& key) {
                 return keyValueStore_.find(key) != keyValueStore_.end();
               });
      });
      success = true;
    }

    if (shutdownServer_.load()) {
      success = false;
    }

    for (const auto& key : missingKeys) {
      auto it = keyCVs_.find(key);
      if (it != keyCVs_.end()) {
        auto& cvs = it->second;
        cvs.erase(std::remove(cvs.begin(), cvs.end(), cv), cvs.end());
        if (cvs.empty()) {
          keyCVs_.erase(it);
        }
      }
    }
  }

  uint8_t result = success ? 1 : 0;
  return detail::sendBytes(socket, &result, sizeof(result));
}

bool TCPStore::ServerImpl::handleDeleteCommand(socket_t socket) {
  uint64_t keySize;
  if (!detail::recvBytes(socket, &keySize, sizeof(keySize))) {
    return false;
  }

  std::string key(keySize, '\0');
  if (!detail::recvBytes(socket, &key[0], keySize)) {
    return false;
  }

  bool success = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto kvIt = keyValueStore_.find(key);
    if (kvIt != keyValueStore_.end()) {
      keyValueStore_.erase(kvIt);
      success = true;
    }
  }

  uint8_t result = success ? 1 : 0;
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
    LOGE("TCPStore: Failed to send worker registration confirmation");
    registeredWorkers_.fetch_sub(1);
    return false;
  }

  if (currentWorkers >= numWorkers_) {
    std::lock_guard<std::mutex> lock(mutex_);
    workerCV_.notify_all();
  }

  LOGI("TCPStore: Worker registered. Current workers: %d/%d", currentWorkers, numWorkers_);
  return true;
}

bool TCPStore::ServerImpl::handleWorkerUnregister(socket_t socket) {
  UNUSED(socket);
  registeredWorkers_.fetch_sub(1);
  return true;
}

}  // namespace tinytorch::distributed