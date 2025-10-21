/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "DistributedProcessGroup.h"

#include <iostream>
#include <regex>

#include "BackendNCCL.h"
#include "FileStore.h"
#include "Utils/Logger.h"

namespace tinytorch::distributed {

bool DistributedProcessGroup::initProcessGroup(BackendType backend, const std::string& initMethod, int rank,
                                               int worldSize, std::chrono::milliseconds timeout, bool waitWorkers) {
  if (processGroup_ != nullptr) {
    LOGE("ProcessGroup already initialized");
    return false;
  }
  auto config = parseInitString(initMethod, rank, worldSize);
  config.timeout = timeout;
  config.waitWorkers = waitWorkers;

  if (config.rank == -1 || config.worldSize == -1) {
    LOGE("Invalid ProcessGroup config");
    return false;
  }

  if (config.method == InitMethod::TCP || config.method == InitMethod::ENV) {
    LOGD("Init ProcessGroup: method=%s, rank=%d, world_size=%d, master=%s:%d, isServer=%s, waitWorkers=%s",
         initMethodToString(config.method),         //
         config.rank,                               //
         config.worldSize,                          //
         config.masterAddr.c_str(),                 //
         config.masterPort,                         //
         (config.rank == 0 ? "true" : "false"),     //
         (config.waitWorkers ? "true" : "false"));  //
  } else if (config.method == InitMethod::FILE) {
    LOGD("Init ProcessGroup: method=%s, rank=%d, world_size=%d, file=%s",
         initMethodToString(config.method),  //
         config.rank,                        //
         config.worldSize,                   //
         config.filePath.c_str());           //
  } else {
    LOGE("Invalid ProcessGroup config, initMethod: %s", initMethod.c_str());
    return false;
  }

  auto store = createStore(config);
  processGroup_ = std::make_shared<ProcessGroup>(store, config.rank, config.worldSize);
  processGroup_->setTimeout(timeout);
  processGroup_->setDefaultBackend(backend);

  switch (backend) {
    case NCCL: {
      auto ncclBackend = std::make_shared<BackendNCCL>(store, config.rank, config.worldSize);
      processGroup_->setBackend(DeviceType::CUDA, NCCL, ncclBackend);
      break;
    }
    default: {
      LOGE("Invalid ProcessGroup backend, backend=%d", backend);
      return false;
    }
  }

  LOGD("Init ProcessGroup success, backend=%s", backendTypeToString(backend));
  return true;
}

DistributedProcessGroup::InitConfig DistributedProcessGroup::parseInitString(const std::string& initString, int rank,
                                                                             int worldSize) {
  InitConfig config;
  config.rank = rank;
  config.worldSize = worldSize;

  if (initString == "env://") {
    config.method = InitMethod::ENV;
    const char* envAddr = std::getenv("MASTER_ADDR");
    const char* envPort = std::getenv("MASTER_PORT");
    const char* envWorldSize = std::getenv("WORLD_SIZE");
    const char* envRank = std::getenv("RANK");

    if (!envAddr) {
      LOGE("InitMethod: env, MASTER_ADDR not set");
      return {};
    }

    config.masterAddr = envAddr;
    if (envPort) {
      config.masterPort = std::stoi(envPort);
    } else {
      config.masterPort = TCPStore::kDefaultPort;
    }

    if (envWorldSize && config.worldSize == -1) {
      config.worldSize = std::stoi(envWorldSize);
    }
    if (envRank && config.rank == -1) {
      config.rank = std::stoi(envRank);
    }
  } else if (initString.substr(0, 6) == "tcp://") {
    config.method = InitMethod::TCP;

    // tcp://host:port or tcp://host (default port)
    std::regex tcpRegexWithPort(R"(tcp://([^:]+):(\d+))");
    std::regex tcpRegexNoPort(R"(tcp://([^:]+))");
    std::smatch matches;

    if (std::regex_match(initString, matches, tcpRegexWithPort)) {
      config.masterAddr = matches[1].str();
      config.masterPort = std::stoi(matches[2].str());
    } else if (std::regex_match(initString, matches, tcpRegexNoPort)) {
      config.masterAddr = matches[1].str();
      config.masterPort = TCPStore::kDefaultPort;
    } else {
      LOGE("InitMethod: tcp, format error: tcp://host[:port]");
      return {};
    }
  } else if (initString.substr(0, 7) == "file://") {
    config.method = InitMethod::FILE;
    config.filePath = initString.substr(7);

  } else {
    LOGE("InitMethod not support: %s", initString.c_str());
    return {};
  }
  return config;
}

std::shared_ptr<Store> DistributedProcessGroup::createStore(const InitConfig& config) {
  auto timeout = kBackendDefaultTimeout;

  // env
  if (config.method == InitMethod::ENV) {
    bool isServer = (config.rank == 0);
    int numWorkers = config.worldSize;
    bool waitWorkers = true;
    return std::make_shared<TCPStore>(config.masterAddr,  // host
                                      config.masterPort,  // port
                                      isServer,           // isServer
                                      waitWorkers,        // waitWorkers
                                      numWorkers,         // numWorkers
                                      timeout             // timeout
    );
  }

  // tcp
  if (config.method == InitMethod::TCP) {
    bool isServer = (config.rank == 0);
    int numWorkers = config.worldSize;
    bool waitWorkers = true;
    return std::make_shared<TCPStore>(config.masterAddr,                         // host
                                      static_cast<uint16_t>(config.masterPort),  // port
                                      isServer,                                  // isServer
                                      waitWorkers,                               // waitWorkers
                                      numWorkers,                                // numWorkers
                                      timeout                                    // timeout
    );
  }

  // file
  if (config.method == InitMethod::FILE) {
    return std::make_shared<FileStore>(config.filePath);
  }

  LOGE("Invalid config: %s", initMethodToString(config.method));
  return nullptr;
}

}  // namespace tinytorch::distributed