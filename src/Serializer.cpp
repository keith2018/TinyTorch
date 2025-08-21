/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Serializer.h"

#include <fstream>
#include <vector>

#include "Utils/Logger.h"
#include "Utils/MMapUtils.h"
#include "ankerl/unordered_dense.h"

namespace tinytorch {

bool Serializer::save(nn::Module& module, const std::string& path) {
  auto namedStates = module.namedStates();

  std::ofstream ofs(path, std::ios::binary);
  if (!ofs.is_open()) {
    LOGE("Error open file: %s", path.c_str());
    return false;
  }

  auto magic = MAGIC_NUMBER;
  auto version = VERSION;
  auto tensorCount = namedStates.size();

  ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
  ofs.write(reinterpret_cast<const char*>(&tensorCount), sizeof(tensorCount));

  size_t headerSize = sizeof(magic) + sizeof(version) + sizeof(tensorCount);
  for (const auto& [name, tensor] : namedStates) {
    headerSize += sizeof(TensorHeader);
    headerSize += name.length();
    headerSize += tensor->shape().size() * sizeof(int64_t);
  }

  size_t dataOffset = headerSize;

  // write header
  for (const auto& [name, tensor] : namedStates) {
    TensorHeader header{};
    header.nameLength = name.length();
    header.ndim = tensor->shape().size();
    header.dtype = static_cast<uint64_t>(tensor->dtype());
    header.dataOffset = dataOffset;
    header.dataSize = tensor->numel() * dtypeSize(tensor->dtype());

    ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));
    ofs.write(name.c_str(), static_cast<std::streamsize>(name.length()));

    for (auto dim : tensor->shape()) {
      int64_t dimension = dim;
      ofs.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
    }

    dataOffset += header.dataSize;
  }

  // write tensor data
  for (const auto& [name, tensor] : namedStates) {
    size_t tensorSize = tensor->numel() * dtypeSize(tensor->dtype());

    if (tensor->device().isCpu()) {
      ofs.write(static_cast<const char*>(tensor->dataPtr<>()), static_cast<std::streamsize>(tensorSize));
    } else {
      auto cpuTensor = tensor->to(DeviceType::CPU);
      ofs.write(static_cast<const char*>(cpuTensor.dataPtr<>()), static_cast<std::streamsize>(tensorSize));
    }
  }

  ofs.close();
  return true;
}

bool Serializer::load(nn::Module& module, const std::string& path, bool strict) {
  MMappingResult mappingResult = MMapUtils::mapFileForRead(path);
  if (!mappingResult.success) {
    LOGE("Error mapFileForRead: %s", path.c_str());
    return false;
  }

  const char* dataPtr = static_cast<const char*>(mappingResult.dataPtr);
  size_t offset = 0;

  // read header
  uint64_t magic = *reinterpret_cast<const uint64_t*>(dataPtr + offset);
  offset += sizeof(magic);

  if (magic != MAGIC_NUMBER) {
    LOGE("Invalid magic number in file: %s", path.c_str());
    MMapUtils::unmapFile(mappingResult);
    return false;
  }

  uint64_t version = *reinterpret_cast<const uint64_t*>(dataPtr + offset);
  offset += sizeof(version);

  if (version != VERSION) {
    LOGE("Unsupported version %u in file: %s", version, path.c_str());
    MMapUtils::unmapFile(mappingResult);
    return false;
  }

  uint64_t tensorCount = *reinterpret_cast<const uint64_t*>(dataPtr + offset);
  offset += sizeof(tensorCount);

  ankerl::unordered_dense::map<std::string, TensorPtr> name2tensor;
  for (const auto& [name, tensor] : module.namedStates()) {
    name2tensor[name] = tensor;
  }

  bool success = true;
  ankerl::unordered_dense::set<std::string> fileKeys;
  // read tensors
  for (uint64_t i = 0; i < tensorCount; i++) {
    const auto* header = reinterpret_cast<const TensorHeader*>(dataPtr + offset);
    offset += sizeof(TensorHeader);

    std::string name(dataPtr + offset, header->nameLength);
    offset += header->nameLength;
    fileKeys.insert(name);

    SizeVector shape;
    for (uint64_t j = 0; j < header->ndim; j++) {
      int64_t dim = *reinterpret_cast<const int64_t*>(dataPtr + offset);
      shape.pushBack(dim);
      offset += sizeof(int64_t);
    }

    auto iter = name2tensor.find(name);
    if (iter == name2tensor.end()) {
      LOGW("Unexpected key: %s", name.c_str());
      if (strict) {
        success = false;
      }
      continue;
    }

    TensorPtr tensor = iter->second;

    // check shape
    if (shape != tensor->shape()) {
      LOGE("shape not equal for tensor: %s", name.c_str());
      success = false;
      continue;
    }

    // check dtype
    auto dtype = static_cast<DType>(header->dtype);
    if (dtype != tensor->dtype()) {
      LOGE("dtype not equal for tensor: %s", name.c_str());
      success = false;
      continue;
    }

    // check dataSize
    size_t expectedSize = tensor->numel() * dtypeSize(tensor->dtype());
    if (header->dataSize != expectedSize) {
      LOGE("size not equal for tensor: %s", name.c_str());
      success = false;
      continue;
    }

    // check dataOffset
    if (header->dataOffset + header->dataSize > mappingResult.fileSize) {
      LOGE("Tensor data out of file range: %s", name.c_str());
      success = false;
      continue;
    }

    // copy data
    const void* tensorDataPtr = dataPtr + header->dataOffset;
    Storage::copyOnDevice(tensor->dataPtr<>(), tensor->device(), tensorDataPtr, Device::cpu(),
                          static_cast<int64_t>(header->dataSize));
  }

  // check missing keys
  for (const auto& [name, tensor] : name2tensor) {
    if (!fileKeys.count(name)) {
      LOGW("Missing key: %s", name.c_str());
      if (strict) {
        success = false;
      }
    }
  }

  MMapUtils::unmapFile(mappingResult);
  return success;
}

}  // namespace tinytorch