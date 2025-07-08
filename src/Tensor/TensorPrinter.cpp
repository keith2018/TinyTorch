/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "TensorPrinter.h"

#include <sstream>

#include "TensorImpl.h"
#include "Utils/Logger.h"

namespace tinytorch {

template <typename T>
static std::string printArray(const ArrayView<T>& vec, bool full) {
  auto size = vec.size();
  std::ostringstream oss;
  oss << "[";
  if (full || size <= 16) {
    for (size_t i = 0; i < size; i++) {
      oss << std::to_string(vec[i]);
      if (i != size - 1) {
        oss << ", ";
      }
    }
  } else {
    for (size_t i = 0; i < 8; i++) {
      oss << std::to_string(vec[i]);
      if (i != 7) {
        oss << ", ";
      }
    }
    oss << ", ... , ";
    for (size_t i = size - 8; i < size; i++) {
      oss << std::to_string(vec[i]);
      if (i != size - 1) {
        oss << ", ";
      }
    }
  }
  oss << "]";
  return oss.str();
}

template <typename T>
static std::string printArray(const std::vector<T>& vec, bool full) {
  ArrayView<T> arr(vec);
  return printArray(arr, full);
}

static std::string printDevice(const Device& device) {
  std::ostringstream oss;
  oss << "{ type:" << device::toString(device.type);
  oss << ", index:" << std::to_string(device.index);
  oss << " }";
  return oss.str();
}

static std::string printOptions(const Options& options) {
  std::ostringstream oss;
  oss << "{ device:" << printDevice(options.device_);
  oss << ", dtype:" << dtype::toString(options.dtype_);
  oss << ", requiresGrad:" << std::to_string(options.requiresGrad_);
  oss << ", pinnedMemory:" << std::to_string(options.pinnedMemory_);
  oss << " }";
  return oss.str();
}

static std::string printData(const TensorImpl& tensor, bool full) {
  std::ostringstream oss;
#define CASE_PRINT_DATA(T)                                             \
  case DType::T:                                                       \
    oss << printArray(tensor.toList<DTypeToType_t<DType::T>>(), full); \
    break

  switch (tensor.dtype()) {
    CASE_PRINT_DATA(Float32);
    CASE_PRINT_DATA(Float16);
    CASE_PRINT_DATA(BFloat16);
    CASE_PRINT_DATA(Int32);
    CASE_PRINT_DATA(Int64);
    CASE_PRINT_DATA(Bool);
    default:
      oss << "<unknown>";
  }
  return oss.str();
}

static std::string printTensorImpl(const TensorImpl& tensor, bool full) {
  std::ostringstream oss;
  oss << "Tensor {\n  shape: " << printArray(tensor.shape(), true);
  oss << ",\n  options: " << printOptions(tensor.options());
  oss << ",\n  data: " << printData(tensor, full);
  oss << "\n}";
  return oss.str();
}

void print(const TensorImpl& tensor, const char* tag, bool full) {
  std::ostringstream oss;
  if (tag) {
    oss << "[\"" << tag << "\"] ";
  }

  oss << printTensorImpl(tensor, full);
  LOGD("%s", oss.str().c_str());
}

void print(const Tensor& tensor, const char* tag, bool full) {
  std::ostringstream oss;
  if (tag) {
    oss << "[\"" << tag << "\"] ";
  }
  if (!tensor.defined()) {
    LOGD("Tensor { undefined }");
    return;
  }
  oss << printTensorImpl(tensor.getImpl(), full);
  LOGD("%s", oss.str().c_str());
}

}  // namespace tinytorch
