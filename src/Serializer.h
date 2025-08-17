/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <string>

#include "Modules.h"

namespace tinytorch {

class Serializer {
 public:
  static bool save(nn::Module& module, const std::string& path);
  static bool load(nn::Module& module, const std::string& path, bool strict = true);

 private:
  static constexpr uint64_t MAGIC_NUMBER = 0x54494E59;  // "TINY"
  static constexpr uint64_t VERSION = 1;

  struct TensorHeader {
    uint64_t nameLength;
    uint64_t ndim;
    uint64_t dtype;
    uint64_t dataOffset;
    uint64_t dataSize;
  };
};

inline bool save(nn::Module& module, const std::string& path) { return Serializer::save(module, path); }
inline bool load(nn::Module& module, const std::string& path, bool strict = true) {
  return Serializer::load(module, path, strict);
}

}  // namespace tinytorch
