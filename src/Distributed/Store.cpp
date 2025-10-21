/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Store.h"

namespace tinytorch::distributed {

void Store::set(const std::string& key, const std::string& value) {
  set(key, std::vector<uint8_t>(value.begin(), value.end()));
}

std::string Store::getString(const std::string& key) {
  auto value = get(key);
  return {value.begin(), value.end()};
}

}  // namespace tinytorch::distributed