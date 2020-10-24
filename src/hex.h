#pragma once

#include "octopus_structs.h"

#include <cassert>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace hex {

static inline bool is_hex_digit(char hex) {
  return (hex >= 'a' && hex <= 'f') || (hex >= 'A' && hex <= 'F') ||
         (hex >= '0' && hex <= '9');
}

static inline char hex_digit_to_char(char hex) {
  if (hex >= 'A' && hex <= 'F') {
    return hex - 'A' + 10;
  } else if (hex >= 'a' && hex <= 'f') {
    return hex - 'a' + 10;
  } else {
    return hex - '0';
  }
}

static inline char char_to_hex_digit(char v) {
  assert(v < 16);
  if (v < 10) {
    return '0' + v;
  } else {
    return 'a' + (v - 10);
  }
}

static inline std::string to_hex_string(uint64_t nonce) {
  std::ostringstream sout;
  sout << std::hex << nonce;
  return sout.str();
}

static inline std::vector<char> hex_to_byte_vector(const std::string hex_str,
                                                   size_t fixed_size = 0) {
  std::vector<char> ret;
  ret.clear();
  size_t start = 0;
  if ((hex_str.size() >= 2) &&
      (hex_str[0] == '0' && (hex_str[1] == 'x' || hex_str[1] == 'X'))) {
    start = 2;
  }
  std::string inp_str;
  inp_str = hex_str.substr(start);
  if ((fixed_size != 0) && (2 * fixed_size > inp_str.size())) {
    inp_str = std::string(2 * fixed_size - inp_str.size(), '0') + inp_str;
  }
  for (size_t i = 0; i < inp_str.size(); i += 2) {
    if (!is_hex_digit(inp_str[i]) || !is_hex_digit(inp_str[i + 1])) {
      assert(false);
      return std::vector<char>();
    }
    char c = hex_digit_to_char(inp_str[i + 1]) +
             (hex_digit_to_char(inp_str[i]) << 4);
    ret.push_back(c);
  }
  return ret;
}

static inline octopus_h256_t
byte_vector_to_h256(const std::vector<char> bytes) {
  octopus_h256_t ret;
  memset(ret.b, 0, sizeof(ret));
  memcpy(ret.b, &bytes[0], bytes.size());
  return ret;
}

} // namespace hex