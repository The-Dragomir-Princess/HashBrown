#include "hash_wrapper.h"

const static uint32_t HASH_SEED = 0x1234abcd;

auto cityhash32(uint32_t key) -> uint32_t {
  return static_cast<uint32_t>(CityHash64(reinterpret_cast<const char *>(&key), sizeof(uint32_t)));
}

auto farmhash32(uint32_t key) -> uint32_t {
  return farmhash::Hash32(reinterpret_cast<const char *>(&key), sizeof(uint32_t));
}

auto murmurhash32(uint32_t key) -> uint32_t {
  uint32_t ret;
  MurmurHash3_x86_32(&key, sizeof(uint32_t), HASH_SEED, &ret);
  return ret;
}

auto xxhash32(uint32_t key) -> uint32_t {
  return XXH32(&key, sizeof(uint32_t), HASH_SEED);
}