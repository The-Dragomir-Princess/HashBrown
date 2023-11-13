#include "hash_wrapper.h"

#include <iostream>
#include <random>
#include <unordered_set>
#include <limits>

const hash32_t hash_fn[] = { cityhash32, farmhash32, murmurhash32, xxhash32 };
const char *hash_fn_name[] = { "CityHash", "FarmHash", "MurmurHash3", "xxHash" };
const int num_hash_fn = 4;

void print_hex(FILE *out, const uint8_t *buf, uint32_t len) {
  // fprintf(out, "0x");
  for (auto c = buf; c < buf + len; c++) {
    fprintf(out, "%02x", *c);
  }
}

void hash_random(size_t size) {
  const char *fmt = "random-%llu-%d";
  const int num_iter = 5;
  
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint32_t> dist;
  
  char buf[64];
  FILE *out;

  if (size == 0) {
    return;
  }

  for (auto i = 0; i < num_iter; i++) {
    std::unordered_set<uint32_t> keys;

    sprintf(buf, fmt, size, i);
    if ((out = fopen(buf, "w")) == nullptr) {
      fprintf(stderr, "cannot open file %s\n", buf);
      exit(1);
    }

    for (auto j = 0; j < num_hash_fn; j++) {
      fprintf(out, "%s;", hash_fn_name[j]);
    }
    fprintf(out, "\n");

    for (size_t j = 0; j < size; j++) {
      uint32_t key;

      while (keys.count(key = dist(gen)));
      keys.insert(key);
      for (auto k = 0; k < num_hash_fn; k++) {
        auto hash = hash_fn[k](key);
        print_hex(out, reinterpret_cast<const uint8_t *>(&hash), sizeof(uint32_t));
        fprintf(out, ";");
      }
      fprintf(out, "\n");
    }
  }

  fclose(out);
}

void hash_contiguous(size_t size) {
  const char *fmt = "contiguous-%llu-%d";
  const int num_iter = 5;

  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint32_t> dist;
  
  char buf[64];
  FILE *out;
  
  if (size == 0) {
    return;
  }

  for (auto i = 0; i < num_iter; i++) {
    uint32_t begin;

    sprintf(buf, fmt, size, i);
    if ((out = fopen(buf, "w")) == nullptr) {
      fprintf(stderr, "cannot open file %s\n", buf);
      exit(1);
    }

    for (auto j = 0; j < num_hash_fn; j++) {
      fprintf(out, "%s;", hash_fn_name[j]);
    }
    fprintf(out, "\n");

    while ((begin = dist(gen)) >= std::numeric_limits<uint32_t>::max() - size);
    for (uint32_t key = begin; key < begin + size; key++) {
      for (auto k = 0; k < num_hash_fn; k++) {
        auto hash = hash_fn[k](key);
        print_hex(out, reinterpret_cast<const uint8_t *>(&hash), sizeof(uint32_t));
        fprintf(out, ";");
      }
      fprintf(out, "\n");
    }
  }

  fclose(out);
}

int main() {
  const int size = (1 << 20) / 5;
  hash_random(size);
  hash_contiguous(size);
}