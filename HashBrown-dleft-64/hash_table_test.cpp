#include "dleft_fp_stash.hpp"
#include "xxhash.h"
#include <stdint.h>

#include <iostream>
#include <string>

#include <cassert>

# include "libcuckoo/cuckoohash_map.hh"
# include <unordered_map>
# include <unordered_set>
# include <vector>
# include <map>

# include <random>
# include <limits>
# include <chrono>
# include <type_traits>

#define __TEST_PERFORMANCE__

template<class K, class V, class Hasher>
class std_unordered_map_wrapper {
 public:
  auto insert(K &&key, V &&value) -> bool { return map.insert({key, value}).second; }

  auto erase(const K &key) -> bool { return map.erase(key) > 0; }

  auto find(const K &key, V &value) const -> bool {
    auto itr = map.find(key);
    if (itr == map.end()) {
      return false;
    }
    value = itr->second;
    return true;
  }

  void reserve(size_t size) { map.reserve(size); }

  void clear() { map.clear(); }

  auto load_factor() const -> double { return map.load_factor(); }
 private:
  std::unordered_map<K, V, Hasher> map;
};

class HashTableTest {
 private:
  static constexpr uint64_t seed64 = 0x42ae2f8ce193f9da;

  template<class K, uint64_t seed>
  class HasherULL {
    public:
    auto operator()(const K &key) const -> uint64_t {
      return XXH64(&key, sizeof(K), seed);
    }
  };
  using Hasher64 = HasherULL<uint32_t, seed64>;

  // TODO: add more hash maps here
  using std_unordered_map = std_unordered_map_wrapper<uint32_t, uint32_t, Hasher64>;
  using cuckoo_map = libcuckoo::cuckoohash_map<uint32_t, uint32_t, Hasher64>;
  using dleft_map = DleftFpStash<uint32_t, uint32_t, Hasher64>;

  static constexpr char std_unordered_map_name[] = "std_unordered_map";
  static constexpr char cuckoo_map_name[] = "cuckoohash_map";
  static constexpr char dleft_map_name[] = "dleft_map";

 public:
  static void RunAllTests() {
    // TODO: test more hash maps
    TestPerformance<std_unordered_map, std_unordered_map_name>();
    TestPerformance<cuckoo_map, cuckoo_map_name>();
    TestPerformance<dleft_map, dleft_map_name>();
  }

 private:
  template<class map_type, const char *name>
  static void TestPerformance() {
    printf("[PERFORMANCE TEST]\nTesting %s\n", name);

    std::vector<uint32_t> keys;
    std::unordered_set<uint32_t> key_set;
    GetDataset(keys, key_set);

    map_type map;
    map.clear();
    map.reserve(keys.size());

    std::string filename = std::string("data/") + name + ".csv";
    FILE *file = fopen(filename.c_str(), "w");
    // FILE *file = stdout;
    fprintf(file, "Load Factor, Write Latency(ns), Postive Read Latency(ns), Negative Read Latency(ns)\n");

    int num_batches = 16;
    int batch_size = keys.size() / num_batches;
    for (int i = 0; i < num_batches; i++) {
      double write_latency = TestWriteLatency(map, keys, i * batch_size, (i + 1) * batch_size);
      double positive_read_latency = TestReadPositiveLatency(map, keys, 0, (i + 1) * batch_size);
      double negative_read_latency = TestReadNegativeLatency(map, key_set, (i + 1) * batch_size);
      double load_factor = TestLoadFactor(map);
      fprintf(file, "%lf,%lf,%lf,%lf\n", load_factor, write_latency, positive_read_latency, negative_read_latency);
    }
  }

  template<class map_type>
  static auto TestWriteLatency(map_type &map, const std::vector<uint32_t> &dataset,
                               int begin, int end) -> double {
    size_t total_ns = 0;
    for (int i = begin; i < end; i++) {
      auto key = dataset[i], value = dataset[i];
      const auto start = std::chrono::high_resolution_clock::now();
      map.insert(std::forward<uint32_t>(key), std::forward<uint32_t>(value));
      const auto end = std::chrono::high_resolution_clock::now();
      total_ns += (end - start).count();
    }
    return 1.0 * total_ns / (end - begin);
  }

  template<class map_type>
  static auto TestReadPositiveLatency(const map_type &map, const std::vector<uint32_t> &dataset,
                                      int begin, int end) -> double {
    size_t total_ns = 0;
    for (int i = begin; i < end; i++) {
      uint32_t value;
      const auto start = std::chrono::high_resolution_clock::now();
      map.find(dataset[i], value);
      const auto end = std::chrono::high_resolution_clock::now();
      total_ns += (end - start).count();
    }
    return 1.0 * total_ns / (end - begin);
  }

  template<class map_type>
  static auto TestReadNegativeLatency(const map_type &map, const std::unordered_set<uint32_t> &dataset, int size) -> double {
    size_t total_ns = 0;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint32_t> dist;

    for (size_t i = 0; i < size; i++) {
      uint32_t key, value;
      while (dataset.count(key = dist(gen)));
      const auto start = std::chrono::high_resolution_clock::now();
      map.find(key, value);
      const auto end = std::chrono::high_resolution_clock::now();
      total_ns += (end - start).count();
    }
    return 1.0 * total_ns / size;
  }

  template<class map_type>
  static auto TestLoadFactor(const map_type &map) -> double {
    return map.load_factor();
  }

  // TODO: use a networking dataset
  static void GetDataset(std::vector<uint32_t> &keys, std::unordered_set<uint32_t> &key_set) {
    const size_t size = 1000000;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint32_t> dist;

    for (size_t i = 0; i < size; i++) {
      uint32_t key;
      while (key_set.count(key = dist(gen)));
      key_set.insert(key);
      keys.emplace_back(key);
    }
  }
};

int main() {
 #ifdef __TEST_DLEFT__
  DleftTest::RunAllTests();
 #endif
 #ifdef __TEST_PERFORMANCE__
  HashTableTest::RunAllTests();
 #endif
}