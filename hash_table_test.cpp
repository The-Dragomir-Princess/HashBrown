#include "dleft_fp_stash.hpp"
#include "xxhash.h"
#include <stdint.h>

#include <iostream>
#include <string>

#include <cassert>

#define __PERFORMANCE_TEST__
#ifdef __PERFORMANCE_TEST__
# include "libcuckoo/cuckoohash_map.hh"
# include <unordered_map>
# include <unordered_set>
# include <vector>
# include <map>

# include <random>
# include <limits>
# include <chrono>
# include <type_traits>
#endif

const uint32_t seed1 = 0x5d445e6e;
const uint32_t seed2 = 0xf09ad611;
const uint64_t seed64 = 0x42ae2f8ce193f9da;

template<class K, uint64_t seed>
class Hasher {
 public:
  auto operator()(const K &key) const -> uint32_t {
    return XXH32(&key, sizeof(K), seed);
  }
};

using Hasher1 = Hasher<uint32_t, seed1>;
using Hasher2 = Hasher<uint32_t, seed2>;

#ifdef __PERFORMANCE_TEST__
template<class K, uint64_t seed>
class HasherULL {
  public:
  auto operator()(const K &key) const -> uint64_t {
    return XXH64(&key, sizeof(K), seed);
  }
};
using Hasher64 = HasherULL<uint32_t, seed64>;

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

using std_unordered_map = std_unordered_map_wrapper<uint32_t, uint32_t, Hasher64>;
using cuckoo_map = libcuckoo::cuckoohash_map<uint32_t, uint32_t, Hasher64>;
using dleft_map = DleftFpStash<uint32_t, uint32_t, Hasher1, Hasher2>;

const char std_unordered_map_name[] = "std_unordered_map";
const char cuckoo_map_name[] = "cuckoohash_map";
const char dleft_map_name[] = "dleft_map";
#else
using HashTableType = DleftFpStash<uint32_t, uint32_t, Hasher1, Hasher2>;
#endif

class HashTableTest {
 public:
  static void RunAllTests() {
#ifdef __PERFORMANCE_TEST__
    TestPerformance<std_unordered_map, std_unordered_map_name>();
    TestPerformance<cuckoo_map, cuckoo_map_name>();
    TestPerformance<dleft_map, dleft_map_name>();

    TestMaxLoadFactor();
#else
    TestStashBucket();
    TestBucket();
    TestDleft();
#endif
  }

 private:
#ifdef __PERFORMANCE_TEST__
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

    if constexpr (std::is_same<map_type, dleft_map>::value) {
     #ifdef __COUNT_FALSE_POSITIVES__
      false_positive = 0;
      TestReadPositiveLatency(map, keys, 0, keys.size());
      printf("Positive Read: %ld false positives\n", false_positive);

      false_positive = 0;
      TestReadNegativeLatency(map, key_set, keys.size());
      printf("Negative Read: %ld false positives\n", false_positive);
     #endif
     #ifdef __COUNT_OVERFLOWS__
      double bucket_lf = 1.0 * (map.size() - minor_overflows - major_overflows) / map.BucketCapacity();
      double stash_lf = 1.0 * (minor_overflows + major_overflows) / map.StashBucketCapacity();
      printf("Bucket Load Factor: %lf\n", bucket_lf);
      printf("Stash Bucket Load Factor: %lf\n", stash_lf);
      printf("# of Minor Overflows: %ld\n", minor_overflows);
      printf("# of Major Overflows: %ld\n", major_overflows);
     #endif
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

  static void TestMaxLoadFactor() {
    dleft_map map(1000000);
    size_t bucket_total{0}, stash_bucket_total{0};

    uint32_t key = 0;
    while (map.Append(std::forward<uint32_t>(key), std::forward<uint32_t>(key), Hasher1()(key), Hasher2()(key))) {
      key++;
    }

    std::map<size_t, size_t> bucket_distribution, stash_bucket_distribution;
    for (size_t i = 0; i < map.num_buckets_; i++) {
      bucket_distribution[map.buckets_[i].GetTotal()]++;
      bucket_total += map.buckets_[i].GetSize();
    }
    for (size_t i = 0; i < map.num_stash_buckets_; i++) {
      stash_bucket_distribution[map.stash_buckets_[i].GetSize()]++;
      stash_bucket_total += map.stash_buckets_[i].GetSize();
    }

    printf("Bucket Distribution:\n");
    for (auto &p : bucket_distribution) {
      printf("%lu:%lu,", p.first, p.second);
    }
    printf("\nStash Bucket Distribution:\n");
    for (auto &p : stash_bucket_distribution) {
      printf("%lu:%lu,", p.first, p.second);
    }
    printf("\nBucket Load Factor: %lf, Stash Bucket Load Factor: %lf\n",
           1.0 * bucket_total / map.BucketCapacity(),
           1.0 * stash_bucket_total / map.StashBucketCapacity());
  }
#else
  static void TestStashBucket() {
    TestStashBucketInsertMinorOverflow();
    TestStashBucketEraseMinorOverflow();
    TestStashBucketFindMinorOverflow();

    TestStashBucketAppendMajorOverflow();
    TestStashBucketEraseMajorOverflow();
    TestStashBucketFindMajorOverflow();
    TestStashBucketInsertMajorOverflow();
  }

  static void TestBucket() {
    TestBucketAppend();
    TestBucketErase();
    TestBucketFind();

    TestBucketAppendWithOverflow();
    TestBucketEraseWithOverflow();
    TestBucketFindWithOverflow();
    TestBucketInsertWithOverflow();
  }

  static void TestDleft() {
    TestDleftAppend();
    TestDleftErase();
    TestDleftFind();
    TestDleftInsert();
    TestDleftResize();
  }

  static void TestStashBucketInsertMinorOverflow() {
    printf("[TEST STASH BUCKET INSERT MINOR OVERFLOW]\n");

    HashTableType::StashBucket bucket;

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.InsertMinorOverflow(i, i) != bucket.invalid_pos);
    }
    assert(bucket.InsertMinorOverflow(2023u, 2023u) == bucket.invalid_pos);

    printf("[PASSED]\n");
  }

  static void TestStashBucketEraseMinorOverflow() {
    printf("[TEST STASH BUCKET ERASE MINOR OVERFLOW]\n");

    HashTableType::StashBucket bucket;
    uint8_t pos[bucket.bucket_capacity];

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      pos[i] = bucket.InsertMinorOverflow(i, i);
      assert(pos[i] != bucket.invalid_pos);
    }

    for (auto i = 0; i < bucket.bucket_capacity; i += 2) {
      assert(!bucket.EraseMinorOverflow(i+1, pos[i]));
      assert(bucket.EraseMinorOverflow(i, pos[i]));
    }

    for (auto i = 0; i < bucket.bucket_capacity; i += 2) {
      assert(bucket.InsertMinorOverflow(i, i) != bucket.invalid_pos);
    }

    printf("[PASSED]\n");
  }

  static void TestStashBucketFindMinorOverflow() {
    printf("[TEST STASH BUCKET FIND MINOR OVERFLOW]\n");

    HashTableType::StashBucket bucket;
    uint8_t pos[bucket.bucket_capacity];

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      pos[i] = bucket.InsertMinorOverflow(i, i);
      assert(pos[i] != bucket.invalid_pos);
    }

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      uint32_t value;
      assert(!bucket.FindMinorOverflow(i+1, &value, pos[i]));
      assert(bucket.FindMinorOverflow(i, &value, pos[i]));
      assert(value == i);
    }

    for (auto i = 0; i < bucket.bucket_capacity; i += 2) {
      assert(bucket.EraseMinorOverflow(i, pos[i]));
    }

    for (auto i = bucket.bucket_capacity - 2; i >= 0; i -= 2) {
      uint32_t value;
      pos[i] = bucket.InsertMinorOverflow(i, i*2);
      assert(pos[i] != bucket.invalid_pos);
      assert(bucket.FindMinorOverflow(i, &value, pos[i]));
      assert(value == i*2);
    }

    printf("[PASSED]\n");
  }

  static void TestStashBucketAppendMajorOverflow() {
    printf("[TEST STASH BUCKET APPEND MAJOR OVERFLOW]\n");

    {
      HashTableType::StashBucket bucket;
      for (auto i = 0; i < bucket.max_major_overflows; i++) {
        assert(bucket.AppendMajorOverflow(i, i, Hasher1()(i)));
      }
      assert(!bucket.AppendMajorOverflow(2023u, 2023u, static_cast<uint16_t>(Hasher1()(2023u))));
    }
    {
      HashTableType::StashBucket bucket;

      for (auto i = 0; i < bucket.bucket_capacity - bucket.max_major_overflows / 2; i++) {
        assert(bucket.InsertMinorOverflow(i, i) != bucket.invalid_pos);
      }
      for (auto i = bucket.bucket_capacity - bucket.max_major_overflows / 2; i < bucket.bucket_capacity; i++) {
        assert(bucket.AppendMajorOverflow(i, i, Hasher1()(i)));
      }
      assert(!bucket.AppendMajorOverflow(2023u, 2023u, static_cast<uint16_t>(Hasher1()(2023u))));
    }

    printf("[PASSED]\n");
  }

  static void TestStashBucketEraseMajorOverflow() {
    printf("[TEST STASH BUCKET ERASE MAJOR OVERFLOW]\n");

    HashTableType::StashBucket bucket;

    for (auto i = 0; i < bucket.max_major_overflows; i++) {
      assert(bucket.AppendMajorOverflow(i, i, Hasher1()(i)));
    }

    for (auto i = 0; i < bucket.max_major_overflows; i += 2) {
      assert(bucket.EraseMajorOverflow(i, Hasher1()(i)));
      assert(!bucket.EraseMajorOverflow(i, Hasher1()(i)));
    }

    for (auto i = 0; i < bucket.max_major_overflows; i += 2) {
      assert(bucket.AppendMajorOverflow(i, i, Hasher1()(i)));
    }

    printf("[PASSED]\n");
  }

  static void TestStashBucketFindMajorOverflow() {
    printf("[TEST STASH BUCKET FIND MAJOR OVERFLOW]\n");

    HashTableType::StashBucket bucket;

    for (auto i = 0; i < bucket.max_major_overflows; i++) {
      assert(bucket.AppendMajorOverflow(i, i, Hasher1()(i)));
    }
    for (auto i = 0; i < bucket.max_major_overflows; i++) {
      uint32_t value;
      assert(bucket.FindMajorOverflow(i, &value, Hasher1()(i)));
      assert(!bucket.FindMajorOverflow(i+1, &value, Hasher1()(i)));
      assert(value == i);
    }

    for (auto i = 0; i < bucket.max_major_overflows; i += 2) {
      assert(bucket.EraseMajorOverflow(i, Hasher1()(i)));
      assert(!bucket.FindMajorOverflow(i, nullptr, Hasher1()(i)));
    }

    for (auto i = bucket.max_major_overflows - 2; i >= 0; i -= 2) {
      uint32_t value;
      assert(bucket.AppendMajorOverflow(i, i*2, Hasher1()(i)));
      assert(bucket.FindMajorOverflow(i, &value, Hasher1()(i)));
      assert(value == i*2);
    }

    printf("[PASSED]\n");
  }

  static void TestStashBucketInsertMajorOverflow() {
    printf("[TEST STASH BUCKET INSERT MAJOR OVERFLOW]\n");

    HashTableType::StashBucket bucket;

    for (auto i = 0; i < bucket.max_major_overflows; i++) {
      assert(bucket.AppendMajorOverflow(i, i, Hasher1()(i)));
    }
    for (auto i = 0; i < bucket.max_major_overflows; i += 2) {
      assert(bucket.InsertMajorOverflow(i, i*2, Hasher1()(i)));
    }
    for (auto i = 0; i < bucket.max_major_overflows; i += 2) {
      uint32_t value;
      assert(bucket.FindMajorOverflow(i, &value, Hasher1()(i)));
      if (i % 2 == 0) {
        assert(value == i*2);
      } else {
        assert(value == i);
      }
    }

    printf("[PASSED]\n");
  }

  static void TestBucketAppend() {
    printf("[TEST BUCKET APPEND]\n");

    HashTableType::Bucket bucket;

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Append(i, i, Hasher1()(i), nullptr));
    }
    assert(!bucket.Append(2023u, 2023u, Hasher1()(2023u), nullptr));

    printf("[PASSED]\n");
  }

  static void TestBucketErase() {
    printf("[TEST BUCKET ERASE]\n");

    HashTableType::Bucket bucket;

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Append(i, i, Hasher1()(i), nullptr));
    }

    for (auto i = 0; i < bucket.bucket_capacity; i += 2) {
      assert(bucket.Erase(i, Hasher1()(i), nullptr));
      assert(!bucket.Erase(i, Hasher1()(i), nullptr));
    }

    printf("[PASSED]\n");
  }

  static void TestBucketFind() {
    printf("[TEST BUCKET FIND]\n");

    HashTableType::Bucket bucket;

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Append(i, i, Hasher1()(i), nullptr));
    }

    for (auto i = 0; i < bucket.bucket_capacity; i += 2) {
      uint32_t value;
      assert(bucket.Find(i, &value, Hasher1()(i), nullptr));
      assert(value == i);
      assert(bucket.Erase(i, Hasher1()(i), nullptr));
      assert(!bucket.Find(i, nullptr, Hasher1()(i), nullptr));
      assert(bucket.Append(i, i*2, Hasher1()(i), nullptr));
    }

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      uint32_t value;
      assert(bucket.Find(i, &value, Hasher1()(i), nullptr));
      if (i % 2 == 0) {
        assert(value == i*2);
      } else {
        assert(value == i);
      }
    }

    printf("[PASSED]\n");
  }

  static void TestBucketInsert() {
    printf("[TEST BUCKET INSERT]\n");

    HashTableType::Bucket bucket;

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Append(i, i, Hasher1()(i), nullptr));
    }

    for (auto i = 0; i < bucket.bucket_capacity; i += 2) {
      assert(bucket.Insert(i, i*2, Hasher1()(i), nullptr));
    }

    for (auto i = 0; i < bucket.bucket_capacity; i += 2) {
      uint32_t value;
      assert(bucket.Find(i, &value, Hasher1()(i), nullptr));
      if (i % 2 == 0) {
        assert(value == i*2);
      } else {
        assert(value == i);
      }
    }

    printf("[PASSED]\n");
  }

  static void TestBucketAppendWithOverflow() {
    printf("[TEST BUCKET APPEND WITH OVERFLOW]\n");

    HashTableType::Bucket bucket;
    HashTableType::StashBucket stash_bucket;

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Append(i, i, Hasher1()(i), &stash_bucket));
    }
    assert(bucket.overflow_count_ == 0);

    for (auto i = bucket.bucket_capacity; i < bucket.bucket_capacity + bucket.max_minor_overflows; i++) {
      assert(bucket.Append(i, i, Hasher1()(i), &stash_bucket));
    }
    assert(bucket.overflow_count_ == bucket.max_minor_overflows);
    assert(bucket.GetMinorOverflowCount() == bucket.max_minor_overflows);
    for (auto i = 0; i < stash_bucket.max_major_overflows; i++) {
      assert(stash_bucket.position_[i] == stash_bucket.invalid_pos);
    }

    for (auto i = bucket.bucket_capacity + bucket.max_minor_overflows;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i++) {
      assert(bucket.Append(i, i, Hasher1()(i), &stash_bucket));
    }
    assert(bucket.overflow_count_ == bucket.max_minor_overflows + stash_bucket.max_major_overflows);
    assert(bucket.GetMinorOverflowCount() == bucket.max_minor_overflows);
    for (auto i = 0; i < stash_bucket.max_major_overflows; i++) {
      assert(stash_bucket.position_[i] != stash_bucket.invalid_pos);
    }
  
    assert(!bucket.Append(2023u, 2023u, Hasher1()(2023u), &stash_bucket));

    printf("[PASSED]\n");
  }

  static void TestBucketEraseWithOverflow() {
    printf("[TEST BUCKET ERASE WITH OVERFLOW]\n");

    HashTableType::Bucket bucket;
    HashTableType::StashBucket stash_bucket;

    for (auto i = 0;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i++) {
      assert(bucket.Append(i, i, Hasher1()(i), &stash_bucket));
    }
    assert(bucket.overflow_count_ == bucket.max_minor_overflows + stash_bucket.max_major_overflows);
    assert(bucket.GetMinorOverflowCount() == bucket.max_minor_overflows);
    for (auto i = 0; i < stash_bucket.max_major_overflows; i++) {
      assert(stash_bucket.position_[i] != stash_bucket.invalid_pos);
    }

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Erase(i, Hasher1()(i), &stash_bucket));
      assert(!bucket.Erase(i, Hasher1()(i), &stash_bucket));
    }
    assert(bucket.GetSize() == 0);

    for (auto i = bucket.bucket_capacity + bucket.max_minor_overflows;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i++) {
      assert(bucket.Erase(i, Hasher1()(i), &stash_bucket));
      assert(!bucket.Erase(i, Hasher1()(i), &stash_bucket));
    }
    assert(bucket.overflow_count_ == bucket.max_minor_overflows);
    assert(bucket.GetMinorOverflowCount() == bucket.max_minor_overflows);
    for (auto i = 0; i < stash_bucket.max_major_overflows; i++) {
      assert(stash_bucket.position_[i] == stash_bucket.invalid_pos);
    }

    for (auto i = bucket.bucket_capacity; i < bucket.bucket_capacity + bucket.max_minor_overflows; i++) {
      assert(bucket.Erase(i, Hasher1()(i), &stash_bucket));
      assert(!bucket.Erase(i, Hasher1()(i), &stash_bucket));
    }
    assert(bucket.overflow_count_ == 0);
    assert(bucket.GetMinorOverflowCount() == 0);

    printf("[PASSED]\n");
  }

  static void TestBucketFindWithOverflow() {
    printf("[TEST BUCKET FIND WITH OVERFLOW]\n");

    HashTableType::Bucket bucket;
    HashTableType::StashBucket stash_bucket;

    for (auto i = 0;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i++) {
      assert(bucket.Append(i, i, Hasher1()(i), &stash_bucket));
    }
    assert(bucket.overflow_count_ == bucket.max_minor_overflows + stash_bucket.max_major_overflows);
    assert(bucket.GetMinorOverflowCount() == bucket.max_minor_overflows);
    for (auto i = 0; i < stash_bucket.max_major_overflows; i++) {
      assert(stash_bucket.position_[i] != stash_bucket.invalid_pos);
    }

    for (auto i = 0;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i++) {
      uint32_t value;
      assert(bucket.Find(i, &value, Hasher1()(i), &stash_bucket));
      assert(value == i);
      
      assert(bucket.Erase(i, Hasher1()(i), &stash_bucket));
      assert(!bucket.Find(i, nullptr, Hasher1()(i), &stash_bucket));

      assert(bucket.Append(i, i*2, Hasher1()(i), &stash_bucket));
      assert(bucket.Find(i, &value, Hasher1()(i), &stash_bucket));
      assert(value == i*2);
    }

    printf("[PASSED]\n");
  }

  static void TestBucketInsertWithOverflow() {
    printf("[TEST BUCKET INSERT WITH OVERFLOW]\n");

    HashTableType::Bucket bucket;
    HashTableType::StashBucket stash_bucket;

    for (auto i = 0;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i++) {
      assert(bucket.Append(i, i, Hasher1()(i), &stash_bucket));
    }

    for (auto i = 0;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i += 2) {
      assert(bucket.Insert(i, i*2, Hasher1()(i), &stash_bucket));
    }

    for (auto i = 0;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i += 2) {
      uint32_t value;
      assert(bucket.Find(i, &value, Hasher1()(i), &stash_bucket));
      if (i % 2 == 0) {
        assert(value == i*2);
      } else {
        assert(value == i);
      }
    }

    printf("[PASSED]\n");
  }

  static void TestDleftAppend() {
    printf("[TEST DLEFT APPEND]\n");

    const int testcase_size = 60000;
    HashTableType hash_table(testcase_size);

    for (auto i = 0; i < testcase_size; i++) {
      assert(hash_table.Append(i, i, Hasher1()(i), Hasher2()(i)));
      assert(hash_table.size_ == i + 1);
    }

    printf("[PASSED]\n");
  }

  static void TestDleftErase() {
    printf("[TEST DLEFT ERASE]\n");

    const int testcase_size = 60000;
    HashTableType hash_table(testcase_size);

    for (auto i = 0; i < testcase_size; i++) {
      assert(hash_table.Append(i, i, Hasher1()(i), Hasher2()(i)));
    }
    assert(hash_table.size_ == testcase_size);

    for (auto i = 0; i < testcase_size; i += 2) {
      assert(hash_table.Erase(i, Hasher1()(i), Hasher2()(i)));
      assert(hash_table.size_ == testcase_size - i / 2 - 1);
    }

    for (auto i = 0; i < testcase_size; i += 2) {
      assert(!hash_table.Erase(i, Hasher1()(i), Hasher2()(i)));
      assert(hash_table.size_ == testcase_size / 2);
    }

    printf("[PASSED]\n");
  }

  static void TestDleftFind() {
    printf("[TEST DLEFT FIND]\n");

    const int testcase_size = 60000;
    HashTableType hash_table(testcase_size);

    for (auto i = 0; i < testcase_size; i++) {
      assert(hash_table.Append(i, i, Hasher1()(i), Hasher2()(i)));
    }

    for (auto i = 0; i < testcase_size; i++) {
      uint32_t value;
      assert(hash_table.Find(i, &value, Hasher1()(i), Hasher2()(i)));
      assert(value == i);
    }

    for (auto i = 0; i < testcase_size; i += 2) {
      assert(hash_table.Erase(i, Hasher1()(i), Hasher2()(i)));
      assert(hash_table.Append(i, i*2, Hasher1()(i), Hasher2()(i)));
    }

    for (auto i = 0; i < testcase_size; i++) {
      uint32_t value;
      assert(hash_table.Find(i, &value, Hasher1()(i), Hasher2()(i)));
      if (i % 2 == 0) {
        assert(value == i*2);
      } else {
        assert(value == i);
      }
    }

    printf("[PASSED]\n");
  }

  static void TestDleftInsert() {
    printf("[TEST DLEFT INSERT]\n");

    const int testcase_size = 60000;
    HashTableType hash_table(testcase_size);

    for (auto i = 0; i < testcase_size; i++) {
      assert(hash_table.Append(i, i, Hasher1()(i), Hasher2()(i)));
    }

    for (auto i = 0; i < testcase_size; i += 2) {
      assert(hash_table.Insert(i, i*2, Hasher1()(i), Hasher2()(i)) == HashTableType::InsertStatus::EXISTED);
    }

    for (auto i = 0; i < testcase_size; i++) {
      uint32_t value;
      assert(hash_table.Find(i, &value, Hasher1()(i), Hasher2()(i)));
      if (i % 2 == 0) {
        assert(value == i*2);
      } else {
        assert(value == i);
      }
    }

    printf("[PASSED]\n");
  }

  static void TestDleftResize() {
    printf("[TEST DLEFT RESIZE]\n");

    const int testcase_size = 60000;
    HashTableType hash_table(testcase_size);

    for (auto i = 0; i < testcase_size; i++) {
      assert(hash_table.Append(i, i, Hasher1()(i), Hasher2()(i)));
    }

    for (auto i = 0; i < testcase_size; i++) {
      uint32_t value;
      assert(hash_table.Find(i, &value, Hasher1()(i), Hasher2()(i)));
      assert(value == i);
    }

    assert(hash_table.Resize(testcase_size * 2));

    for (auto i = 0; i < testcase_size; i++) {
      uint32_t value;
      assert(hash_table.Find(i, &value, Hasher1()(i), Hasher2()(i)));
      assert(value == i);
    }

    printf("[PASSED]\n");
  }
#endif
};

int main() {
  HashTableTest::RunAllTests();
}