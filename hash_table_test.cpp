#include "dleft_fp_stash.hpp"
#include "xxhash.h"
#include <stdint.h>

#include <iostream>

#include <cassert>

const uint64_t seed1 = 0x1234abcd;
const uint64_t seed2 = 0xabcd4321;

template<class KeyType, uint64_t seed>
class Hasher {
 public:
  auto operator()(const KeyType &key) -> uint32_t {
    return XXH32(&key, sizeof(KeyType), seed);
  }
};

using Hasher1 = Hasher<uint32_t, seed1>;
using Hasher2 = Hasher<uint32_t, seed2>;
using HashTableType = DleftFpStash<uint32_t, uint32_t, Hasher1, Hasher2>;

class HashTableTest {
 public:
  static void RunAllTests() {
    TestStashBucket();
  }

 private:
  static void TestStashBucket() {
    TestStashBucketInsertMinorOverflow();
    TestStashBucketEraseMinorOverflow();
    TestStashBucketFindMinorOverflow();

    TestStashBucketInsertMajorOverflow();
    TestStashBucketEraseMajorOverflow();
    TestStashBucketFindMajorOverflow();
  }

  static void TestStashBucketInsertMinorOverflow() {
    HashTableType::StashBucket bucket;
    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.InsertMinorOverflow(i, i) != bucket.invalid_pos);
    }
    assert(bucket.InsertMinorOverflow(2023u, 2023u) == bucket.invalid_pos);
  }

  static void TestStashBucketEraseMinorOverflow() {
    HashTableType::StashBucket bucket;
    uint8_t pos[bucket.bucket_capacity];
    
    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      pos[i] = bucket.InsertMinorOverflow(i, i);
      assert(pos[i] != bucket.invalid_pos);
    }

    for (auto i = 0; i < bucket.bucket_capacity; i += 2) {
      assert(bucket.EraseMinorOverflow(i, pos[i]));
    }

    for (auto i = 0; i < bucket.bucket_capacity; i += 2) {
      assert(bucket.InsertMinorOverflow(i, i) != bucket.invalid_pos);
    }
  }

  static void TestStashBucketFindMinorOverflow() {
    HashTableType::StashBucket bucket;
    uint8_t pos[bucket.bucket_capacity];

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      pos[i] = bucket.InsertMinorOverflow(i, i);
      assert(pos[i] != bucket.invalid_pos);
    }

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      uint32_t value;
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
  }

  static void TestStashBucketInsertMajorOverflow() {
    {
      HashTableType::StashBucket bucket;
      for (auto i = 0; i < bucket.major_overflow_capacity; i++) {
        assert(bucket.InsertMajorOverflow(i, i, static_cast<uint16_t>(Hasher1()(i))));
      }
      assert(!bucket.InsertMajorOverflow(2023u, 2023u, static_cast<uint16_t>(Hasher1()(2023u))));
    }
    {
      HashTableType::StashBucket bucket;

      for (auto i = 0; i < bucket.bucket_capacity - bucket.major_overflow_capacity / 2; i++) {
        assert(bucket.InsertMinorOverflow(i, i) != bucket.invalid_pos);
      }
      for (auto i = bucket.bucket_capacity - bucket.major_overflow_capacity / 2; i < bucket.bucket_capacity; i++) {
        assert(bucket.InsertMajorOverflow(i, i, static_cast<uint16_t>(Hasher1()(i))));
      }
      assert(!bucket.InsertMajorOverflow(2023u, 2023u, static_cast<uint16_t>(Hasher1()(2023u))));
    }
  }

  static void TestStashBucketEraseMajorOverflow() {
    HashTableType::StashBucket bucket;

    for (auto i = 0; i < bucket.major_overflow_capacity; i++) {
      assert(bucket.InsertMajorOverflow(i, i, static_cast<uint16_t>(Hasher1()(i))));
    }

    for (auto i = 0; i < bucket.major_overflow_capacity; i += 2) {
      assert(bucket.EraseMajorOverflow(i, static_cast<uint16_t>(Hasher1()(i))));
      assert(!bucket.EraseMajorOverflow(i, static_cast<uint16_t>(Hasher1()(i))));
    }

    for (auto i = 0; i < bucket.major_overflow_capacity; i += 2) {
      assert(bucket.InsertMajorOverflow(i, i, static_cast<uint16_t>(Hasher1()(i))));
    }
  }

  static void TestStashBucketFindMajorOverflow() {
    HashTableType::StashBucket bucket;

    for (auto i = 0; i < bucket.major_overflow_capacity; i++) {
      assert(bucket.InsertMajorOverflow(i, i, static_cast<uint16_t>(Hasher1()(i))));
    }
    for (auto i = 0; i < bucket.major_overflow_capacity; i++) {
      uint32_t value;
      assert(bucket.FindMajorOverflow(i, &value, static_cast<uint16_t>(Hasher1()(i))));
      assert(value == i);
    }

    for (auto i = 0; i < bucket.major_overflow_capacity; i += 2) {
      assert(bucket.EraseMajorOverflow(i, static_cast<uint16_t>(Hasher1()(i))));
      assert(!bucket.FindMajorOverflow(i, nullptr, static_cast<uint16_t>(Hasher1()(i))));
    }

    for (auto i = bucket.major_overflow_capacity - 2; i >= 0; i -= 2) {
      uint32_t value;
      assert(bucket.InsertMajorOverflow(i, i*2, static_cast<uint16_t>(Hasher1()(i))));
      assert(bucket.FindMajorOverflow(i, &value, static_cast<uint16_t>(Hasher1()(i))));
      assert(value == i*2);
    }
  }
};

int main() {
  HashTableTest::RunAllTests();
}