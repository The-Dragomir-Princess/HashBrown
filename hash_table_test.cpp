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

  static void TestBucket() {
    TestBucketInsert();
    TestBucketErase();
    TestBucketFind();

    TestBucketInsertWithOverflow();
    TestBucketEraseWithOverflow();
    TestBucketFindWithOverflow();
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
      assert(!bucket.EraseMinorOverflow(i+1, pos[i]));
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
  }

  static void TestStashBucketInsertMajorOverflow() {
    {
      HashTableType::StashBucket bucket;
      for (auto i = 0; i < bucket.max_major_overflows; i++) {
        assert(bucket.InsertMajorOverflow(i, i, Hasher1()(i)));
      }
      assert(!bucket.InsertMajorOverflow(2023u, 2023u, static_cast<uint16_t>(Hasher1()(2023u))));
    }
    {
      HashTableType::StashBucket bucket;

      for (auto i = 0; i < bucket.bucket_capacity - bucket.max_major_overflows / 2; i++) {
        assert(bucket.InsertMinorOverflow(i, i) != bucket.invalid_pos);
      }
      for (auto i = bucket.bucket_capacity - bucket.max_major_overflows / 2; i < bucket.bucket_capacity; i++) {
        assert(bucket.InsertMajorOverflow(i, i, Hasher1()(i)));
      }
      assert(!bucket.InsertMajorOverflow(2023u, 2023u, static_cast<uint16_t>(Hasher1()(2023u))));
    }
  }

  static void TestStashBucketEraseMajorOverflow() {
    HashTableType::StashBucket bucket;

    for (auto i = 0; i < bucket.max_major_overflows; i++) {
      assert(bucket.InsertMajorOverflow(i, i, Hasher1()(i)));
    }

    for (auto i = 0; i < bucket.max_major_overflows; i += 2) {
      assert(bucket.EraseMajorOverflow(i, Hasher1()(i)));
      assert(!bucket.EraseMajorOverflow(i, Hasher1()(i)));
    }

    for (auto i = 0; i < bucket.max_major_overflows; i += 2) {
      assert(bucket.InsertMajorOverflow(i, i, Hasher1()(i)));
    }
  }

  static void TestStashBucketFindMajorOverflow() {
    HashTableType::StashBucket bucket;

    for (auto i = 0; i < bucket.max_major_overflows; i++) {
      assert(bucket.InsertMajorOverflow(i, i, Hasher1()(i)));
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
      assert(bucket.InsertMajorOverflow(i, i*2, Hasher1()(i)));
      assert(bucket.FindMajorOverflow(i, &value, Hasher1()(i)));
      assert(value == i*2);
    }
  }

  static void TestBucketInsert() {
    HashTableType::Bucket bucket;

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Insert(i, i, Hasher1()(i), nullptr));
    }
    assert(!bucket.Insert(2023u, 2023u, Hasher1()(2023u), nullptr));
  }

  static void TestBucketErase() {
    HashTableType::Bucket bucket;

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Insert(i, i, Hasher1()(i), nullptr));
    }

    for (auto i = 0; i < bucket.bucket_capacity; i += 2) {
      assert(bucket.Erase(i, Hasher1()(i), nullptr));
      assert(!bucket.Erase(i, Hasher1()(i), nullptr));
    }

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      if (i % 2 == 0) {
        assert(bucket.Insert(i, i*2, Hasher1()(i), nullptr));
      } else {
        assert(!bucket.Insert(i, i*2, Hasher1()(i), nullptr));
      }
    }
  }

  static void TestBucketFind() {
    HashTableType::Bucket bucket;

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Insert(i, i, Hasher1()(i), nullptr));
    }

    for (auto i = 0; i < bucket.bucket_capacity; i += 2) {
      uint32_t value;
      assert(bucket.Find(i, &value, Hasher1()(i), nullptr));
      assert(value == i);
      assert(bucket.Erase(i, Hasher1()(i), nullptr));
      assert(!bucket.Find(i, nullptr, Hasher1()(i), nullptr));
      assert(bucket.Insert(i, i*2, Hasher1()(i), nullptr));
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
  }

  static void TestBucketInsertWithOverflow() {
    HashTableType::Bucket bucket;
    HashTableType::StashBucket stash_bucket;
    auto size = bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;

    for (auto i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Insert(i, i, Hasher1()(i), &stash_bucket));
    }
    assert(bucket.overflow_count_ == 0);

    for (auto i = bucket.bucket_capacity; i < bucket.bucket_capacity + bucket.max_minor_overflows; i++) {
      assert(bucket.Insert(i, i, Hasher1()(i), &stash_bucket));
    }
    assert(bucket.overflow_count_ == bucket.max_minor_overflows);
    assert(bucket.GetMinorOverflowCount() == bucket.max_minor_overflows);
    for (auto i = 0; i < stash_bucket.max_major_overflows; i++) {
      assert(stash_bucket.position_[i] == stash_bucket.invalid_pos);
    }

    for (auto i = bucket.bucket_capacity + bucket.max_minor_overflows;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i++) {
      assert(bucket.Insert(i, i, Hasher1()(i), &stash_bucket));
    }
    assert(bucket.overflow_count_ == bucket.max_minor_overflows + stash_bucket.max_major_overflows);
    assert(bucket.GetMinorOverflowCount() == bucket.max_minor_overflows);
    for (auto i = 0; i < stash_bucket.max_major_overflows; i++) {
      assert(stash_bucket.position_[i] != stash_bucket.invalid_pos);
    }
  
    assert(!bucket.Insert(2023u, 2023u, Hasher1()(2023u), &stash_bucket));
  }

  static void TestBucketEraseWithOverflow() {
    HashTableType::Bucket bucket;
    HashTableType::StashBucket stash_bucket;

    for (auto i = 0;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i++) {
      assert(bucket.Insert(i, i, Hasher1()(i), &stash_bucket));
      assert(!bucket.Erase(i, Hasher1()(i), &stash_bucket));
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
  }

  static void TestBucketFindWithOverflow() {
    HashTableType::Bucket bucket;
    HashTableType::StashBucket stash_bucket;

    for (auto i = 0;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i++) {
      assert(bucket.Insert(i, i, Hasher1()(i), &stash_bucket));
      assert(!bucket.Erase(i, Hasher1()(i), &stash_bucket));
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

      assert(bucket.Insert(i, i*2, Hasher1()(i), &stash_bucket));
      assert(bucket.Find(i, &value, Hasher1()(i), &stash_bucket));
      assert(value == i*2);
    }
  }
};

int main() {
  HashTableTest::RunAllTests();
}