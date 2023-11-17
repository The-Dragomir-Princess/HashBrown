#include "dleft_fp_stash.hpp"
#include "xxhash.h"
#include <stdint.h>

#include <iostream>

#include <cassert>

const uint64_t seed1 = 0x1234abcd;
const uint64_t seed2 = 0xcdef5678;

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
    TestBucket();
    TestDleft();
  }

 private:
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

    const int testcase_size = 65536;
    HashTableType hash_table(testcase_size / 16);

    for (auto i = 0; i < testcase_size; i++) {
      assert(hash_table.Append(i, i, Hasher1()(i), Hasher2()(i)));
    }

    printf("[PASSED]\n");
  }

  static void TestDleftErase() {
    printf("[TEST DLEFT ERASE]\n");

    const int testcase_size = 65536;
    HashTableType hash_table(testcase_size / 16);

    for (auto i = 0; i < testcase_size; i++) {
      assert(hash_table.Append(i, i, Hasher1()(i), Hasher2()(i)));
    }

    for (auto i = 0; i < testcase_size; i += 2) {
      assert(hash_table.Erase(i, Hasher1()(i), Hasher2()(i)));
    }

    for (auto i = 0; i < testcase_size; i += 2) {
      assert(!hash_table.Erase(i, Hasher1()(i), Hasher2()(i)));
    }

    printf("[PASSED]\n");
  }

  static void TestDleftFind() {
    printf("[TEST DLEFT FIND]\n");

    const int testcase_size = 65536;
    HashTableType hash_table(testcase_size / 16);

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

    const int testcase_size = 65536;
    HashTableType hash_table(testcase_size / 16);

    for (auto i = 0; i < testcase_size; i++) {
      assert(hash_table.Append(i, i, Hasher1()(i), Hasher2()(i)));
    }

    for (auto i = 0; i < testcase_size; i += 2) {
      assert(hash_table.Insert(i, i*2, Hasher1()(i), Hasher2()(i)));
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

    const int testcase_size = 65536;
    HashTableType hash_table(testcase_size / 16);

    for (auto i = 0; i < testcase_size; i++) {
      assert(hash_table.Append(i, i, Hasher1()(i), Hasher2()(i)));
    }

    for (auto i = 0; i < testcase_size; i++) {
      uint32_t value;
      assert(hash_table.Find(i, &value, Hasher1()(i), Hasher2()(i)));
      assert(value == i);
    }

    assert(hash_table.Resize(testcase_size / 8));

    for (auto i = 0; i < testcase_size; i++) {
      uint32_t value;
      assert(hash_table.Find(i, &value, Hasher1()(i), Hasher2()(i)));
      assert(value == i);
    }

    printf("[PASSED]\n");
  }
};

int main() {
  HashTableTest::RunAllTests();
}