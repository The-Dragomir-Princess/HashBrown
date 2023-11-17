#pragma once

#include <stdint.h>

#include <immintrin.h>

#include <iostream>

#include <cassert>
#include <cstring>

#define __TEST_DLEFT__

#define CACHELINE_SIZE (64)

#define BUCKET_STASH_BUCKET_RATIO (128)

#define BYTE_ROUND_UP(n) (((n) + 7) / 8)
#define ROUNDUP_POWER_2(n) ((n) == 0 ? 1 : ((n) ^ ((n) - 1)) == 0 ? (n) : 1 << (64 -__builtin_clzll(n)))

#define GET_BIT(bits, n)   (bits & (1ull << (n)))
#define SET_BIT(bits, n)   (bits |= (1ull << (n)))
#define CLEAR_BIT(bits, n) (bits &= ~(1ull << (n)))

#define FINGERPRINT8(hash)  (static_cast<uint8_t>(hash))
#define FINGERPRINT16(hash) (static_cast<uint16_t>(hash))
#define BUCKET_IDX(hash)  ((hash) >> 16)

#define LIKELY(foo)   foo
#define UNLIKELY(foo) foo

#define SEARCH_8_128(val, src, mask) \
	do { \
		__m128i val_vec = _mm_set1_epi8(static_cast<uint8_t>(val)); \
		__m128i src_vec  = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src)); \
		__m128i result = _mm_cmpeq_epi8(val_vec, src_vec); \
		mask = _mm_movemask_epi8(result); \
	} while (0)

#define SEARCH_16_128(val, src, mask) \
	do { \
		__m128i val_vec = _mm_set1_epi16(static_cast<uint16_t>(val)); \
		__m128i src_vec  = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src)); \
		__m128i result = _mm_cmpeq_epi16(val_vec, src_vec); \
		mask = _mm_movemask_epi8(result); \
	} while (0)

template <class KeyType, class ValueType, class HashFunc1, class HashFunc2>
class DleftFpStash {
 public:
  DleftFpStash(size_t = 0);

  auto Insert(const KeyType &, const ValueType &) -> bool;

  auto Erase(const KeyType &) -> bool;

  auto Find(const KeyType &, ValueType *) const -> bool;

  void Resize(size_t);

 private:
  using Tuple = struct {
    KeyType key;
    ValueType value;
  } __attribute__((packed));

	static constexpr size_t tuple_size = sizeof(Tuple);

	struct Bucket;
	struct StashBucket;

  struct Bucket {
    static constexpr size_t header_size = 32;
    static constexpr int bucket_capacity = 16;
    static constexpr int max_minor_overflows = 4;
    static constexpr size_t buf_size = CACHELINE_SIZE - header_size;
    static constexpr int buf_capacity = buf_size / tuple_size;

    // header
    uint8_t fingerprints_[bucket_capacity];
    uint16_t validity_{0};
		uint8_t overflow_count_{0};
		uint8_t overflow_info_{0};
		uint16_t overflow_fp_[max_minor_overflows];
		uint8_t overflow_pos_[max_minor_overflows];

		// TODO: write buffer optimization
    // // write buffer (in the same cacheline as header)
    // Tuple buf_[buf_capacity];
    // uint8_t dummy_[buf_size - sizeof(buf_)];

    // // key-value pairs
    // Tuple tuples_[bucket_capacity - buf_capacity];

		Tuple tuples_[bucket_capacity];

		Bucket() = default;

		auto Insert(const KeyType &, const ValueType &, uint32_t, StashBucket *) -> bool;

		auto Erase(const KeyType &, uint32_t, StashBucket *) -> bool;

		auto Find(const KeyType &, ValueType *, uint32_t, const StashBucket *) const -> bool;

		auto GetSize() const -> uint8_t { return __builtin_popcount(validity_); }

		auto GetTotal() const -> uint8_t { return GetSize() + overflow_count_; }

		auto GetStashBucketNum() const -> uint8_t { return overflow_info_ >> 6; }

		void SetStashBucketNum(uint8_t num) { overflow_info_ = (num << 6) | GetMinorOverflowValidity(); } 

		auto GetMinorOverflowValidity() const -> uint8_t { return overflow_info_ & 0xf; }

		auto GetMinorOverflowCount() const -> uint8_t { return __builtin_popcount(GetMinorOverflowValidity()); }
  } __attribute__((packed));

  struct StashBucket {
    static constexpr size_t header_size = 32;
    static constexpr int max_major_overflows = 8;
    static constexpr int bucket_capacity = 64;
		static constexpr uint8_t invalid_pos = 0xff;

    // header
    uint16_t fingerprints_[max_major_overflows];
		uint8_t position_[max_major_overflows];
    uint64_t validity_{0};

		// overflow key-value pairs
    Tuple tuples_[bucket_capacity];

		StashBucket() { memset(position_, invalid_pos, sizeof(position_)); };

    auto InsertMajorOverflow(const KeyType &, const ValueType &, uint32_t) -> bool;

    auto EraseMajorOverflow(const KeyType &, uint32_t) -> bool;

    auto FindMajorOverflow(const KeyType &, ValueType *, uint32_t) const -> bool;

    auto InsertMinorOverflow(const KeyType &, const ValueType &) -> uint8_t;

    auto EraseMinorOverflow(const KeyType &, uint8_t) -> bool;

    auto FindMinorOverflow(const KeyType &, ValueType *, uint8_t) const -> bool;

		auto GetSize() const -> uint8_t { return __builtin_popcountll(validity_); };
  } __attribute__((packed));

	size_t num_buckets_;

	size_t num_stash_buckets_;

	Bucket *buckets_;

	StashBucket *stash_buckets_;

#ifdef __TEST_DLEFT__
	friend class HashTableTest;
#endif
};

#define DLEFT_TEMPLATE template <class KeyType, class ValueType, class HashFunc1, class HashFunc2>
#define DLEFT_TYPE DleftFpStash<KeyType, ValueType, HashFunc1, HashFunc2>

DLEFT_TEMPLATE
DLEFT_TYPE::DleftFpStash(size_t num_buckets)
		: num_buckets_(ROUNDUP_POWER_2(num_buckets)),
			num_stash_buckets_(num_buckets_ / BUCKET_STASH_BUCKET_RATIO) {
	if (num_buckets_ > (1 << 16)) {
		printf("error: table is too large\n");
		exit(1);
	}

	buckets_ = new Bucket[num_buckets_];
	assert(buckets_ != nullptr);

	if (num_stash_buckets_ == 0) {
		stash_buckets_ = nullptr;
	} else {
		stash_buckets_ = new StashBucket[num_stash_buckets_];
	}
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::Insert(const KeyType &key, const ValueType &value) -> bool {
	uint32_t hash1 = HashFunc1()(key), hash2 = HashFunc2()(key), hash;
	uint16_t idx1 = BUCKET_IDX(hash1) & (num_buckets_ - 1);
	uint16_t idx2 = BUCKET_IDX(hash2) & (num_buckets_ - 1);
	uint16_t min_idx;
	Bucket *min_bucket;
	StashBucket *stash_bucket;

	if (buckets_[idx1].GetTotal() <= buckets_[idx2].GetTotal()) {
		min_idx = idx1;
		hash = hash1;
	} else {
		min_idx = idx2;
		hash = hash2;
	}
	min_bucket = &buckets_[min_idx];

	if (min_bucket->overflow_count_ == 0) {
		if (min_bucket->Insert(key, value, hash, nullptr)) {
			return true;
		}
		if (stash_buckets_ == nullptr) {
			return false;
		}

		uint8_t stash_base = min_idx / BUCKET_STASH_BUCKET_RATIO, min_stash_num;
		uint8_t min_stash_size = 0xff;
		for (uint8_t stash_num = 0; stash_num < 4; stash_num++) {
			if (stash_base + stash_num >= num_stash_buckets_) {
				break;
			}
			if (stash_buckets_[stash_base + stash_num].GetSize() < min_stash_size) {
				min_stash_num = stash_base + stash_num;
				min_stash_size = stash_buckets_[min_stash_num].GetSize();
			}
		}
		min_bucket->SetStashBucketNum(min_stash_num);
	}

	assert(stash_buckets_ != nullptr);
	stash_bucket = &stash_buckets_[min_idx / BUCKET_STASH_BUCKET_RATIO + min_bucket->GetStashBucketNum()];
	return min_bucket->Insert(key, value, hash, stash_bucket);
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::Erase(const KeyType &key) -> bool {
	uint32_t hash1 = HashFunc1()(key), hash2 = HashFunc2()(key), hash;
	uint16_t idx1 = BUCKET_IDX(hash1) & (num_buckets_ - 1);
	uint16_t idx2 = BUCKET_IDX(hash2) & (num_buckets_ - 1);
	Bucket *bucket1 = &buckets_[idx1], *bucket2 = &buckets_[idx2];
	StashBucket *stash_bucket1{nullptr}, *stash_bucket2{nullptr};

	if (bucket1->overflow_count_ > 0 && stash_buckets_ != nullptr) {
		stash_bucket1 = &stash_buckets_[idx1 / BUCKET_STASH_BUCKET_RATIO + bucket1->GetStashBucketNum()];
	}
	if (bucket1->Erase(key, hash1, stash_bucket1)) {
		return true;
	} else if (idx1 == idx2) {
		return false;
	}

	if (bucket2->overflow_count_ > 0 && stash_buckets_ != nullptr) {
		stash_bucket2 = &stash_buckets_[idx2 / BUCKET_STASH_BUCKET_RATIO + bucket1->GetStashBucketNum()];
	}
	return bucket2->Erase(key, hash2, stash_bucket2);
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::Find(const KeyType &key, ValueType *value) const -> bool {
	// TODO: parallelize the probing of two buckets
	uint32_t hash1 = HashFunc1()(key), hash2 = HashFunc2()(key), hash;
	uint16_t idx1 = BUCKET_IDX(hash1) & (num_buckets_ - 1);
	uint16_t idx2 = BUCKET_IDX(hash2) & (num_buckets_ - 1);
	Bucket *bucket1 = &buckets_[idx1], *bucket2 = &buckets_[idx2];
	StashBucket *stash_bucket1{nullptr}, *stash_bucket2{nullptr};

	if (bucket1->overflow_count_ > 0 && stash_buckets_ != nullptr) {
		stash_bucket1 = &stash_buckets_[idx1 / BUCKET_STASH_BUCKET_RATIO + bucket1->GetStashBucketNum()];
	}
	if (bucket1->Find(key, value, hash1, stash_bucket1)) {
		return true;
	} else if (idx1 == idx2) {
		return false;
	}

	if (bucket2->overflow_count_ > 0 && stash_buckets_ != nullptr) {
		stash_bucket2 = &stash_buckets_[idx2 / BUCKET_STASH_BUCKET_RATIO + bucket1->GetStashBucketNum()];
	}
	return bucket2->Find(key, value, hash2, stash_bucket2);
}

DLEFT_TEMPLATE
void DLEFT_TYPE::Resize(size_t new_size) {
	size_t old_num_buckets = num_buckets_;
	size_t old_num_stash_buckets = num_stash_buckets_;
	Bucket *old_buckets = buckets_;
	StashBucket *old_stash_buckets = stash_buckets_;
	
	if ((num_buckets_ = ROUNDUP_POWER_2(new_size)) > (1 << 16)) {
		printf("error: table is too large\n");
		exit(1);
	}
	num_stash_buckets_ = num_buckets_ / BUCKET_STASH_BUCKET_RATIO;
	buckets_ = new Bucket[num_buckets_];
	assert(buckets_ != nullptr);
	stash_buckets_ = (num_stash_buckets_ > 0 ? new Bucket[num_stash_buckets_] : nullptr);

	for (size_t i = 0; i < old_num_buckets; i++) {
		for (auto j = 0; j < Bucket::bucket_capacity; j++) {
			if (!GET_BIT(old_buckets[i].validity_, j)) {
				continue;
			}
			if (!Insert(old_buckets[i].tuples_[j].key, old_buckets[i].tuples_[j].value)) {
				goto resize_failed;
			}
		}
	}
	for (size_t i = 0; i < old_num_stash_buckets; i++) {
		for (auto j = 0; j < StashBucket::bucket_capacity; j++) {
			if (!GET_BIT(old_stash_buckets[i].validity_, j)) {
				continue;
			}
			if (!Insert(old_stash_buckets[i].tuples_[j].key, old_stash_buckets[i].tuples_[j].value)) {
				goto resize_failed;
			}
		}
	}

	delete old_buckets;
	delete old_num_buckets;
	
	return;

 resize_failed:
 	delete buckets_;
	delete stash_buckets_;

	num_buckets_ = old_num_buckets;
	num_stash_buckets_ = old_num_stash_buckets;
	buckets_ = old_buckets;
	stash_buckets_ = old_stash_buckets;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::Bucket::Insert(const KeyType &key, const ValueType &value, uint32_t hash, StashBucket *stash_bucket) -> bool {
	int mask;
	uint8_t idx, pos;

	pos = __builtin_ctz(~validity_);
	if (pos < bucket_capacity) {
		tuples_[pos] = {key, value};
		fingerprints_[pos] = FINGERPRINT8(hash);
		SET_BIT(validity_, pos);

		return true;
	}

	if (stash_bucket == nullptr) {
		return false;
	}

	if (GetMinorOverflowCount() == max_minor_overflows) {
		if (stash_bucket->InsertMajorOverflow(key, value, hash)) {
			overflow_count_++;
			return true;
		}
		return false;
	}

	pos = stash_bucket->InsertMinorOverflow(key, value);
	if (pos == StashBucket::invalid_pos) {
		return false;
	}

	idx = __builtin_ctz(~GetMinorOverflowValidity());
	overflow_count_++;
	overflow_fp_[idx] = FINGERPRINT16(hash);
	overflow_pos_[idx] = pos;
	SET_BIT(overflow_info_, idx);

	return true;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::Bucket::Erase(const KeyType &key, uint32_t hash, StashBucket *stash_bucket) -> bool {
	int mask;
	uint8_t idx, pos;

	SEARCH_8_128(FINGERPRINT8(hash), fingerprints_, mask);
	mask &= validity_;
	while (mask != 0) {
		pos = __builtin_ctz(mask);
		if (LIKELY( tuples_[pos].key == key )) {
			CLEAR_BIT(validity_, pos);
			return true;
		}
		mask &= ~(1 << pos);
	}

	if (stash_bucket == nullptr) {
		return false;
	}

	SEARCH_16_128(FINGERPRINT16(hash), overflow_fp_, mask);
	while (mask != 0) {
		idx = __builtin_ctz(mask) / 2;
		if (GET_BIT(overflow_info_, idx) && LIKELY( stash_bucket->EraseMinorOverflow(key, overflow_pos_[idx]) )) {
			CLEAR_BIT(overflow_info_, idx);
			overflow_count_--;
			return true;
		}
		mask &= ~(3 << (idx * 2));
	}

	if (UNLIKELY( overflow_count_ > GetMinorOverflowCount() )) {
		if (stash_bucket->EraseMajorOverflow(key, hash)) {
			overflow_count_--;
			return true;
		}
	}
	return false;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::Bucket::Find(const KeyType &key, ValueType *value, uint32_t hash, const StashBucket *stash_bucket) const -> bool {
	int mask;
	uint8_t idx, pos;

	SEARCH_8_128(FINGERPRINT8(hash), fingerprints_, mask);
	mask &= validity_;
	while (mask != 0) {
		pos = __builtin_ctz(mask);
		if (LIKELY( tuples_[pos].key == key )) {
			if (value != nullptr) {
				*value = tuples_[pos].value;
			}
			return true;
		}
		mask &= ~(1 << pos);
	}

	if (stash_bucket == nullptr) {
		return false;
	}

	SEARCH_16_128(FINGERPRINT16(hash), overflow_fp_, mask);
	while (mask != 0) {
		idx = __builtin_ctz(mask) / 2;
		if (GET_BIT(overflow_info_, idx) && LIKELY( stash_bucket->FindMinorOverflow(key, value, overflow_pos_[idx]) )) {
			return true;
		}
		mask &= ~(3 << (idx * 2));
	}

	if (UNLIKELY( overflow_count_ > GetMinorOverflowCount() )) {
		if (stash_bucket->FindMajorOverflow(key, value, hash)) {
			return true;
		}
	}
	return false;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::InsertMajorOverflow(const KeyType &key, const ValueType &value, uint32_t hash) -> bool {
	int mask;
	uint8_t idx, pos;

	if (~validity_ == 0) {
		return false;
	}

	SEARCH_8_128(invalid_pos, position_, mask);
	mask &= 0xff;
	if (UNLIKELY( mask == 0 )) {
		return false;
	}

	pos = __builtin_ctzll(~validity_);
	tuples_[pos] = {key, value};
	SET_BIT(validity_, pos);

	idx = __builtin_ctz(mask);
	position_[idx] = pos;
	fingerprints_[idx] = FINGERPRINT16(hash);

	return true;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::EraseMajorOverflow(const KeyType &key, uint32_t hash) -> bool {
	int mask;
	uint8_t idx;

	SEARCH_16_128(FINGERPRINT16(hash), fingerprints_, mask);
	while (mask != 0) {
		idx = __builtin_ctz(mask) / 2;
		if (position_[idx] != invalid_pos && LIKELY( tuples_[position_[idx]].key == key )) {
			CLEAR_BIT(validity_, position_[idx]);
			position_[idx] = invalid_pos;
			return true;
		}
		mask &= ~(3 << (idx * 2));
	}
	return false;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::FindMajorOverflow(const KeyType &key, ValueType *value, uint32_t hash) const -> bool {
	int mask;
	uint8_t idx;

	SEARCH_16_128(FINGERPRINT16(hash), fingerprints_, mask);
	while (mask != 0) {
		idx = __builtin_ctz(mask) / 2;
		if (position_[idx] != invalid_pos && LIKELY( tuples_[position_[idx]].key == key )) {
			if (value != nullptr) {
				*value = tuples_[position_[idx]].value;
			}
			return true;
		}
		mask &= ~(3 << (idx * 2));
	}
	return false;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::InsertMinorOverflow(const KeyType &key, const ValueType &value) -> uint8_t {
	uint8_t pos;

	if (~validity_ == 0) {
		return invalid_pos;
	}
	pos = __builtin_ctzll(~validity_);
	tuples_[pos] = {key, value};
	SET_BIT(validity_, pos);

	return pos;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::EraseMinorOverflow(const KeyType &key, uint8_t pos) -> bool {
	assert(GET_BIT(validity_, pos));
	if (LIKELY( tuples_[pos].key == key )) {
		CLEAR_BIT(validity_, pos);
		return true;
	}
	return false;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::FindMinorOverflow(const KeyType &key, ValueType *value, uint8_t pos) const -> bool {
	assert(GET_BIT(validity_, pos));
	if (LIKELY( tuples_[pos].key == key )) {
		if (value != nullptr) {
			*value = tuples_[pos].value;
		}
		return true;
	}
	return false;
}

#undef DLEFT_TYPE
#undef UNLIKELY
#undef LIKELY
#undef FINGERPRINT16
#undef FINGERPRINT8
#undef GET_BIT
#undef SET_BIT
#undef CLEAR_BIT
#undef ROUNDUP_POWER_2
#undef BYTE_ROUND_UP
#undef BUCKET_STASH_BUCKET_RATIO
#undef CACHELINE_SIZE
