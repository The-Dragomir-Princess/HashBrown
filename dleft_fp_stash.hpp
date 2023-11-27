#pragma once

#include <stdint.h>

#include <immintrin.h>

#include <iostream>

#include <cassert>
#include <cstring>

#define __TEST_DLEFT__

#define CACHELINE_SIZE (64)

#define BUCKET_STASH_BUCKET_RATIO (256)

#define BYTE_ROUND_UP(n) (((n) + 7) / 8)
#define ROUND_UP(n, b) (((n) + (b) - 1) / (b))
#define ROUNDUP_POWER_2(n) ((n) == 0 ? 1 : (((n) & ((n) - 1)) == 0) ? (n) : (1 << (64 -__builtin_clzll(n))))

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


template <class K, class V, class H1, class H2>
class DleftFpStash {
 public:
  DleftFpStash(size_t = 0);

	~DleftFpStash() { delete[] buckets_; delete[] stash_buckets_; }

	auto insert(K &&key, V &&value) -> bool {
		uint32_t hash1 = H1()(key), hash2 = H2()(key);
		InsertStatus status = Insert<false>(std::forward<K>(key), std::forward<V>(value), hash1, hash2);
		while (status == InsertStatus::FAILED) {
			Resize(capacity() * 2);
			status = Insert<false>(std::forward<K>(key), std::forward<V>(value), hash1, hash2);
		}
		return status == InsertStatus::INSERTED;
	}

	auto erase(const K &key) -> bool { return Erase(key, H1()(key), H2()(key)); }

	auto find(const K &key, V &value) const -> bool { return Find(key, &value, H1()(key), H2()(key)); }

	void clear() {
		for (size_t i = 0; i < num_buckets_; i++) {
			buckets_[i].Clear();
		}
		for (size_t i = 0; i < num_stash_buckets_; i++) {
			stash_buckets_[i].Clear();
		}
	}

	void reserve(size_t size) { Resize(size); }

	auto load_factor() const -> double { return 1.0 * size_ / capacity(); }

	auto capacity() const -> size_t { return Bucket::bucket_capacity * num_buckets_ + StashBucket::bucket_capacity * num_stash_buckets_; }

 private:
  using Tuple = struct {
    K key;
    V value;
  };

	static constexpr size_t tuple_size = sizeof(Tuple);

	struct Bucket;
	struct StashBucket;

  struct Bucket {
    static constexpr size_t header_size = 32;
    static constexpr int bucket_capacity = 16;
    static constexpr int max_minor_overflows = 4;
    static constexpr size_t buf_size = CACHELINE_SIZE - header_size;
    static constexpr int buf_capacity = buf_size / tuple_size;

    // header (32 bytes)
    uint8_t fingerprints_[bucket_capacity];      // fingerprints for each in-bucket key
    uint16_t validity_{0};                       // validity bitmap for each in-bucket key
		uint8_t overflow_count_{0};                  // the total number of overflows
		uint8_t overflow_info_{0};                   // the 2 highest bits specify the stash bucket;
		                                             // the 4 lowest bits serve as the validity bits for minor overflows
		uint16_t overflow_fp_[max_minor_overflows];  // fingerprints for minor overflows
		uint8_t overflow_pos_[max_minor_overflows];  // positions of minor overflows in the stash bucket

		// TODO: write buffer optimization
    // // write buffer (in the same cacheline as header)
    // Tuple buf_[buf_capacity];
    // uint8_t dummy_[buf_size - sizeof(buf_)];

    // // key-value pairs
    // Tuple tuples_[bucket_capacity - buf_capacity];

		// key-value pairs
		Tuple tuples_[bucket_capacity];

		enum class TupleStatus { IN_BUCKET, MINOR_OVERFLOW, MAJOR_OVERFLOW, NOT_FOUND };

		Bucket() = default;

		// Inserts a key, overwriting duplicates
		auto Insert(K &&, V &&, uint32_t, StashBucket *) -> bool;

		// Inserts a key without duplicate checks
		auto Append(K &&, V &&, uint32_t, StashBucket *) -> bool;

		// Removes a key
		auto Erase(const K &, uint32_t, StashBucket *) -> bool;

		// Looks for a key and returns the associated value
		auto Find(const K &, V *, uint32_t, const StashBucket *) const -> bool;

		// Searches for a key and returns its position
		// `status` == IN_BUCKET: returns the key's position in bucket
		// `status` == MINOR_OVERFLOW: returns the index of the key's `overflow_fp_` and `overflow_pos_`
		// `status` == MAJOR_OVERFLOW: returns the index of the key's `fingerprints_` and `position_` in stash bucket
		auto FindPos(const K &, uint32_t, const StashBucket *, TupleStatus &) const -> uint8_t;

		void Clear() {
			validity_ = 0; overflow_count_ = 0; overflow_info_ = 0;
			memset(overflow_pos_, StashBucket::invalid_pos, sizeof(overflow_pos_));
		}

		// Get the number of valid keys in bucket (not including overflows)
		auto GetSize() const -> uint8_t { return __builtin_popcount(validity_); }

		// Get the total number of valid keys & overflow keys in bucket
		auto GetTotal() const -> uint8_t { return GetSize() + overflow_count_; }

		// Get its stash bucket's number (each bucket has at most four candidate stash buckets)
		auto GetStashBucketNum() const -> uint8_t { return overflow_info_ >> 6; }

		// Bind the bucket to a stash bucket; once bound, it cannot be unbounded unless the number of overflows becomes 0
		void SetStashBucketNum(uint8_t num) { overflow_info_ = (num << 6) | GetMinorOverflowValidity(); }

		auto GetStashBucketIndex(size_t idx, size_t max) const -> size_t {
			return (idx / BUCKET_STASH_BUCKET_RATIO + GetStashBucketNum()) & (max - 1);
		}

		// Get the validity bitmap for its moinor overflows
		auto GetMinorOverflowValidity() const -> uint8_t { return overflow_info_ & 0xf; }

		// Get the number of minor overflows
		auto GetMinorOverflowCount() const -> uint8_t { return __builtin_popcount(GetMinorOverflowValidity()); }
  };

  struct StashBucket {
    static constexpr size_t header_size = 32;
    static constexpr int max_major_overflows = 8;
    static constexpr int bucket_capacity = 64;
		static constexpr uint8_t invalid_pos = 0xff;

    // header
    uint16_t fingerprints_[max_major_overflows];  // fingerprints for major overflows
		uint8_t position_[max_major_overflows];       // positions of each major overflow in stash bucket
    uint64_t validity_{0};                        // validity bitmap for each overflow key in stash bucket

		// key-value pairs
    Tuple tuples_[bucket_capacity];

		StashBucket() { memset(position_, invalid_pos, sizeof(position_)); }

		// Inserts a major overflow, overwriting duplicates
    auto InsertMajorOverflow(K &&, V &&, uint32_t) -> bool;

		// Inserts a major overflow without checking duplicates
		auto AppendMajorOverflow(K &&, V &&, uint32_t) -> bool;

		// Removes a major overflow key
    auto EraseMajorOverflow(const K &, uint32_t) -> bool;

		// Searches for a major overflow key and returns its associated value
    auto FindMajorOverflow(const K &, V *, uint32_t) const -> bool;

		// Searches for a major overflow key and returns its index of `fingerprints_` and `position_`
		auto FindMajorOverflowIdx(const K &, uint32_t) const -> uint8_t;

    auto InsertMinorOverflow(K &&, V &&) -> uint8_t;

    auto EraseMinorOverflow(const K &, uint8_t) -> bool;

    auto FindMinorOverflow(const K &, V *, uint8_t) const -> bool;

		void Clear() { validity_ = 0; memset(position_, invalid_pos, sizeof(position_)); }

		// Returns the number of valid overflow keys in bucket (either major or minor)
		auto GetSize() const -> uint8_t { return __builtin_popcountll(validity_); };
  };

	enum class InsertStatus { INSERTED, EXISTED, FAILED };

	// Inserts a key into the hash table; If a duplicate is found, the value is overwritten
	// Returns `INSERTED` if insertion was successful, `EXISTED` if a duplicate key is found,
	// and `FAILED` if the insertion failed (e.g. when running out of space)
	// template argument `upsert` defines whether to overwrite duplicates
	template<bool upsert = true>
  auto Insert(K &&, V &&, uint32_t, uint32_t) -> InsertStatus;

	// Inserts a key into the hash table without duplicate checks
	// Returns `true` if insertion is successful and `false` otherwise (e.g. when running out of space)
	auto Append(K &&, V &&, uint32_t, uint32_t) -> bool;

	// Removes a key from the hash table
	// Returns `true` if found and `false` otherwise
  auto Erase(const K &, uint32_t, uint32_t) -> bool;

	// Searches for a key from the hash table
	// Returns `true` if found and `false` otherwise; value is stored in the second argument
  auto Find(const K &, V *, uint32_t, uint32_t) const -> bool;

	// Resize the table; may fail if the new size is smaller than current size
	// Returns `true` if resize is successful and false otherwise
  auto Resize(size_t) -> bool;

	size_t num_buckets_;

	size_t num_stash_buckets_;

	size_t size_{0};

	Bucket *buckets_;

	StashBucket *stash_buckets_;

#ifdef __TEST_DLEFT__
	friend class HashTableTest;
#endif
};

#define DLEFT_TEMPLATE template <class K, class V, class H1, class H2>
#define DLEFT_TYPE DleftFpStash<K, V, H1, H2>

DLEFT_TEMPLATE
DLEFT_TYPE::DleftFpStash(size_t size)
		: num_buckets_(ROUNDUP_POWER_2(ROUND_UP(size, Bucket::bucket_capacity))),
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
template<bool upsert>
auto DLEFT_TYPE::Insert(K &&key, V &&value, uint32_t hash1, uint32_t hash2) -> InsertStatus {
	using TupleStatus = typename Bucket::TupleStatus;
	
	uint16_t idx1 = BUCKET_IDX(hash1) & (num_buckets_ - 1);
	uint16_t idx2 = BUCKET_IDX(hash2) & (num_buckets_ - 1);
	Bucket *bucket1 = &buckets_[idx1], *bucket2 = &buckets_[idx2];
	StashBucket *stash_bucket1{nullptr}, *stash_bucket2{nullptr};
	TupleStatus status;
	uint8_t pos;

	if (stash_buckets_ != nullptr && bucket1->overflow_count_ > 0) {
		stash_bucket1 = &stash_buckets_[bucket1->GetStashBucketIndex(idx1, num_stash_buckets_)];
	}
	if (stash_buckets_ != nullptr && bucket2->overflow_count_ > 0) {
		stash_bucket2 = &stash_buckets_[bucket2->GetStashBucketIndex(idx2, num_stash_buckets_)];
	}

	// Check for duplicates
	pos = bucket1->FindPos(key, hash1, stash_bucket1, status);
	if (status == TupleStatus::IN_BUCKET) {
		if (upsert) {
			bucket1->tuples_[pos].value = value;
		}
		return InsertStatus::EXISTED;
	} else if (status == TupleStatus::MINOR_OVERFLOW) {
		if (upsert) {
			stash_bucket1->tuples_[bucket1->overflow_pos_[pos]].value = value;
		}
		return InsertStatus::EXISTED;
	} else if (UNLIKELY( status == TupleStatus::MAJOR_OVERFLOW )) {
		if (upsert) {
			stash_bucket1->tuples_[stash_bucket1->position_[pos]].value = value;
		}
		return InsertStatus::EXISTED;
	}

	pos = bucket2->FindPos(key, hash2, stash_bucket2, status);
	if (status == TupleStatus::IN_BUCKET) {
		if (upsert) {
			bucket2->tuples_[pos].value = value;
		}
		return InsertStatus::EXISTED;
	} else if (status == TupleStatus::MINOR_OVERFLOW) {
		if (upsert) {
			stash_bucket2->tuples_[bucket2->overflow_pos_[pos]].value = value;
		}
		return InsertStatus::EXISTED;
	} else if (UNLIKELY( status == TupleStatus::MAJOR_OVERFLOW )) {
		if (upsert) {
			stash_bucket2->tuples_[stash_bucket2->position_[pos]].value = value;
		}
		return InsertStatus::EXISTED;
	}

	// Otherwise insert at an empty space
	return Append(std::forward<K>(key), std::forward<V>(value), hash1, hash2) ?
				 InsertStatus::INSERTED : InsertStatus::FAILED;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::Append(K &&key, V &&value, uint32_t hash1, uint32_t hash2) -> bool {
	uint32_t hash;
	uint16_t idx1 = BUCKET_IDX(hash1) & (num_buckets_ - 1);
	uint16_t idx2 = BUCKET_IDX(hash2) & (num_buckets_ - 1);
	uint16_t min_idx;
	Bucket *min_bucket;
	StashBucket *stash_bucket;

	// Insert into the more underfull candidate bucket
	if (buckets_[idx1].GetTotal() <= buckets_[idx2].GetTotal()) {
		min_idx = idx1;
		hash = hash1;
	} else {
		min_idx = idx2;
		hash = hash2;
	}
	min_bucket = &buckets_[min_idx];

	if (min_bucket->overflow_count_ == 0) {  // No stash bucket yet
		if (min_bucket->Append(std::forward<K>(key), std::forward<V>(value), hash, nullptr)) {
			size_++;
			return true;
		}  // Bucket is full; need a stash bucket
		if (stash_buckets_ == nullptr) {  // No usable stash bucket, so insertion fails
			return false;
		}

		uint8_t stash_base = min_idx / BUCKET_STASH_BUCKET_RATIO, min_stash_num;
		uint8_t min_stash_size = 0xff;
		for (uint8_t stash_num = 0; stash_num < 4; stash_num++) {  // Bind bucket to its most underfull candidate stash bucket
			uint8_t stash_idx = (stash_base + stash_num) & (num_stash_buckets_ - 1);
			uint8_t size = stash_buckets_[stash_idx].GetSize();
			if (size < min_stash_size) {
				min_stash_num = stash_num;
				min_stash_size = size;
			}
		}
		min_bucket->SetStashBucketNum(min_stash_num);
	}

	// Retry insertion with stash bucket
	assert(stash_buckets_ != nullptr);
	stash_bucket = &stash_buckets_[min_bucket->GetStashBucketIndex(min_idx, num_stash_buckets_)];
	if (min_bucket->Append(std::forward<K>(key), std::forward<V>(value), hash, stash_bucket)) {
		size_++;
		return true;
	}
	return false;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::Erase(const K &key, uint32_t hash1, uint32_t hash2) -> bool {
	uint16_t idx1 = BUCKET_IDX(hash1) & (num_buckets_ - 1);
	uint16_t idx2 = BUCKET_IDX(hash2) & (num_buckets_ - 1);
	Bucket *bucket1 = &buckets_[idx1], *bucket2 = &buckets_[idx2];
	StashBucket *stash_bucket1{nullptr}, *stash_bucket2{nullptr};

	if (bucket1->overflow_count_ > 0 && stash_buckets_ != nullptr) {
		stash_bucket1 = &stash_buckets_[bucket1->GetStashBucketIndex(idx1, num_stash_buckets_)];
	}
	if (bucket1->Erase(key, hash1, stash_bucket1)) {  // Try remove from the first bucket
		size_--;
		return true;
	} else if (idx1 == idx2) {
		return false;
	}

	if (bucket2->overflow_count_ > 0 && stash_buckets_ != nullptr) {
		stash_bucket2 = &stash_buckets_[bucket2->GetStashBucketIndex(idx2, num_stash_buckets_)];
	}  // If not found, try remove from the second bucket
	if (bucket2->Erase(key, hash2, stash_bucket2)) {
		size_--;
		return true;
	}
	return false;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::Find(const K &key, V *value, uint32_t hash1, uint32_t hash2) const -> bool {
	// TODO: parallelize the probing of two buckets
	uint16_t idx1 = BUCKET_IDX(hash1) & (num_buckets_ - 1);
	uint16_t idx2 = BUCKET_IDX(hash2) & (num_buckets_ - 1);
	Bucket *bucket1 = &buckets_[idx1], *bucket2 = &buckets_[idx2];
	StashBucket *stash_bucket1{nullptr}, *stash_bucket2{nullptr};

	if (bucket1->overflow_count_ > 0 && stash_buckets_ != nullptr) {
		stash_bucket1 = &stash_buckets_[bucket1->GetStashBucketIndex(idx1, num_stash_buckets_)];
	}
	if (bucket1->Find(key, value, hash1, stash_bucket1)) {  // Search the first bucket
		return true;
	} else if (idx1 == idx2) {
		return false;
	}

	if (bucket2->overflow_count_ > 0 && stash_buckets_ != nullptr) {
		stash_bucket2 = &stash_buckets_[bucket2->GetStashBucketIndex(idx2, num_stash_buckets_)];
	}  // If not found, search the second bucket
	return bucket2->Find(key, value, hash2, stash_bucket2);
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::Resize(size_t new_size) -> bool {
	size_t old_num_buckets = num_buckets_;
	size_t old_num_stash_buckets = num_stash_buckets_;
	Bucket *old_buckets = buckets_;
	StashBucket *old_stash_buckets = stash_buckets_;
	size_t new_capacity = ROUNDUP_POWER_2(new_size / Bucket::bucket_capacity);

	if (num_buckets_ == new_capacity) {
		return true;
	}

	if ((num_buckets_ = new_capacity) > (1 << 16)) {
		printf("error: table is too large\n");
		exit(1);
	}
	num_stash_buckets_ = num_buckets_ / BUCKET_STASH_BUCKET_RATIO;
	buckets_ = new Bucket[num_buckets_];
	assert(buckets_ != nullptr);
	stash_buckets_ = (num_stash_buckets_ > 0 ? new StashBucket[num_stash_buckets_] : nullptr);

	for (size_t i = 0; i < old_num_buckets; i++) {  // Iterate over normal buckets and rehash the keys
		for (auto j = 0; j < Bucket::bucket_capacity; j++) {
			if (!GET_BIT(old_buckets[i].validity_, j)) {
				continue;
			}
			auto &key = old_buckets[i].tuples_[j].key;
			auto &value = old_buckets[i].tuples_[j].value;
			if (!Append(std::forward<K &&>(key), std::forward<V &&>(value), H1()(key), H2()(key))) {
				goto resize_failed;
			}
		}
	}
	for (size_t i = 0; i < old_num_stash_buckets; i++) {  // Iterate over stash buckets and rehash the keys
		for (auto j = 0; j < StashBucket::bucket_capacity; j++) {
			if (!GET_BIT(old_stash_buckets[i].validity_, j)) {
				continue;
			}
			auto &key = old_stash_buckets[i].tuples_[j].key;
			auto &value = old_stash_buckets[i].tuples_[j].value;
			if (!Append(std::forward<K &&>(key), std::forward<V &&>(value), H1()(key), H2()(key))) {
				goto resize_failed;
			}
		}
	}

	delete old_buckets;
	delete old_stash_buckets;
	
	return true;

 resize_failed:  // If any insertion fails, resize fails
 	delete buckets_;
	delete stash_buckets_;

	num_buckets_ = old_num_buckets;
	num_stash_buckets_ = old_num_stash_buckets;
	buckets_ = old_buckets;
	stash_buckets_ = old_stash_buckets;

	return false;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::Bucket::Insert(K &&key, V &&value, uint32_t hash, StashBucket *stash_bucket) -> bool {
	TupleStatus status;
	int mask;
	uint8_t idx, pos;

	pos = FindPos(key, hash, stash_bucket, status);
	switch (status) {  // If found a duplicate, overwrite it
	 case TupleStatus::IN_BUCKET:
		tuples_[pos].value = value;
		return true;

	 case TupleStatus::MINOR_OVERFLOW:
	 	stash_bucket->tuples_[overflow_pos_[pos]].value = value;
		return true;

	 UNLIKELY( case TupleStatus::MAJOR_OVERFLOW: )
	 	stash_bucket->tuples_[stash_bucket->position_[pos]].value = value;
		return true;

	 default:  // Otherwise find an empty slot and insert
	 	assert(status == TupleStatus::NOT_FOUND);
	  return Append(std::forward<K>(key), std::forward<V>(value), hash, stash_bucket);
	}
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::Bucket::Append(K &&key, V &&value, uint32_t hash, StashBucket *stash_bucket) -> bool {
	int mask;
	uint8_t idx, pos;

	pos = __builtin_ctz(~validity_);
	if (pos < bucket_capacity) {  // If bucket has a free slot, insert there
		tuples_[pos] = {key, value};
		fingerprints_[pos] = FINGERPRINT8(hash);
		SET_BIT(validity_, pos);
		return true;
	}  // Otherwise insert into the stash bucket

	if (stash_bucket == nullptr) {
		return false;
	}

	if (LIKELY( GetMinorOverflowCount() < max_minor_overflows )) {  // Insert as a minor overflow
		pos = stash_bucket->InsertMinorOverflow(std::forward<K>(key), std::forward<V>(value));
		if (pos == StashBucket::invalid_pos) {
			return false;
		}

		// Insert minor overflow metadata
		idx = __builtin_ctz(~GetMinorOverflowValidity());
		overflow_count_++;
		overflow_fp_[idx] = FINGERPRINT16(hash);
		overflow_pos_[idx] = pos;
		SET_BIT(overflow_info_, idx);

		return true;
	}

	if (stash_bucket->AppendMajorOverflow(std::forward<K>(key), std::forward<V>(value), hash)) {
		// Minor overflow slots used up; Insert as a major overflow
		overflow_count_++;
		return true;
	}
	return false;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::Bucket::Erase(const K &key, uint32_t hash, StashBucket *stash_bucket) -> bool {
	TupleStatus status;
	uint8_t pos;

	pos = FindPos(key, hash, stash_bucket, status);
	switch (status) {  // Remove `key` depending on its position
	 case TupleStatus::IN_BUCKET:
		CLEAR_BIT(validity_, pos);
		return true;

	 case TupleStatus::MINOR_OVERFLOW:
	 	CLEAR_BIT(stash_bucket->validity_, overflow_pos_[pos]);
	 	CLEAR_BIT(overflow_info_, pos);
		overflow_count_--;
		// TODO: may consider bringing back a major overflow if there is one
		return true;

	 case TupleStatus::MAJOR_OVERFLOW:
	 	CLEAR_BIT(stash_bucket->validity_, stash_bucket->position_[pos]);
		stash_bucket->position_[pos] = StashBucket::invalid_pos;
		overflow_count_--;
		return true;

	 default:
	 	assert(status == TupleStatus::NOT_FOUND);
	  return false;
	}
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::Bucket::Find(const K &key, V *value, uint32_t hash, const StashBucket *stash_bucket) const -> bool {
	TupleStatus status;
	uint8_t pos;

	pos = FindPos(key, hash, stash_bucket, status);
	switch (status) {  // Store `key`'s associate value depending on its position
	 case TupleStatus::IN_BUCKET:
		*value = tuples_[pos].value;
		return true;

	 case TupleStatus::MINOR_OVERFLOW:
	 	*value = stash_bucket->tuples_[overflow_pos_[pos]].value;
		return true;

	 case TupleStatus::MAJOR_OVERFLOW:
	 	*value = stash_bucket->tuples_[stash_bucket->position_[pos]].value;
		return true;

	 default:
	 	assert(status == TupleStatus::NOT_FOUND);
	  return false;
	}
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::Bucket::FindPos(const K &key, uint32_t hash, const StashBucket *stash_bucket, TupleStatus &status) const -> uint8_t {
	int mask;
	uint8_t idx, pos;

	SEARCH_8_128(FINGERPRINT8(hash), fingerprints_, mask);  // Search normal keys, filtering out unlikely slots using fingerprints
	mask &= validity_;
	while (mask != 0) {
		pos = __builtin_ctz(mask);
		if (LIKELY( tuples_[pos].key == key )) {
			status = TupleStatus::IN_BUCKET;
			return pos;
		}
		mask &= ~(1 << pos);
	}

	if (stash_bucket == nullptr) {
		goto not_found;
	}

	SEARCH_16_128(FINGERPRINT16(hash), overflow_fp_, mask);  // Search minor overflows, again using fingerprints
	while (mask != 0) {
		idx = __builtin_ctz(mask) / 2;
		if (GET_BIT(overflow_info_, idx) && LIKELY( stash_bucket->tuples_[overflow_pos_[idx]].key == key )) {
			status = TupleStatus::MINOR_OVERFLOW;
			return idx;
		}
		mask &= ~(3 << (idx * 2));
	}

	if (UNLIKELY( overflow_count_ > GetMinorOverflowCount() )) {  // Search major overflows in stash bucket
		idx = stash_bucket->FindMajorOverflowIdx(key, hash);
		if (idx != StashBucket::invalid_pos) {
			status = TupleStatus::MAJOR_OVERFLOW;
			return idx;
		}
	}

 not_found:
	status = TupleStatus::NOT_FOUND;
	return StashBucket::invalid_pos;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::InsertMajorOverflow(K &&key, V &&value, uint32_t hash) -> bool {
	int mask;
	uint8_t idx, pos;

	idx = FindMajorOverflowIdx(key, hash);
	if (idx != invalid_pos) {  // Overwrite duplicate if found
		tuples_[position_[idx]].value = value;
		return true;
	}

	return AppendMajorOverflow(std::forward<K>(key), std::forward<V>(value), hash);  // Insert at an empty slot
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::AppendMajorOverflow(K &&key, V &&value, uint32_t hash) -> bool {
	int mask;
	uint8_t idx, pos;

	if (~validity_ == 0) {  // No free slots, so insertion fails
		return false;
	}

	SEARCH_8_128(invalid_pos, position_, mask);  // Search for an major overflow slot
	mask &= 0xff;
	if (UNLIKELY( mask == 0 )) {  // Major overflow metadata used up, so insertion fails
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
auto DLEFT_TYPE::StashBucket::EraseMajorOverflow(const K &key, uint32_t hash) -> bool {
	uint8_t idx = FindMajorOverflowIdx(key, hash);
	if (idx == invalid_pos) {
		return false;
	}
	CLEAR_BIT(validity_, position_[idx]);
	position_[idx] = invalid_pos;
	return true;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::FindMajorOverflow(const K &key, V *value, uint32_t hash) const -> bool {
	uint8_t idx = FindMajorOverflowIdx(key, hash);
	if (idx == invalid_pos) {
		return false;
	}
	if (value != nullptr) {
		*value = tuples_[position_[idx]].value;
	}
	return true;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::FindMajorOverflowIdx(const K &key, uint32_t hash) const -> uint8_t {
	int mask;
	uint8_t idx;

	SEARCH_16_128(FINGERPRINT16(hash), fingerprints_, mask);  // Use fingerprints to filter out unlikely slots
	while (mask != 0) {
		idx = __builtin_ctz(mask) / 2;
		if (position_[idx] != invalid_pos && LIKELY( tuples_[position_[idx]].key == key )) {
			return idx;
		}
		mask &= ~(3 << (idx * 2));
	}
	return invalid_pos;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::InsertMinorOverflow(K &&key, V &&value) -> uint8_t {
	uint8_t pos;

	if (~validity_ == 0) {
		return invalid_pos;
	}  // Find an empty slot and insert
	pos = __builtin_ctzll(~validity_);
	tuples_[pos] = {key, value};
	SET_BIT(validity_, pos);

	return pos;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::EraseMinorOverflow(const K &key, uint8_t pos) -> bool {
	assert(GET_BIT(validity_, pos));
	if (LIKELY( tuples_[pos].key == key )) {
		CLEAR_BIT(validity_, pos);
		return true;
	}
	return false;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::FindMinorOverflow(const K &key, V *value, uint8_t pos) const -> bool {
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
