/**
 * @file dleft_fp_stash.hpp
 * @brief A read-optimized D-left hashing implementation with fingerprinting & stashing
 * Below is the bucket size distribution after inserting 1m keys into 65536 buckets using
 * 2-left hashing in one experiment:
 * -------------------------------------------------------------------------------------
 * | Bucket Size  | 9 | 10 | 11 | 12  | 13  |  14  |  15   |  16   |  17   |  18  | 19 |
 * -------------------------------------------------------------------------------------
 * | Bucket Count | 2 | 8  | 20 | 122 | 699 | 3176 | 12745 | 28572 | 18545 | 1643 | 4  |
 * -------------------------------------------------------------------------------------
 * We can conclude that with high probabilitym most buckets are of size at most 16,
 * and no bucket's size exceeds 19.
 * 
 * Based on this observation, if we set bucket size to 19, the space utilization will be
 * bad. But if we set bucket size to 16, then with high probability most buckets will not
 * overflow, and those that do will have at most 3 overflow keys.
 * 
 * For keys that do not overflow, we speed up their searching using fingerprints, as they
 * provide better cache locality and thus better performance. A fingerprint for a normal
 * key is just an 8-bit hash of the key that works as a filter for the key, with false
 * positive rate 1/256.
 * 
 * As for overflow keys, we store them in shared stash buckets. To achieve better load
 * factor, we do not statically assign stash buckets. Instead, we associate 4 candidate
 * stash buckets with each bucket. When that bucket overflows for the first time, we
 * dynamically bind it to its least loaded candidate stash bucket, and try to insert the
 * overflow key into the stash bucket. Note that once a bucket and a stash bucket are
 * bound, they cannot be unbound until all overflow keys are deleted. When the stash
 * bucket also fails to resolve an insertion, the entire table is expanded and rehashed.
 * We do not grow the stash bucket chain indefinitely as this hurts read latency, which
 * is against our design principle. This is also why we use relatively large (64-slot)
 * stash buckets, as this makes a stash bucket less likely to run out of space, deferring
 * expensive resizing.
 * 
 * When we fail to find a key in an overflowing bucket, we may need to search its entire
 * stash bucket, which can be daunting. To circumvent this, we use 4 more "tail" finger-
 * prints for the first 4 keys overflowing from this bucket. We also use 4 "pointers"
 * (which are actually indexes) to point to their locations in the stash bucket, so that
 * we can retrieve them without having to scan the entire stash bucket. Note that 4 is
 * more than enough, as most buckets have at most 3 overflow keys. Since the overflow keys
 * are searched only after the entire bucket is searched, and because overflow keys have
 * bad spatial locality, we use 16-bit fingerprints for them, so as to bound the worst-
 * case scenario.
 * 
 * We call the first 4 overflows from each bucket "minor", as they can be efficiently
 * retrieved using the fingerprints and the pointers. However, we still have to search
 * the entire stash bucket when there are more than 4 overflows, which is why we call
 * overflows other than the first 4 "major". While major overflows are extremely unlikely
 * to occur, we allocate 2 16-bit fingerprints and 2 pointers (indexes) in each stash
 * bucket to handle them, so that they can still be retrieved in reasonable time.
 * 
 * Finally, we neeed to determine the (# of buckets) to (# of stash buckets) ratio. Since
 * there are about 2% overflow keys in our experiments, we use 256 : 1. This also keeps
 * these two numbers powers of 2, allowing for quick modulo operations.
 * 
 * TODO: 1. The "one move" strategy in George Varghese's paper can be applied; (Edit: DONE)
 * 			 2. Since the bucket header size is less than one cacheline size, we can use the
 * 					remaining space as a write buffer, as described in the Pea Hash paper;
 * 			 3. Our hash table now uses only 32-bit hash, which means it supports at most
 * 					65,536 buckets. We can consider using larger hash and extend the hash table
 * 					to more buckets.
 */
#pragma once

#include <stdint.h>

#include <immintrin.h>

#include <iostream>
#include <limits>

#include <cassert>
#include <cstring>

// #define __TEST_DLEFT__
// #define __DEBUG_DLEFT__

// #define __COUNT_FALSE_POSITIVES__
// #define __COUNT_OVERFLOWS__

#ifdef __COUNT_FALSE_POSITIVES__
static uint64_t false_positive{0};
static uint64_t overflow_false_positives{0};
#endif

#ifdef __COUNT_OVERFLOWS__
static uint64_t minor_overflows{0};
static uint64_t major_overflows{0};
#endif

#ifdef __DEBUG_DLEFT__
# define DEBUG_DLEFT(foo) foo
#else
# define DEBUG_DLEFT(foo)
#endif

#define CACHELINE_SIZE (64)

#define BUCKET_STASH_BUCKET_RATIO (1024)

#define MAX_LOAD_FACTOR_100 (95)

#define BYTE_ROUND_UP(n) (((n) + 7) / 8)
#define ROUND_UP(n, b) (((n) + (b) - 1) / (b))
#define ROUNDUP_POWER_2(n) ((n) == 0 ? 1 : (((n) & ((n) - 1)) == 0) ? (n) : (1 << (64 -__builtin_clzll(n))))

#define GET_BIT(bits, n)   (bits & (1ull << (n)))
#define SET_BIT(bits, n)   (bits |= (1ull << (n)))
#define CLEAR_BIT(bits, n) (bits &= ~(1ull << (n)))

#define GET_BIT_256(bits, n)   (bits[n/64] & (1ull << (n%64)))
#define SET_BIT_256(bits, n)   (bits[n/64] |= (1ull << (n%64)))
#define CLEAR_BIT_256(bits, n) (bits[n/64] &= ~(1ull << (n%64)))

#define FP(hash)   (static_cast<uint8_t>(hash))
#define OFP(hash)  (static_cast<uint16_t>(hash))
#define IDX1(hash) (static_cast<uint32_t>(hash))
#define IDX2(hash) (static_cast<uint32_t>(hash >> 32))

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

static void *buckets;
static void *stash_buckets;

template <class K, class V, class H>
class DleftFpStash {
 public:
	using hash_t = uint64_t;
	using idx_t  = uint32_t;
	using pos_t  = uint8_t;
	using fp_t   = uint8_t;
	using ofp_t  = uint16_t;

	DleftFpStash(size_t size = 0)
			: num_buckets_(ROUNDUP_POWER_2(size / Bucket::bucket_capacity)),
				num_stash_buckets_(num_buckets_ / BUCKET_STASH_BUCKET_RATIO) {
		if (num_buckets_ > (1 << 16)) {
			printf("error: table is too large\n");
			exit(1);
		}

		buckets = buckets_ = new Bucket[num_buckets_];
		assert(buckets_ != nullptr);

		if (num_stash_buckets_ == 0) {
			stash_buckets_ = nullptr;
		} else {
			stash_buckets = stash_buckets_ = new StashBucket[num_stash_buckets_];
			assert(stash_buckets_ != nullptr);
		}
	}

	~DleftFpStash() { delete[] buckets_; delete[] stash_buckets_; }

	auto insert(K &&key, V &&value) -> bool {
		hash_t hash = H()(key);
		InsertStatus status = Insert<false>(std::forward<K>(key), std::forward<V>(value), hash);
		while (status == InsertStatus::FAILED) {
			Resize(capacity() * 2);
			status = Insert<false>(std::forward<K>(key), std::forward<V>(value), hash);
		}
		return status == InsertStatus::INSERTED;
	}

	auto erase(const K &key) -> bool { return Erase(key, H()(key)); }

	auto find(const K &key, V &value) const -> bool { return Find(key, &value, H()(key)); }

	void clear() {
		for (idx_t i = 0; i < num_buckets_; i++) {
			buckets_[i].Clear();
		}
		for (idx_t i = 0; i < num_stash_buckets_; i++) {
			stash_buckets_[i].Clear();
		}
	}

	void reserve(size_t size) { Resize(size); }

	auto load_factor() const -> double { return 1.0 * size_ / capacity(); }

	auto capacity() const -> size_t { return BucketCapacity() + StashBucketCapacity(); }

	auto size() const -> size_t { return size_; }

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
    // static constexpr size_t buf_size = CACHELINE_SIZE - header_size;
    // static constexpr int buf_capacity = buf_size / tuple_size;

    // header (36 bytes)
		// TODO: probably use partial keys instead of fingerprints to further enhance space utilization
    fp_t fingerprints_[bucket_capacity];       // fingerprints for each in-bucket key
    uint16_t validity_{0};                     // validity bitmap for each in-bucket key
		pos_t overflow_count_{0};                  // the total number of overflows
		uint8_t overflow_info_{0};                 // the 4 highest bits specify the stash bucket;
		                                           // the 4 lowest bits serve as the validity bits for minor overflows
		ofp_t overflow_fp_[max_minor_overflows];   // fingerprints for minor overflows
		pos_t overflow_pos_[max_minor_overflows];  // positions of minor overflows in the stash bucket
		idx_t stash_stride_;

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
		auto Insert(K &&key, V &&value, ofp_t fp, StashBucket *stash_bucket) -> bool {
			TupleStatus status;
			int mask;
			pos_t idx, pos;

			pos = FindPos(key, fp, stash_bucket, status);
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
				return Append(std::forward<K>(key), std::forward<V>(value), fp, stash_bucket);
			}
		}

		// Inserts a key without duplicate checks
		auto Append(K &&key, V &&value, ofp_t fp, StashBucket *stash_bucket) -> bool {
			int mask;
			pos_t idx, pos;

			pos = __builtin_ctz(~validity_);
			if (pos < bucket_capacity) {  // If bucket has a free slot, insert there
				InsertAt(std::forward<K>(key), std::forward<V>(value), pos, FP(fp));
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
				overflow_fp_[idx] = fp;
				overflow_pos_[idx] = pos;
				SET_BIT(overflow_info_, idx);
				DEBUG_DLEFT(
					printf("Minor overflow from bucket %ld to stash bucket %ld\n",
								 this - reinterpret_cast<Bucket *>(buckets),
								 stash_bucket - reinterpret_cast<StashBucket *>(stash_buckets));
				)

				return true;
			}

			// Minor overflow slots used up; Insert as a major overflow
			if (stash_bucket->AppendMajorOverflow(std::forward<K>(key), std::forward<V>(value), fp)) {
				overflow_count_++;
				DEBUG_DLEFT(
					printf("Major overflow from bucket %ld to stash bucket %ld\n",
								 this - reinterpret_cast<Bucket *>(buckets),
								 stash_bucket - reinterpret_cast<StashBucket *>(stash_buckets));
				)
				return true;
			}
			return false;
		}

		// Removes a key
		auto Erase(const K &key, ofp_t fp, StashBucket *stash_bucket) -> bool {
			TupleStatus status;
			pos_t pos;

			pos = FindPos(key, fp, stash_bucket, status);
			switch (status) {  // Remove `key` depending on its position
			 case TupleStatus::IN_BUCKET:
				CLEAR_BIT(validity_, pos);
				return true;

			 case TupleStatus::MINOR_OVERFLOW:
				CLEAR_BIT_256(stash_bucket->validity_, overflow_pos_[pos]);
				CLEAR_BIT(overflow_info_, pos);
				overflow_count_--;
				// TODO: may consider bringing back a major overflow if there is one
				return true;

			 case TupleStatus::MAJOR_OVERFLOW:
				CLEAR_BIT_256(stash_bucket->validity_, stash_bucket->position_[pos]);
				stash_bucket->position_[pos] = StashBucket::invalid_pos;
				overflow_count_--;
				return true;

			 default:
				assert(status == TupleStatus::NOT_FOUND);
				return false;
			}
		}

		// Looks for a key and returns the associated value
		auto Find(const K &key, V *value, ofp_t fp, const StashBucket *stash_bucket) const -> bool {
			TupleStatus status;
			pos_t pos;

			pos = FindPos(key, fp, stash_bucket, status);
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

		void InsertAt(K &&key, V &&value, pos_t pos, fp_t fp) {
			tuples_[pos].key = key;
			tuples_[pos].value = value;
			fingerprints_[pos] = fp;
			SET_BIT(validity_, pos);
		}

		// Searches for a key and returns its position
		// `status` == IN_BUCKET: returns the key's position in bucket
		// `status` == MINOR_OVERFLOW: returns the index of the key's `overflow_fp_` and `overflow_pos_`
		// `status` == MAJOR_OVERFLOW: returns the index of the key's `fingerprints_` and `position_` in stash bucket
		auto FindPos(const K &key, ofp_t fp, const StashBucket *stash_bucket, TupleStatus &status) const -> pos_t {
			int mask;
			pos_t idx, pos;

			SEARCH_8_128(FP(fp), fingerprints_, mask);  // Search normal keys, filtering out unlikely slots using fingerprints
			mask &= validity_;
			while (mask != 0) {
				pos = __builtin_ctz(mask);
				if (LIKELY( tuples_[pos].key == key )) {
					status = TupleStatus::IN_BUCKET;
					return pos;
				}
			 #ifdef __COUNT_FALSE_POSITIVES__
				false_positive++;
			 #endif
				mask &= ~(1 << pos);
			}

			if (stash_bucket == nullptr) {
				goto not_found;
			}

			for (int i = 0; i < Bucket::max_minor_overflows; i += 2) {
				if (GET_BIT(overflow_info_, i) && overflow_fp_[i] == fp) {
					if (LIKELY( stash_bucket->tuples_[overflow_pos_[i]].key == key )) {
						status = TupleStatus::MINOR_OVERFLOW;
						return i;
					}
				 #ifdef __COUNT_FALSE_POSITIVES__
					false_positive++;
					overflow_false_positives++;
				 #endif
				}
				if (GET_BIT(overflow_info_, i + 1) && overflow_fp_[i + 1] == fp) {
					if (LIKELY( stash_bucket->tuples_[overflow_pos_[i + 1]].key == key )) {
						status = TupleStatus::MINOR_OVERFLOW;
						return i + 1;
					}
				 #ifdef __COUNT_FALSE_POSITIVES__
					false_positive++;
					overflow_false_positives++;
				 #endif
				}
			}

			if (UNLIKELY( overflow_count_ > GetMinorOverflowCount() )) {  // Search major overflows in stash bucket
				idx = stash_bucket->FindMajorOverflowIdx(key, fp);
				if (idx != StashBucket::invalid_pos) {
					status = TupleStatus::MAJOR_OVERFLOW;
					return idx;
				}
			}

		 not_found:
			status = TupleStatus::NOT_FOUND;
			return StashBucket::invalid_pos;
		}

		void Clear() {
			validity_ = 0; overflow_count_ = 0; overflow_info_ = 0;
			memset(overflow_pos_, StashBucket::invalid_pos, sizeof(overflow_pos_));
		}

		// Get the number of valid keys in bucket (not including overflows)
		auto GetSize() const -> pos_t { return __builtin_popcount(validity_); }

		// Get the total number of valid keys & overflow keys in bucket
		auto GetTotal() const -> pos_t { return GetSize() + overflow_count_; }

		// Get its stash bucket's number (each bucket has at most four candidate stash buckets)
		auto GetStashBucketNum() const -> uint8_t { return overflow_info_ >> 4; }

		// Bind the bucket to a stash bucket; once bound, it cannot be unbounded unless the number of overflows becomes 0
		void SetStashBucketNum(uint8_t num) { overflow_info_ = (num << 4) | GetMinorOverflowValidity(); }

		auto GetStashBucketIndex(idx_t idx, idx_t max) const -> size_t {
			return (idx / BUCKET_STASH_BUCKET_RATIO + GetStashBucketNum() * stash_stride_) & (max - 1);
		}

		// Get the validity bitmap for its moinor overflows
		auto GetMinorOverflowValidity() const -> uint8_t { return overflow_info_ & 0xf; }

		// Get the number of minor overflows
		auto GetMinorOverflowCount() const -> uint8_t { return __builtin_popcount(GetMinorOverflowValidity()); }
  };

  struct StashBucket {
    static constexpr size_t header_size = 40;
    static constexpr int max_major_overflows = 2;
    static constexpr int bucket_capacity = 255;
		static constexpr uint8_t invalid_pos = 0xff;

    // header (40 B)
    ofp_t fingerprints_[max_major_overflows];               // fingerprints for major overflows
		pos_t position_[max_major_overflows];                  // positions of each major overflow in stash bucket
    uint64_t validity_[ROUND_UP(bucket_capacity, 64)] {0};  // validity bitmap for each overflow key in stash bucket

		// key-value pairs
    Tuple tuples_[bucket_capacity];

		StashBucket() { memset(position_, invalid_pos, sizeof(position_)); }

		// Inserts a major overflow, overwriting duplicates
    auto InsertMajorOverflow(K &&key, V &&value, ofp_t fp) -> bool {
			pos_t idx, pos;

			idx = FindMajorOverflowIdx(key, fp);
			if (idx != invalid_pos) {  // Overwrite duplicate if found
				tuples_[position_[idx]].value = value;
				return true;
			}

			return AppendMajorOverflow(std::forward<K>(key), std::forward<V>(value), fp);  // Insert at an empty slot
		}

		// Inserts a major overflow without checking duplicates
		auto AppendMajorOverflow(K &&key, V &&value, ofp_t fp) -> bool {
			pos_t idx, pos;

			if ((pos = FindFreeSlot()) == invalid_pos) {  // No free slots, so insertion fails
				return false;
			}

			if (position_[0] == invalid_pos) {
				idx = 0;
			} else if (position_[1] == invalid_pos) {
				idx = 1;
			} else {  // Major overflow metadata used up, so insertion fails
				return false;
			}

			tuples_[pos].key = key;
			tuples_[pos].value = value;
			position_[idx] = pos;
			fingerprints_[idx] = OFP(fp);
			SET_BIT_256(validity_, pos);
		 #ifdef __COUNT_OVERFLOWS__
			major_overflows++;
		 #endif

			return true;
		}

		// Removes a major overflow key
    auto EraseMajorOverflow(const K &key, ofp_t fp) -> bool {
			pos_t idx = FindMajorOverflowIdx(key, fp);
			if (idx == invalid_pos) {
				return false;
			}
			CLEAR_BIT_256(validity_, position_[idx]);
			position_[idx] = invalid_pos;
			return true;
		}

		// Searches for a major overflow key and returns its associated value
    auto FindMajorOverflow(const K &key, V *value, ofp_t fp) const -> bool {
			pos_t idx = FindMajorOverflowIdx(key, fp);
			if (idx == invalid_pos) {
				return false;
			}
			if (value != nullptr) {
				*value = tuples_[position_[idx]].value;
			}
			return true;
		}

		// Searches for a major overflow key and returns its index of `fingerprints_` and `position_`
		auto FindMajorOverflowIdx(const K &key, ofp_t fp) const -> pos_t {
			pos_t idx;

			if (position_[0] != invalid_pos && fingerprints_[0] == fp) {
				if (LIKELY( tuples_[position_[0]].key == key )) {
					return 0;
				}
			 #ifdef __COUNT_FALSE_POSITIVES__
				false_positive++;
				overflow_false_positives++;
			 #endif
			}
			if (position_[1] != invalid_pos && fingerprints_[1] == fp) {
				if (LIKELY( tuples_[position_[1]].key == key )) {
					return 1;
				}
			 #ifdef __COUNT_FALSE_POSITIVES__
				false_positive++;
				overflow_false_positives++;
			 #endif
			}
			return invalid_pos;
		}

    auto InsertMinorOverflow(K &&key, V &&value) -> pos_t {
			pos_t pos;

			// Find an empty slot and insert
			if ((pos = FindFreeSlot()) == invalid_pos) {
				return invalid_pos;
			}
			tuples_[pos].key = key;
			tuples_[pos].value = value;
			SET_BIT_256(validity_, pos);
		 #ifdef __COUNT_OVERFLOWS__
			minor_overflows++;
		 #endif

			return pos;
		}

    auto EraseMinorOverflow(const K &key, pos_t pos) -> bool {
			assert(GET_BIT_256(validity_, pos));
			if (LIKELY( tuples_[pos].key == key )) {
				CLEAR_BIT_256(validity_, pos);
				return true;
			}
			return false;
		}

    auto FindMinorOverflow(const K &key, V *value, pos_t pos) const -> bool {
			assert(GET_BIT_256(validity_, pos));
			if (LIKELY( tuples_[pos].key == key )) {
				if (value != nullptr) {
					*value = tuples_[pos].value;
				}
				return true;
			}
			return false;
		}

		void Clear() { memset(validity_, 0, sizeof(validity_)); memset(position_, invalid_pos, sizeof(position_)); }

		// Returns the number of valid overflow keys in bucket (either major or minor)
		auto GetSize() const -> pos_t {
			return __builtin_popcountll(validity_[0]) + __builtin_popcountll(validity_[1])
						 + __builtin_popcountll(validity_[2]) + __builtin_popcountll(validity_[3]);
		};

		auto FindFreeSlot() const -> pos_t {
		 	if (~validity_[0] != 0) {
				return __builtin_ctzll(~validity_[0]);
			} else if (~validity_[1] != 0) {
				return 64 + __builtin_ctzll(~validity_[1]);
			} else if (~validity_[2] != 0) {
				return 128 + __builtin_ctzll(~validity_[2]);
			}
			return 192 + __builtin_ctzll(~validity_[3]);
		}
  };

	enum class InsertStatus { INSERTED, EXISTED, FAILED };

	// Check for duplicate key in a bucket; If found, return `true` and overwrite the value if `upsert`
	template<bool upsert = true>
	auto CheckDuplicate(K &&key, V &&value, idx_t idx, hash_t hash) -> bool {
		using TupleStatus = typename Bucket::TupleStatus;
		Bucket *bucket = &buckets_[idx];
		TupleStatus status;
		StashBucket *stash_bucket = nullptr;
		pos_t pos;

		if (stash_buckets_ != nullptr && bucket->overflow_count_ > 0) {
			stash_bucket = &stash_buckets_[bucket->GetStashBucketIndex(idx, num_stash_buckets_)];
		}

		pos = bucket->FindPos(key, hash, stash_bucket, status);
		if (status == TupleStatus::IN_BUCKET) {
			if (upsert) {
				bucket->tuples_[pos].value = value;
			}
			return true;
		} else if (status == TupleStatus::MINOR_OVERFLOW) {
			if (upsert) {
				stash_bucket->tuples_[bucket->overflow_pos_[pos]].value = value;
			}
			return true;
		} else if (UNLIKELY( status == TupleStatus::MAJOR_OVERFLOW )) {
			if (upsert) {
				stash_bucket->tuples_[stash_bucket->position_[pos]].value = value;
			}
			return true;
		}
		return false;
	}

	// Try to insert a kv pair into bucket, without duplicate check
	auto TryInsert(K &&key, V &&value, idx_t idx, hash_t hash) -> bool {
		Bucket *bucket = &buckets_[idx];
		StashBucket *stash_bucket = nullptr;

		if (bucket->overflow_count_ == 0) {  // No stash bucket yet
			if (bucket->Append(std::forward<K>(key), std::forward<V>(value), hash, nullptr)) {
				size_++;
				return true;
			}  // Bucket is full; need a stash bucket
			if (stash_buckets_ == nullptr) {  // No usable stash bucket, so insertion fails
				return false;
			}

			size_t stash_idx = (idx / BUCKET_STASH_BUCKET_RATIO) & (num_stash_buckets_ - 1);
			uint8_t min_stash_num;
			pos_t min_stash_size = 0xff;
			bucket->stash_stride_ = GetStride(idx);
			for (uint8_t stash_num = 0; stash_num < 16; stash_num++) {  // Bind bucket to its most underfull candidate stash bucket
				stash_idx = (stash_idx + bucket->stash_stride_) & (num_stash_buckets_ - 1);
				pos_t size = stash_buckets_[stash_idx].GetSize();
				if (size < min_stash_size) {
					min_stash_num = stash_num;
					min_stash_size = size;
				}
				DEBUG_DLEFT( printf("Candidate stash bucket %u of bucket %u: %lu\n", stash_num, idx, stash_idx); )
			}
			bucket->SetStashBucketNum(min_stash_num);
			assert(bucket->GetStashBucketNum() == min_stash_num);
			DEBUG_DLEFT(
				printf("Bucket %u bound with stash bucket %lu(%u)\n",
							idx, bucket->GetStashBucketIndex(idx, num_stash_buckets_), min_stash_num);
			)
		}

		// Retry insertion with stash bucket
		assert(stash_buckets_ != nullptr);
		stash_bucket = &stash_buckets_[bucket->GetStashBucketIndex(idx, num_stash_buckets_)];
		if (bucket->Append(std::forward<K>(key), std::forward<V>(value), hash, stash_bucket)) {
			size_++;
			return true;
		}
		return false;
	}

	// Try to move one key in a bucket to its alternative bucket
	// Returns the index of the moved key; If no key can be moved, return `invalid_pos`
	auto OneMove(idx_t idx) -> pos_t {
		Bucket *bucket = &buckets_[idx];

		// @note Computing (potentially) two hashes for each key seems a bit expensive here;
		//       We should consider using one additional bit for each key to indicate which hash
		//       function to use, or simply store the other hash as fingerprint (in which case
		//       the distance between the two buckets must be limited).
		//       For now we just keep it that way, since we're focusing on read latency.
		assert(bucket->GetSize() == Bucket::bucket_capacity);
		for (int i = 0; i < Bucket::bucket_capacity; i++) {
			hash_t hash = H()(bucket->tuples_[i].key);
			idx_t alt_idx = IDX1(hash) & (num_buckets_ - 1);
			uint8_t alt_fp = FP(IDX2(hash));
			if (alt_idx == idx) {
				alt_fp = FP(alt_idx);
				alt_idx = IDX2(hash) & (num_buckets_ - 1);
				if (UNLIKELY( alt_idx == idx )) {
					continue;
				}
			}
			Bucket *alt_bucket = &buckets_[alt_idx];
			if (alt_bucket->GetSize() == Bucket::bucket_capacity) {
				continue;
			}  // `alt_bucket` has free space, so move the key there
			alt_bucket->Append(std::move(bucket->tuples_[i].key), std::move(bucket->tuples_[i].value), hash, nullptr);
			return static_cast<uint8_t>(i);
		}
		return StashBucket::invalid_pos;
	}

	// Inserts a key into the hash table; If a duplicate is found, the value is overwritten
	// Returns `INSERTED` if insertion was successful, `EXISTED` if a duplicate key is found,
	// and `FAILED` if the insertion failed (e.g. when running out of space)
	// template argument `upsert` defines whether to overwrite duplicates
	template<bool upsert = true>
  auto Insert(K &&key, V &&value, hash_t hash) -> InsertStatus {
		idx_t idx1 = IDX1(hash) & (num_buckets_ - 1);
		idx_t idx2 = IDX2(hash) & (num_buckets_ - 1);

		// Check for duplicates
		if (CheckDuplicate<upsert>(std::forward<K>(key), std::forward<V>(value), idx1, OFP(idx2)) ||
				CheckDuplicate<upsert>(std::forward<K>(key), std::forward<V>(value), idx2, OFP(idx1))) {
			return InsertStatus::EXISTED;
		}  // If not found, insert
		return Append(std::forward<K>(key), std::forward<V>(value), hash) ?
					 InsertStatus::INSERTED : InsertStatus::FAILED;
	}

	// Inserts a key into the hash table without duplicate checks
	// Returns `true` if insertion is successful and `false` otherwise (e.g. when running out of space)
	auto Append(K &&key, V &&value, hash_t hash) -> bool {
		idx_t idx1 = IDX1(hash) & (num_buckets_ - 1);
		idx_t idx2 = IDX2(hash) & (num_buckets_ - 1);

		// Try inserting into the more underfull candidate bucket first
		if (buckets_[idx1].GetTotal() <= buckets_[idx2].GetTotal()) {
			if (TryInsert(std::forward<K>(key), std::forward<V>(value), idx1, OFP(idx2)) ||
					TryInsert(std::forward<K>(key), std::forward<V>(value), idx2, OFP(idx1))) {
				return true;
			}
		} else {
			if (TryInsert(std::forward<K>(key), std::forward<V>(value), idx2, OFP(idx1)) ||
					TryInsert(std::forward<K>(key), std::forward<V>(value), idx1, OFP(idx2))) {
				return true;
			}
		}

		// Insertion failed; do one move on both buckets
		pos_t pos;
		if ((pos = OneMove(idx1)) != StashBucket::invalid_pos) {
			buckets_[idx1].InsertAt(std::forward<K>(key), std::forward<V>(key), pos, OFP(idx2));
		} else if ((pos = OneMove(idx2)) != StashBucket::invalid_pos) {
			buckets_[idx2].InsertAt(std::forward<K>(key), std::forward<V>(key), pos, OFP(idx1));
		}

		return false;
	}

	// Removes a key from the hash table
	// Returns `true` if found and `false` otherwise
  auto Erase(const K &key, hash_t hash) -> bool {
		idx_t idx1 = IDX1(hash) & (num_buckets_ - 1);
		idx_t idx2 = IDX2(hash) & (num_buckets_ - 1);
		Bucket *bucket1 = &buckets_[idx1], *bucket2 = &buckets_[idx2];
		StashBucket *stash_bucket1{nullptr}, *stash_bucket2{nullptr};

		if (bucket1->overflow_count_ > 0 && stash_buckets_ != nullptr) {
			stash_bucket1 = &stash_buckets_[bucket1->GetStashBucketIndex(idx1, num_stash_buckets_)];
		}
		if (bucket1->Erase(key, OFP(idx2), stash_bucket1)) {  // Try remove from the first bucket
			size_--;
			return true;
		} else if (idx1 == idx2) {
			return false;
		}

		if (bucket2->overflow_count_ > 0 && stash_buckets_ != nullptr) {
			stash_bucket2 = &stash_buckets_[bucket2->GetStashBucketIndex(idx2, num_stash_buckets_)];
		}  // If not found, try remove from the second bucket
		if (bucket2->Erase(key, OFP(idx1), stash_bucket2)) {
			size_--;
			return true;
		}
		return false;
	}

	// Searches for a key from the hash table
	// Returns `true` if found and `false` otherwise; value is stored in the second argument
  auto Find(const K &key, V *value, hash_t hash) const -> bool {
		// TODO: parallelize the probing of two buckets
		idx_t idx1 = IDX1(hash) & (num_buckets_ - 1);
		idx_t idx2 = IDX2(hash) & (num_buckets_ - 1);
		Bucket *bucket1 = &buckets_[idx1], *bucket2 = &buckets_[idx2];
		StashBucket *stash_bucket1{nullptr}, *stash_bucket2{nullptr};

		if (bucket1->overflow_count_ > 0 && stash_buckets_ != nullptr) {
			stash_bucket1 = &stash_buckets_[bucket1->GetStashBucketIndex(idx1, num_stash_buckets_)];
		}
		if (bucket1->Find(key, value, OFP(idx2), stash_bucket1)) {  // Search the first bucket
			return true;
		} else if (idx1 == idx2) {
			return false;
		}

		if (bucket2->overflow_count_ > 0 && stash_buckets_ != nullptr) {
			stash_bucket2 = &stash_buckets_[bucket2->GetStashBucketIndex(idx2, num_stash_buckets_)];
		}  // If not found, search the second bucket
		return bucket2->Find(key, value, OFP(idx1), stash_bucket2);
	}

	// Resize the table; may fail if the new size is smaller than current size
	// Returns `true` if resize is successful and false otherwise
  auto Resize(size_t new_size) -> bool {
		idx_t old_num_buckets = num_buckets_;
		idx_t old_num_stash_buckets = num_stash_buckets_;
		Bucket *old_buckets = buckets_;
		StashBucket *old_stash_buckets = stash_buckets_;
		size_t new_capacity = ROUNDUP_POWER_2(new_size / Bucket::bucket_capacity);

		if (num_buckets_ == new_capacity) {
			return true;
		}

		if (new_capacity > std::numeric_limits<idx_t>::max()) {
			printf("error: table is too large\n");
			exit(1);
		}
		num_buckets_ = new_capacity;
		num_stash_buckets_ = num_buckets_ / BUCKET_STASH_BUCKET_RATIO;
		buckets_ = new Bucket[num_buckets_];
		assert(buckets_ != nullptr);
		stash_buckets_ = (num_stash_buckets_ > 0 ? new StashBucket[num_stash_buckets_] : nullptr);

		for (size_t i = 0; i < old_num_buckets; i++) {  // Iterate over normal buckets and rehash the keys
			for (int j = 0; j < Bucket::bucket_capacity; j++) {
				if (!GET_BIT(old_buckets[i].validity_, j)) {
					continue;
				}
				auto &key = old_buckets[i].tuples_[j].key;
				auto &value = old_buckets[i].tuples_[j].value;
				hash_t hash = H()(key);
				if (!Append(std::move(key), std::move(value), hash)) {
					goto resize_failed;
				}
			}
		}
		for (size_t i = 0; i < old_num_stash_buckets; i++) {  // Iterate over stash buckets and rehash the keys
			for (int j = 0; j < StashBucket::bucket_capacity; j++) {
				if (!GET_BIT_256(old_stash_buckets[i].validity_, j)) {
					continue;
				}
				auto &key = old_stash_buckets[i].tuples_[j].key;
				auto &value = old_stash_buckets[i].tuples_[j].value;
				hash_t hash = H()(key);
				if (!Append(std::move(key), std::move(value), hash)) {
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

	auto BucketCapacity() const -> size_t { return Bucket::bucket_capacity * num_buckets_; }

	auto StashBucketCapacity() const -> size_t { return StashBucket::bucket_capacity * num_stash_buckets_; }

	static auto GetStride(uint16_t idx) -> idx_t {
		hash_t hash = H()(idx);
		return IDX1(hash) ^ IDX2(hash);
		// return idx * idx + 7 * idx + 457;
	}

	idx_t num_buckets_;

	idx_t num_stash_buckets_;

	size_t size_{0};

	size_t overflow_count_{0};

	Bucket *buckets_;

	StashBucket *stash_buckets_;

#ifdef __TEST_DLEFT__
	friend class DleftTest;
#endif
};

#ifdef __TEST_DLEFT__

#include "xxhash.h"
#include <map>

class DleftTest {
 public:
 	static void RunAllTests() {
		TestStashBucket();
    TestBucket();
    TestDleft();
	}
 private:
	static constexpr uint64_t seed = 0x42ae2f8ce193f9da;

	template<class K, uint64_t seed>
  class HasherUll {
    public:
    auto operator()(const K &key) const -> uint64_t {
      return XXH64(&key, sizeof(K), seed);
    }
  };

	using Hasher = HasherUll<uint32_t, seed>;

	using DleftType = DleftFpStash<uint32_t, uint32_t, Hasher>;

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

		TestDleftFalsePositives();
		TestDleftMaxLoadFactor();
  }

  static void TestStashBucketInsertMinorOverflow() {
    printf("[TEST STASH BUCKET INSERT MINOR OVERFLOW]\n");

    DleftType::StashBucket bucket;

    for (int i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.InsertMinorOverflow(i, i) != bucket.invalid_pos);
    }
    assert(bucket.InsertMinorOverflow(2023u, 2023u) == bucket.invalid_pos);

    printf("[PASSED]\n");
  }

  static void TestStashBucketEraseMinorOverflow() {
    printf("[TEST STASH BUCKET ERASE MINOR OVERFLOW]\n");

    DleftType::StashBucket bucket;
    uint8_t pos[bucket.bucket_capacity];

    for (int i = 0; i < bucket.bucket_capacity; i++) {
      pos[i] = bucket.InsertMinorOverflow(i, i);
      assert(pos[i] != bucket.invalid_pos);
    }

    for (int i = 0; i < bucket.bucket_capacity; i += 2) {
      assert(!bucket.EraseMinorOverflow(i+1, pos[i]));
      assert(bucket.EraseMinorOverflow(i, pos[i]));
    }

    for (int i = 0; i < bucket.bucket_capacity; i += 2) {
      assert(bucket.InsertMinorOverflow(i, i) != bucket.invalid_pos);
    }

    printf("[PASSED]\n");
  }

  static void TestStashBucketFindMinorOverflow() {
    printf("[TEST STASH BUCKET FIND MINOR OVERFLOW]\n");

    DleftType::StashBucket bucket;
    uint8_t pos[bucket.bucket_capacity];

    for (int i = 0; i < bucket.bucket_capacity; i++) {
      pos[i] = bucket.InsertMinorOverflow(i, i);
      assert(pos[i] != bucket.invalid_pos);
    }

    for (int i = 0; i < bucket.bucket_capacity; i++) {
      uint32_t value;
      assert(!bucket.FindMinorOverflow(i+1, &value, pos[i]));
      assert(bucket.FindMinorOverflow(i, &value, pos[i]));
      assert(value == i);
    }

    for (int i = 0; i < bucket.bucket_capacity; i += 2) {
      assert(bucket.EraseMinorOverflow(i, pos[i]));
    }

    for (int i = bucket.bucket_capacity - 2; i >= 0; i -= 2) {
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
      DleftType::StashBucket bucket;
      for (int i = 0; i < bucket.max_major_overflows; i++) {
        assert(bucket.AppendMajorOverflow(i, i, OFP(Hasher()(i))));
      }
      assert(!bucket.AppendMajorOverflow(2023u, 2023u, OFP(Hasher()(2023u))));
    }
    {
      DleftType::StashBucket bucket;

      for (int i = 0; i < bucket.bucket_capacity - bucket.max_major_overflows / 2; i++) {
        assert(bucket.InsertMinorOverflow(i, i) != bucket.invalid_pos);
      }
      for (int i = bucket.bucket_capacity - bucket.max_major_overflows / 2; i < bucket.bucket_capacity; i++) {
        assert(bucket.AppendMajorOverflow(i, i, OFP(Hasher()(i))));
      }
      assert(!bucket.AppendMajorOverflow(2023u, 2023u, OFP(Hasher()(2023u))));
    }

    printf("[PASSED]\n");
  }

  static void TestStashBucketEraseMajorOverflow() {
    printf("[TEST STASH BUCKET ERASE MAJOR OVERFLOW]\n");

    DleftType::StashBucket bucket;

    for (int i = 0; i < bucket.max_major_overflows; i++) {
      assert(bucket.AppendMajorOverflow(i, i, OFP(Hasher()(i))));
    }

    for (int i = 0; i < bucket.max_major_overflows; i += 2) {
      assert(bucket.EraseMajorOverflow(i, OFP(Hasher()(i))));
      assert(!bucket.EraseMajorOverflow(i, OFP(Hasher()(i))));
    }

    for (int i = 0; i < bucket.max_major_overflows; i += 2) {
      assert(bucket.AppendMajorOverflow(i, i, OFP(Hasher()(i))));
    }

    printf("[PASSED]\n");
  }

  static void TestStashBucketFindMajorOverflow() {
    printf("[TEST STASH BUCKET FIND MAJOR OVERFLOW]\n");

    DleftType::StashBucket bucket;

    for (int i = 0; i < bucket.max_major_overflows; i++) {
      assert(bucket.AppendMajorOverflow(i, i, OFP(Hasher()(i))));
    }
    for (int i = 0; i < bucket.max_major_overflows; i++) {
      uint32_t value;
      assert(bucket.FindMajorOverflow(i, &value, OFP(Hasher()(i))));
      assert(!bucket.FindMajorOverflow(i+1, &value, OFP(Hasher()(i))));
      assert(value == i);
    }

    for (int i = 0; i < bucket.max_major_overflows; i += 2) {
      assert(bucket.EraseMajorOverflow(i, OFP(Hasher()(i))));
      assert(!bucket.FindMajorOverflow(i, nullptr, OFP(Hasher()(i))));
    }

    for (int i = bucket.max_major_overflows - 2; i >= 0; i -= 2) {
      uint32_t value;
      assert(bucket.AppendMajorOverflow(i, i*2, OFP(Hasher()(i))));
      assert(bucket.FindMajorOverflow(i, &value, OFP(Hasher()(i))));
      assert(value == i*2);
    }

    printf("[PASSED]\n");
  }

  static void TestStashBucketInsertMajorOverflow() {
    printf("[TEST STASH BUCKET INSERT MAJOR OVERFLOW]\n");

    DleftType::StashBucket bucket;

    for (int i = 0; i < bucket.max_major_overflows; i++) {
      assert(bucket.AppendMajorOverflow(i, i, OFP(Hasher()(i))));
    }
    for (int i = 0; i < bucket.max_major_overflows; i += 2) {
      assert(bucket.InsertMajorOverflow(i, i*2, OFP(Hasher()(i))));
    }
    for (int i = 0; i < bucket.max_major_overflows; i += 2) {
      uint32_t value;
      assert(bucket.FindMajorOverflow(i, &value, OFP(Hasher()(i))));
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

    DleftType::Bucket bucket;

    for (int i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Append(i, i, OFP(Hasher()(i)), nullptr));
    }
    assert(!bucket.Append(2023u, 2023u, OFP(Hasher()(2023u)), nullptr));

    printf("[PASSED]\n");
  }

  static void TestBucketErase() {
    printf("[TEST BUCKET ERASE]\n");

    DleftType::Bucket bucket;

    for (int i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Append(i, i, OFP(Hasher()(i)), nullptr));
    }

    for (int i = 0; i < bucket.bucket_capacity; i += 2) {
      assert(bucket.Erase(i, OFP(Hasher()(i)), nullptr));
      assert(!bucket.Erase(i, OFP(Hasher()(i)), nullptr));
    }

    printf("[PASSED]\n");
  }

  static void TestBucketFind() {
    printf("[TEST BUCKET FIND]\n");

    DleftType::Bucket bucket;

    for (int i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Append(i, i, OFP(Hasher()(i)), nullptr));
    }

    for (int i = 0; i < bucket.bucket_capacity; i += 2) {
      uint32_t value;
      assert(bucket.Find(i, &value, OFP(Hasher()(i)), nullptr));
      assert(value == i);
      assert(bucket.Erase(i, OFP(Hasher()(i)), nullptr));
      assert(!bucket.Find(i, nullptr, OFP(Hasher()(i)), nullptr));
      assert(bucket.Append(i, i*2, OFP(Hasher()(i)), nullptr));
    }

    for (int i = 0; i < bucket.bucket_capacity; i++) {
      uint32_t value;
      assert(bucket.Find(i, &value, OFP(Hasher()(i)), nullptr));
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

    DleftType::Bucket bucket;

    for (int i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Append(i, i, OFP(Hasher()(i)), nullptr));
    }

    for (int i = 0; i < bucket.bucket_capacity; i += 2) {
      assert(bucket.Insert(i, i*2, OFP(Hasher()(i)), nullptr));
    }

    for (int i = 0; i < bucket.bucket_capacity; i += 2) {
      uint32_t value;
      assert(bucket.Find(i, &value, OFP(Hasher()(i)), nullptr));
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

    DleftType::Bucket bucket;
    DleftType::StashBucket stash_bucket;

    for (int i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Append(i, i, OFP(Hasher()(i)), &stash_bucket));
    }
    assert(bucket.overflow_count_ == 0);

    for (int i = bucket.bucket_capacity; i < bucket.bucket_capacity + bucket.max_minor_overflows; i++) {
      assert(bucket.Append(i, i, OFP(Hasher()(i)), &stash_bucket));
    }
    assert(bucket.overflow_count_ == bucket.max_minor_overflows);
    assert(bucket.GetMinorOverflowCount() == bucket.max_minor_overflows);
    for (int i = 0; i < stash_bucket.max_major_overflows; i++) {
      assert(stash_bucket.position_[i] == stash_bucket.invalid_pos);
    }

    for (int i = bucket.bucket_capacity + bucket.max_minor_overflows;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i++) {
      assert(bucket.Append(i, i, OFP(Hasher()(i)), &stash_bucket));
    }
    assert(bucket.overflow_count_ == bucket.max_minor_overflows + stash_bucket.max_major_overflows);
    assert(bucket.GetMinorOverflowCount() == bucket.max_minor_overflows);
    for (int i = 0; i < stash_bucket.max_major_overflows; i++) {
      assert(stash_bucket.position_[i] != stash_bucket.invalid_pos);
    }
  
    assert(!bucket.Append(2023u, 2023u, OFP(Hasher()(2023u)), &stash_bucket));

    printf("[PASSED]\n");
  }

  static void TestBucketEraseWithOverflow() {
    printf("[TEST BUCKET ERASE WITH OVERFLOW]\n");

    DleftType::Bucket bucket;
    DleftType::StashBucket stash_bucket;

    for (int i = 0;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i++) {
      assert(bucket.Append(i, i, OFP(Hasher()(i)), &stash_bucket));
    }
    assert(bucket.overflow_count_ == bucket.max_minor_overflows + stash_bucket.max_major_overflows);
    assert(bucket.GetMinorOverflowCount() == bucket.max_minor_overflows);
    for (int i = 0; i < stash_bucket.max_major_overflows; i++) {
      assert(stash_bucket.position_[i] != stash_bucket.invalid_pos);
    }

    for (int i = 0; i < bucket.bucket_capacity; i++) {
      assert(bucket.Erase(i, OFP(Hasher()(i)), &stash_bucket));
      assert(!bucket.Erase(i, OFP(Hasher()(i)), &stash_bucket));
    }
    assert(bucket.GetSize() == 0);

    for (int i = bucket.bucket_capacity + bucket.max_minor_overflows;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i++) {
      assert(bucket.Erase(i, OFP(Hasher()(i)), &stash_bucket));
      assert(!bucket.Erase(i, OFP(Hasher()(i)), &stash_bucket));
    }
    assert(bucket.overflow_count_ == bucket.max_minor_overflows);
    assert(bucket.GetMinorOverflowCount() == bucket.max_minor_overflows);
    for (int i = 0; i < stash_bucket.max_major_overflows; i++) {
      assert(stash_bucket.position_[i] == stash_bucket.invalid_pos);
    }

    for (int i = bucket.bucket_capacity; i < bucket.bucket_capacity + bucket.max_minor_overflows; i++) {
      assert(bucket.Erase(i, OFP(Hasher()(i)), &stash_bucket));
      assert(!bucket.Erase(i, OFP(Hasher()(i)), &stash_bucket));
    }
    assert(bucket.overflow_count_ == 0);
    assert(bucket.GetMinorOverflowCount() == 0);

    printf("[PASSED]\n");
  }

  static void TestBucketFindWithOverflow() {
    printf("[TEST BUCKET FIND WITH OVERFLOW]\n");

    DleftType::Bucket bucket;
    DleftType::StashBucket stash_bucket;

    for (int i = 0;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i++) {
      assert(bucket.Append(i, i, OFP(Hasher()(i)), &stash_bucket));
    }
    assert(bucket.overflow_count_ == bucket.max_minor_overflows + stash_bucket.max_major_overflows);
    assert(bucket.GetMinorOverflowCount() == bucket.max_minor_overflows);
    for (int i = 0; i < stash_bucket.max_major_overflows; i++) {
      assert(stash_bucket.position_[i] != stash_bucket.invalid_pos);
    }

    for (int i = 0;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i++) {
      uint32_t value;
      assert(bucket.Find(i, &value, OFP(Hasher()(i)), &stash_bucket));
      assert(value == i);
      
      assert(bucket.Erase(i, OFP(Hasher()(i)), &stash_bucket));
      assert(!bucket.Find(i, nullptr, OFP(Hasher()(i)), &stash_bucket));

      assert(bucket.Append(i, i*2, OFP(Hasher()(i)), &stash_bucket));
      assert(bucket.Find(i, &value, OFP(Hasher()(i)), &stash_bucket));
      assert(value == i*2);
    }

    printf("[PASSED]\n");
  }

  static void TestBucketInsertWithOverflow() {
    printf("[TEST BUCKET INSERT WITH OVERFLOW]\n");

    DleftType::Bucket bucket;
    DleftType::StashBucket stash_bucket;

    for (int i = 0;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i++) {
      assert(bucket.Append(i, i, OFP(Hasher()(i)), &stash_bucket));
    }

    for (int i = 0;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i += 2) {
      assert(bucket.Insert(i, i*2, OFP(Hasher()(i)), &stash_bucket));
    }

    for (int i = 0;
         i < bucket.bucket_capacity + bucket.max_minor_overflows + stash_bucket.max_major_overflows;
         i += 2) {
      uint32_t value;
      assert(bucket.Find(i, &value, OFP(Hasher()(i)), &stash_bucket));
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
    DleftType hash_table(testcase_size);

    for (int i = 0; i < testcase_size; i++) {
      assert(hash_table.Append(i, i, Hasher()(i)));
      assert(hash_table.size_ == i + 1);
    }

    printf("[PASSED]\n");
  }

  static void TestDleftErase() {
    printf("[TEST DLEFT ERASE]\n");

    const int testcase_size = 60000;
    DleftType hash_table(testcase_size);

    for (int i = 0; i < testcase_size; i++) {
      assert(hash_table.Append(i, i, Hasher()(i)));
    }
    assert(hash_table.size_ == testcase_size);

    for (int i = 0; i < testcase_size; i += 2) {
      assert(hash_table.Erase(i, Hasher()(i)));
      assert(hash_table.size_ == testcase_size - i / 2 - 1);
    }

    for (int i = 0; i < testcase_size; i += 2) {
      assert(!hash_table.Erase(i, Hasher()(i)));
      assert(hash_table.size_ == testcase_size / 2);
    }

    printf("[PASSED]\n");
  }

  static void TestDleftFind() {
    printf("[TEST DLEFT FIND]\n");

    const int testcase_size = 60000;
    DleftType hash_table(testcase_size);

    for (int i = 0; i < testcase_size; i++) {
      assert(hash_table.Append(i, i, Hasher()(i)));
    }

    for (int i = 0; i < testcase_size; i++) {
      uint32_t value;
      assert(hash_table.Find(i, &value, Hasher()(i)));
      assert(value == i);
    }

    for (int i = 0; i < testcase_size; i += 2) {
      assert(hash_table.Erase(i, Hasher()(i)));
      assert(hash_table.Append(i, i*2, Hasher()(i)));
    }

    for (int i = 0; i < testcase_size; i++) {
      uint32_t value;
      assert(hash_table.Find(i, &value, Hasher()(i)));
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
    DleftType hash_table(testcase_size);

    for (int i = 0; i < testcase_size; i++) {
      assert(hash_table.Append(i, i, Hasher()(i)));
    }

    for (int i = 0; i < testcase_size; i += 2) {
      assert(hash_table.Insert(i, i*2, Hasher()(i)) == DleftType::InsertStatus::EXISTED);
    }

    for (int i = 0; i < testcase_size; i++) {
      uint32_t value;
      assert(hash_table.Find(i, &value, Hasher()(i)));
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
    DleftType hash_table(testcase_size);

    for (int i = 0; i < testcase_size; i++) {
      assert(hash_table.Append(i, i, Hasher()(i)));
    }

    for (int i = 0; i < testcase_size; i++) {
      uint32_t value;
      assert(hash_table.Find(i, &value, Hasher()(i)));
      assert(value == i);
    }

    assert(hash_table.Resize(testcase_size * 2));

    for (int i = 0; i < testcase_size; i++) {
      uint32_t value;
      assert(hash_table.Find(i, &value, Hasher()(i)));
      assert(value == i);
    }

    printf("[PASSED]\n");
  }

	static void TestDleftFalsePositives() {
	 #ifdef __COUNT_FALSE_POSITIVES__
	 	const int testcase_size = 1000000;
    DleftType hash_table(testcase_size);

		for (int i = 0; i < testcase_size; i++) {
			hash_table.insert(i, i);
		}

		false_positive = 0;
		for (int i = 0; i < testcase_size; i++) {
			uint32_t value;
			hash_table.find(i, value);
		}
		printf("Positive Read: %lu false positives, %lu of which occured on overflows\n",
					 false_positive, overflow_false_positives);

		false_positive = 0;
		for (int i = testcase_size; i < testcase_size * 2; i++) {
			uint32_t value;
			hash_table.find(i, value);
		}
		printf("Negative Read: %lu false positives, %lu of which occured on overflows\n",
					 false_positive, overflow_false_positives);
	 #endif
	}

	static void TestDleftMaxLoadFactor() {
    DleftType hash_table(1000000);
    size_t bucket_total{0}, stash_bucket_total{0};

	 #ifdef __COUNT_OVERFLOWS__
	 	minor_overflows = major_overflows = 0;
	 #endif

    uint32_t key = 0;
    while (hash_table.Append(std::forward<uint32_t>(key), std::forward<uint32_t>(key), Hasher()(key))) {
      key++;
    }

    std::map<size_t, size_t> bucket_distribution, stash_bucket_distribution;
    for (size_t i = 0; i < hash_table.num_buckets_; i++) {
      bucket_distribution[hash_table.buckets_[i].GetTotal()]++;
      bucket_total += hash_table.buckets_[i].GetSize();
    }
    for (size_t i = 0; i < hash_table.num_stash_buckets_; i++) {
      stash_bucket_distribution[hash_table.stash_buckets_[i].GetSize()]++;
      stash_bucket_total += hash_table.stash_buckets_[i].GetSize();
    }

    printf("Bucket Distribution (Total: %u):\n", hash_table.num_buckets_);
    for (auto &p : bucket_distribution) {
      printf("%lu:%lu,", p.first, p.second);
    }
    printf("\nStash Bucket Distribution (Total: %u):\n", hash_table.num_stash_buckets_);
    for (auto &p : stash_bucket_distribution) {
      printf("%lu:%lu,", p.first, p.second);
    }
    printf("\nBucket Load Factor: %lf, Stash Bucket Load Factor: %lf\n",
           1.0 * bucket_total / hash_table.BucketCapacity(),
           1.0 * stash_bucket_total / hash_table.StashBucketCapacity());

	 #ifdef __COUNT_OVERFLOWS__
		printf("# of Minor Overflows: %ld\n", minor_overflows);
		printf("# of Major Overflows: %ld\n", major_overflows);
	 #endif
	}
};
#endif

#undef __DEBUG_DLEFT__
#undef DEBUG_DLEFT
#undef UNLIKELY
#undef LIKELY
#undef OFP
#undef FP
#undef GET_BIT
#undef SET_BIT
#undef CLEAR_BIT
#undef ROUNDUP_POWER_2
#undef BYTE_ROUND_UP
#undef BUCKET_STASH_BUCKET_RATIO
#undef CACHELINE_SIZE
