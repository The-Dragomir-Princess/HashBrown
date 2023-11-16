#pragma once

#include <stdint.h>

#include <immintrin.h>

#include <cassert>
#include <cstring>

#define __TEST_DLEFT__

#define CACHELINE_SIZE (64)

#define BYTE_ROUND_UP(n) ((n + 7) / 8)
#define GET_BIT(bits, n)   (reinterpret_cast<uint8_t *>(bits)[n/8] & (1 << (n % 8)))
#define SET_BIT(bits, n)   (reinterpret_cast<uint8_t *>(bits)[n/8] |= (1 << (n % 8)))
#define CLEAR_BIT(bits, n) (reinterpret_cast<uint8_t *>(bits)[n/8] &= ~(1 << (n % 8)))

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
  DleftFpStash(size_t = 0, bool = true);

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

  struct Bucket {
    static constexpr size_t header_size = 32;
    static constexpr int bucket_capacity = 16;
    static constexpr int max_minor_overflows = 4;
    static constexpr size_t buf_size = CACHELINE_SIZE - header_size;
    static constexpr int buf_capacity = buf_size / tuple_size;

    // header
    uint8_t fingerprint_[bucket_capacity + max_minor_overflows];
    uint8_t validity_[BYTE_ROUND_UP(bucket_capacity + max_minor_overflows)];
		uint8_t overflow_pos_[max_minor_overflows];
		uint16_t stash_bucket_idx_;
		uint8_t num_overflows_{0};
    uint8_t dummy1_[header_size - sizeof(fingerprint_) - sizeof(validity_) - sizeof(overflow_pos_)
										- sizeof(stash_bucket_idx_) - sizeof(num_overflows_)];

    // write buffer (in the same cacheline as header)
    Tuple buf_[buf_capacity];
    uint8_t dummy2_[buf_size - sizeof(buf_)];

    // key-value pairs
    Tuple tuples_[bucket_capacity - buf_capacity];
  } __attribute__((packed));

  struct StashBucket {
    static constexpr size_t header_size = 32;
    static constexpr int major_overflow_capacity = 8;
    static constexpr int bucket_capacity = 64;
		static constexpr uint8_t invalid_pos = 0xff;

    // header
    uint16_t fingerprint_[major_overflow_capacity];
		uint8_t position_[major_overflow_capacity];
    uint8_t validity_[BYTE_ROUND_UP(bucket_capacity)];
    uint8_t dummy_[header_size - sizeof(fingerprint_) - sizeof(position_) - sizeof(validity_)];

		// overflow key-value pairs
    Tuple tuples_[bucket_capacity];

		StashBucket();

    auto InsertMajorOverflow(const KeyType &, const ValueType &, uint16_t) -> bool;

    auto EraseMajorOverflow(const KeyType &, uint16_t) -> bool;

    auto FindMajorOverflow(const KeyType &, ValueType *, uint16_t) const -> bool;

    auto InsertMinorOverflow(const KeyType &, const ValueType &) -> uint8_t;

    auto EraseMinorOverflow(const KeyType &, uint8_t) -> bool;

    auto FindMinorOverflow(const KeyType &, ValueType *, uint8_t) const -> bool;
  } __attribute__((packed));

#ifdef __TEST_DLEFT__
	friend class HashTableTest;
#endif
};

#define DLEFT_TEMPLATE template <class KeyType, class ValueType, class HashFunc1, class HashFunc2>
#define DLEFT_TYPE DleftFpStash<KeyType, ValueType, HashFunc1, HashFunc2>

DLEFT_TEMPLATE
DLEFT_TYPE::StashBucket::StashBucket() {
	memset(validity_, 0, sizeof(validity_));
	memset(position_, invalid_pos, sizeof(position_));
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::InsertMajorOverflow(const KeyType &key, const ValueType &value, uint16_t fp) -> bool {
	int mask;
	uint64_t validity_as_ull;
	uint8_t idx, pos;

	validity_as_ull = *reinterpret_cast<uint64_t *>(validity_);
	if (~validity_as_ull == 0) {
		return false;
	}

	SEARCH_8_128(invalid_pos, position_, mask);
	mask &= 0xff;
	if (mask == 0) {
		return false;
	}

	pos = __builtin_ctzll(~validity_as_ull);
	tuples_[pos] = {key, value};
	SET_BIT(validity_, pos);

	idx = __builtin_ctz(mask);
	position_[idx] = pos;
	fingerprint_[idx] = fp;

	return true;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::EraseMajorOverflow(const KeyType &key, uint16_t fp) -> bool {
	int mask;
	uint8_t idx;

	SEARCH_16_128(fp, fingerprint_, mask);
	while (mask != 0) {
		idx = __builtin_ctz(mask) / 2;
		if (position_[idx] != invalid_pos && tuples_[position_[idx]].key == key) {
			CLEAR_BIT(validity_, position_[idx]);
			position_[idx] = invalid_pos;
			return true;
		}
		mask &= ~(3 << (idx * 2));
	}
	return false;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::FindMajorOverflow(const KeyType &key, ValueType *value, uint16_t fp) const -> bool {
	int mask;
	uint8_t idx;

	SEARCH_16_128(fp, fingerprint_, mask);
	while (mask != 0) {
		idx = __builtin_ctz(mask) / 2;
		if (position_[idx] != invalid_pos && tuples_[position_[idx]].key == key) {
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
	uint64_t validity_as_ull;
	uint8_t pos;

	validity_as_ull = *reinterpret_cast<uint64_t *>(validity_);
	if (~validity_as_ull == 0) {
		return invalid_pos;
	}
	pos = __builtin_ctzll(~validity_as_ull);
	tuples_[pos] = {key, value};
	SET_BIT(validity_, pos);

	return pos;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::EraseMinorOverflow(const KeyType &key, uint8_t pos) -> bool {
	if (tuples_[pos].key == key) {
		CLEAR_BIT(validity_, pos);
		return true;
	}
	return false;
}

DLEFT_TEMPLATE
auto DLEFT_TYPE::StashBucket::FindMinorOverflow(const KeyType &key, ValueType *value, uint8_t pos) const -> bool {
	if (tuples_[pos].key == key) {
		if (value != nullptr) {
			*value = tuples_[pos].value;
		}
		return true;
	}
	return false;
}

#undef DLEFT_TYPE
#undef GET_BIT
#undef SET_BIT
#undef CLEAR_BIT
#undef BYTE_ROUND_UP
#undef CACHELINE_SIZE
