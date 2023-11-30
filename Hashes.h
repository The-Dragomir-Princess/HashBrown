#pragma once

#include "Platform.h"
#include "Types.h"
//#include <algorithm>

//----------
// These are _not_ hash functions (even though people tend to use crc32 as one...)

static inline bool BadHash_bad_seeds(std::vector<uint32_t> &seeds)
{
  seeds = std::vector<uint32_t> { UINT32_C(0) };
  return true;
}
void BadHash(const void *key, int len, uint32_t seed, void *out);
static inline bool sumhash_bad_seeds(std::vector<uint32_t> &seeds)
{
  seeds = std::vector<uint32_t> { UINT32_C(0) };
  return true;
}
void sumhash(const void *key, int len, uint32_t seed, void *out);
void sumhash32(const void *key, int len, uint32_t seed, void *out);

void DoNothingHash(const void *key, int len, uint32_t seed, void *out);
void NoopOAATReadHash(const void *key, int len, uint32_t seed, void *out);
void crc32(const void *key, int len, uint32_t seed, void *out);

static inline bool crc32c_bad_seeds(std::vector<uint32_t> &seeds)
{
  seeds = std::vector<uint32_t> { UINT32_C(0x111c2232) };
  return true;
}
//----------
// General purpose hashes

//---- SuperFastHash
#include "superfasthash.h"
inline void SuperFastHash_test(const void *key, int len, uint32_t seed, void *out) {
  *(uint64_t*)out = (uint64_t) SuperFastHash((const char*)key, len, seed);
}

//---- xxHash32 and xxHash64
#define XXH_INLINE_ALL
#include "xxhash.h"
inline void xxHash32_test( const void * key, int len, uint32_t seed, void * out ) {
  // objsize 10-104 + 3e0-5ce: 738
  *(uint32_t*)out = (uint32_t) XXH32(key, (size_t) len, (unsigned) seed);
}
inline void xxHash64_test( const void * key, int len, uint32_t seed, void * out ) {
  // objsize 630-7fc + c10-1213: 1999
  *(uint64_t*)out = (uint64_t) XXH64(key, (size_t) len, (unsigned long long) seed);
}

//---- WyHash
// native 32bit. objsize: 8055230 - 80553da: 426
#include "wyhash32.h"
inline void wyhash32_test (const void * key, int len, uint32_t seed, void * out) {
  *(uint32_t*)out = wyhash32(key, (uint64_t)len, (unsigned)seed);
}
#include "wyhash.h"
// objsize 40dbe0-40ddba: 474
inline void wyhash_test (const void * key, int len, uint32_t seed, void * out) {
  *(uint64_t*)out = wyhash(key, (uint64_t)len, (uint64_t)seed, _wyp);
}
// objsize: 40da00-40dbda: 474
inline void wyhash32low (const void * key, int len, uint32_t seed, void * out) {
  *(uint32_t*)out = 0xFFFFFFFF & wyhash(key, (uint64_t)len, (uint64_t)seed, _wyp);
}

// HashBrown
#include "hashbrown.h"
inline void hashbrown_test (const void *key, int len, uint32_t seed, void *out) {
  *(uint64_t*)out = (uint64_t) hashbrown((unsigned long long) seed, (size_t) len, (void*)key);
}

//---- Lookup3
#include "lookup3.h"
inline void lookup3_test (const void *key, int len, uint32_t seed, void *out) {
  *(uint64_t*)out = (uint64_t) lookup3((const char*)key, (uint32_t)len, (uint32_t)seed);
}

//--- APHash (APartow Hash)
#include "aphash.h"
inline void aphash_test (const void *key, int len, uint32_t seed, void *out) {
  *(uint64_t*)out = (uint64_t) APHash((const char*)key);
}
