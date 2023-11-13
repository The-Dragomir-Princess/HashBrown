#include "City.h"
#include "farmhash.h"
#include "MurmurHash3.h"
#include "xxhash.h"

#include <stdint.h>

typedef uint32_t (*hash32_t)(uint32_t);

auto cityhash32(uint32_t key) -> uint32_t;

auto farmhash32(uint32_t key) -> uint32_t;

auto murmurhash32(uint32_t key) -> uint32_t;

auto xxhash32(uint32_t key) -> uint32_t;