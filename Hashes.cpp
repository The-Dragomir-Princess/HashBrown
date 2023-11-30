#define _HASHES_CPP
#include "Hashes.h"
#include "Random.h"

#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

// ----------------------------------------------------------------------------
//fake / bad hashes

// objsize: 0x2f-0x0: 47
void
BadHash(const void *key, int len, uint32_t seed, void *out)
{
  uint32_t	  h = seed;
  const uint8_t  *data = (const uint8_t *)key;
  const uint8_t *const end = &data[len];

  while (data < end) {
    h ^= h >> 3;
    h ^= h << 5;
    h ^= *data++;
  }

  *(uint32_t *) out = h;
}

// objsize: 0x19b-0x30: 363
void
sumhash(const void *key, int len, uint32_t seed, void *out)
{
  uint32_t	 h = seed;
  const uint8_t *data = (const uint8_t *)key;
  const uint8_t *const end = &data[len];

  while (data < end) {
    h += *data++;
  }

  *(uint32_t *) out = h;
}

// objsize: 0x4ff-0x1a0: 863
void
sumhash32(const void *key, int len, uint32_t seed, void *out)
{
  uint32_t	  h = seed;
  const uint32_t *data = (const uint32_t *)key;
  const uint32_t *const end = &data[len/4];

  while (data < end) {
    h += *data++;
  }
  if (len & 3) {
    uint8_t *dc = (uint8_t*)data; //byte stepper
    const uint8_t *const endc = &((const uint8_t*)key)[len];
    while (dc < endc) {
      h += *dc++ * UINT64_C(11400714819323198485);
    }
  }

  *(uint32_t *) out = h;
}

// objsize: 0x50d-0x500: 13
void
DoNothingHash(const void *, int, uint32_t, void *)
{
}

// objsize: 0x53f-0x510: 47
void
NoopOAATReadHash(const void *key, int len, uint32_t seed, void *out)
{
  uint32_t	 h = seed;
  const uint8_t *data = (const uint8_t *)key;
  const uint8_t *const end = &data[len];

  while (data < end) {
    h = *data++;
  }
  *(uint32_t *) out = h;
}

//-----------------------------------------------------------------------------
