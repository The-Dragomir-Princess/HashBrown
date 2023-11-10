// Hash Brown
// Should move functions to header file and inline most of them
#include <stdint.h>

const uint64_t P1 = 0x8ebc6af09c88c6e3;
const uint64_t P2 = 0xe7037ed1a0b428db;
const uint64_t P3 = 0x1d8e4e27c47d124f;
const uint64_t P4 = 0xa0761d6478bd642f;
const uint64_t P5 = 0x589965cc75374cc3;

uint64_t hashbrown(uint64_t seed, size_t length, void* data) {
    uint64_t* a;
    uint64_t* b;

    seed ^= P1;

    if (length == 0) {
        return seed;
    }

    if (length < 4) {
        *a = *reinterpret_cast<uint8_t*>(data);
        *a |= *(reinterpret_cast<uint8_t*>(data) + (length >> 1)) << 8;
        *a |= *(reinterpret_cast<uint8_t*>(data) + (length - 1)) << 16;
    }
    else if (length == 4) {
        // IP addresses are exactly 32 bits
        *a = read4(data);
        *b = *a;
    }
    else if (length == 6) {
        // MAC addresses are exactly 48 bits = 6 bytes
        *a = read4(data);
        *b = *reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(data) + 4);
    }
    else if (length < 8) {
        *a = read4(data);
        *b = read4(reinterpret_cast<unsigned char*>(data) + (length - 4));
    }
    else if (length == 8) {
        *a = read8(data);
        *b = *a;
    }
    else if (length <= 16) {
        *a = read8(data);
        *b = read8(reinterpret_cast<unsigned char*>(data) + (length - 8));
    }
    else if (length <= 24) {
        *a = (read8(data) * P1) ^ read8(reinterpret_cast<unsigned char*>(data) + 8);
        *b = read8(reinterpret_cast<unsigned char*>(data) + (length - 8));
    }
    else if (length < 32) {
        // TODO: do something
        *a = (read8(data) * P1) ^ read8(reinterpret_cast<unsigned char*>(data) + 8);
        *b = read8(reinterpret_cast<unsigned char*>(data) + 16) + read8(reinterpret_cast<unsigned char*>(data) + (length - 8));
    }
    else {
        // TODO: parallel hashing, inspired by xxhash
        // return hashbrownbig(seed, length, input)
        *a = 123;
        *b = 456;
    }

    uint64_t first_mix = mix(*a ^ P2, *b ^ seed);
    return mix(first_mix, seed ^ P3);
}

uint64_t hashbrownbig(uint64_t seed, size_t length, void *input) {
    const uint64_t MaxBufferSize = 32;
    unsigned char buffer[MaxBufferSize];
    uint64_t bufferSize = 0;
    uint64_t total_length = 0;

    uint64_t state[4];
    state[0] = seed + P1 + P2;
    state[1] = seed + P3;
    state[2] = seed;
    state[3] = seed - P1;

    const unsigned char *data = (const unsigned char*) input;

    // run hash to find result
    uint64_t res;
    res = rotLeft64(state[0], 1) + rotLeft64(state[1], 7) + rotLeft64(state[2], 12) + rotLeft64(state[3], 18);
    int round_count = length / 32;
    for (; round_count > 0; round_count--) {
        hash_round(data, state[0], state[1], state[2], state[3]);
        data += 4;
    }
    res = (res ^ state[0]) * P1 + P5;
    res = (res ^ state[1]) * P1 + P5;
    res = (res ^ state[2]) * P1 + P5;
    res = (res ^ state[3]) * P1 + P5;

    // Deal with remainder bytes
    // TODO: cleanup this "recursive" call. This won't loop b/c length is guarenteed 
    // to be < 32, thus one of the if-statements will executed instead.
    uint64_t remainder = hashbrown(seed, length % 32, (void *) data);

    remainder ^= (remainder >> 33) * P2;
    res = mix(res, remainder);
    return res;
}

// Rotates input left by specified bits amount
uint32_t rotLeft64(uint64_t input, unsigned char amount) {
    return (input << amount) | (input >> (64 - amount));
}

/*
    Conducts a round of hashing
    Uses 4 parallel states of 64-bits each
*/
void hash_round(const void *data, uint64_t& state0, uint64_t& state1, uint64_t& state2, uint64_t& state3) {
    const uint64_t *block = (const uint64_t*) data;
    state0 = rotLeft64(state0 + block[0] * P4, 31) * P5;
    state1 = rotLeft64(state1 + block[1] * P4, 31) * P5;
    state2 = rotLeft64(state2 + block[2] * P4, 31) * P5;
    state3 = rotLeft64(state3 + block[3] * P4, 31) * P5;
}

/* 
    Multiplies two 64-bit numbers and returns high and low parts of the 128-bit result
    Stores the results in the passed in parameters
*/
void mult64(const uint64_t a, const uint64_t b, uint64_t *hi_ptr, uint64_t *lo_ptr) {
    uint64_t a_lo = (uint32_t) a;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = (uint32_t) b;
    uint64_t b_hi = b >> 32;

    // uint64_t a_x_b_hi =  a_hi * b_hi;
    // uint64_t a_x_b_mid = a_hi * b_lo;
    // uint64_t b_x_a_mid = b_hi * a_lo;
    // uint64_t a_x_b_lo =  a_lo * b_lo;
    // *hi_ptr = a_x_b_hi + (a_x_b_mid >> 32) + (b_x_a_mid >> 32);
    
    // Low is just multiplication of a and b (truncates for us)
    *lo_ptr = a * b;
    *hi_ptr = (a_hi * b_hi) + ((a_hi * b_lo) >> 32) + ((b_hi * a_lo) >> 32);
}

/*
    Mixes the two input integers by multiplying then  
    XORing the two halves together
*/
uint64_t mix(uint64_t a, uint64_t b) {
    uint64_t *res_lo;
    uint64_t *res_hi;
    mult64(a, b, res_hi, res_lo);

    return (uint64_t) *res_lo ^ *res_hi;
}

/*
    Reads 32 bits in little-endian fashion
*/
uint32_t read4(const void* data) {
    const uint8_t* bytePtr = static_cast<const uint8_t*>(data);
    
    uint32_t res = bytePtr[0];
    res |= static_cast<uint32_t>(bytePtr[1]) << 8;
    res |= static_cast<uint32_t>(bytePtr[2]) << 16;
    res |= static_cast<uint32_t>(bytePtr[3]) << 24;

    return res;
}

/*
    Reads 64 bits in little-endian fashion
*/
uint64_t read8(const void* data) {
    const uint8_t* bytePtr = static_cast<const uint8_t*>(data);
    uint64_t res = bytePtr[0];

    for (int i = 1; i < 8; i++) {
        res |= static_cast<uint64_t>(bytePtr[i]) << (i * 8);
    }

    return res;
}
