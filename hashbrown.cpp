// Hash Brown
// Should move functions to header file and inline most of them
#include <stdint.h>

const uint64_t P1 = 0x8ebc6af09c88c6e3;
const uint64_t P2 = 0xe7037ed1a0b428db;
const uint64_t P3 = 0x1d8e4e27c47d124f;
// const uint64_t P4 = 0xa0761d6478bd642f;
// const uint64_t P5 = 0x589965cc75374cc3;

uint64_t hashbrown(uint64_t seed, size_t length, void* data) {
    uint64_t* a;
    uint64_t* b;

    seed = seed ^ P1;

    if (length == 0) {
        return seed;
    }

    if (length < 4) {
        *a = *reinterpret_cast<uint8_t*>(data);
        *a |= *(reinterpret_cast<uint8_t*>(data) + (length >> 1)) << 8;
        *a |= *(reinterpret_cast<uint8_t*>(data) + (length-1)) << 16;
    }
    else if (length == 4) {
        // IP addresses are exactly 32 bits
        *a = read4(data);
        *b = *a;
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
    else {
        // todo: parallel hashing, inspired by xxhash
        *a = 123;
        *b = 456;
    }

    uint64_t first_mix = mix(*a ^ P2, *b ^ seed);
    return mix(seed ^ P3, first_mix);
}

/* 
    Multiplies two 64-bit numbers and returns high and low parts of the 128-bit result
    Stores the results in the passed in parameters
*/
void mult64(const uint64_t a, const uint64_t b, uint64_t *hi_ptr, uint64_t *lo_ptr) {
    // Low is just multiplication of a and b (truncates for us)
    *lo_ptr = a * b;

    uint64_t a_lo = (uint32_t) a;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = (uint32_t) b;
    uint64_t b_hi = b >> 32;

    // uint64_t a_x_b_hi =  a_hi * b_hi;
    // uint64_t a_x_b_mid = a_hi * b_lo;
    // uint64_t b_x_a_mid = b_hi * a_lo;
    // uint64_t a_x_b_lo =  a_lo * b_lo;
    // *hi_ptr = a_x_b_hi + (a_x_b_mid >> 32) + (b_x_a_mid >> 32);
    
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
