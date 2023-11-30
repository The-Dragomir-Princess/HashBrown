// Hash Brown
typedef unsigned long long hb_uint64_t;
typedef unsigned int hb_uint32_t;
typedef unsigned short hb_uint16_t;
typedef unsigned char hb_uint8_t;

// Whether or not to inline the functions
#define HB_HAS_INLINE 1

#ifdef HB_HAS_INLINE
#   define HB_INLINE inline
#endif

// Whether or not to pure the functions
#define HB_IS_PURE 1

#if defined (HB_IS_PURE) && defined (__GNUC__)
#   define HB_PURE static __attribute__((pure)) 
#else
#   define HB_PURE static
#endif

// Whether or not to const the functions
#define HB_IS_CONST 1

#if defined (HB_IS_CONST) && defined (__GNUC__)
#   define HB_CONST static __attribute__((const)) 
#else
#   define HB_CONST static const
#endif

#define P1 0x8ebc6af09c88c6e3
#define P2 0xe7037ed1a0b428db
#define P3 0x1d8e4e27c47d124f
#define P4 0xa0761d6478bd642f
#define P5 0x589965cc75374cc3

/*
    Mixes the two input integers by multiplying then  
    XORing the two halves together
*/
HB_INLINE HB_CONST hb_uint64_t mix(hb_uint64_t a, hb_uint64_t b) {
    hb_uint64_t a_lo = (hb_uint32_t) a;
    hb_uint64_t a_hi = a >> 32;
    hb_uint64_t b_lo = (hb_uint32_t) b;
    hb_uint64_t b_hi = b >> 32;
    
    // Mult64 and then mix, also a*b returns just the least sig 64 bits
    return (hb_uint64_t) (a * b) ^ ((a_hi * b_hi) + ((a_hi * b_lo) >> 32) + ((b_hi * a_lo) >> 32));
}

/*
    Reads 32 bits in little-endian fashion
*/
HB_INLINE HB_PURE hb_uint32_t read4(const void *data) {
    return *static_cast<const hb_uint32_t*>(data);
}

/*
    Reads 64 bits in little-endian fashion
*/
HB_INLINE HB_PURE hb_uint64_t read8(const void *data) {
    return *static_cast<const hb_uint64_t*>(data);
}

// Rotates input left by specified bits amount
HB_INLINE HB_CONST hb_uint64_t rotLeft(hb_uint64_t input, unsigned char amount) {
    return (input << amount) | (input >> (64 - amount));
}

/*
    Conducts a round of hashing
    Uses 4 parallel states of 64-bits each
*/
HB_INLINE void hash_round(const void *data, hb_uint64_t& state0, hb_uint64_t& state1, hb_uint64_t& state2, hb_uint64_t& state3) {
    const hb_uint64_t *block = (const hb_uint64_t*) data;
    state0 = rotLeft(state0 + block[0] * P4, 31) * P5;
    state1 = rotLeft(state1 + block[1] * P4, 31) * P5;
    state2 = rotLeft(state2 + block[2] * P4, 31) * P5;
    state3 = rotLeft(state3 + block[3] * P4, 31) * P5;
}
// |------------------------------------------------------------------------------------| //

/*
    Hashes small inputs. Length must be < 32 bytes.
*/
HB_INLINE hb_uint64_t hashbrownsmall(hb_uint64_t seed, size_t length, void *data) {
    hb_uint64_t a;
    hb_uint64_t b;

    seed ^= P1;

    switch(length) {
        case 0:
            return seed;
            break;
        case 1:
        case 2:
        case 3:
            // case len < 4
            a = *reinterpret_cast<hb_uint8_t*>(data);
            a |= *(reinterpret_cast<hb_uint8_t*>(data) + (length >> 1)) << 8;
            a |= *(reinterpret_cast<hb_uint8_t*>(data) + (length - 1)) << 16;
            break;
        case 4:
            // len == 4
            a = read4(data);
            b = a;
            break;
        case 5:
        case 7:
            // len < 8
            a = read4(data);
            b = read4(reinterpret_cast<unsigned char*>(data) + (length - 4));
            break;
        case 6:
            // len == 6 (mac)
            a = read4(data);
            b = *reinterpret_cast<hb_uint16_t*>(reinterpret_cast<unsigned char*>(data) + 4);
            break;
        case 8:
            // len == 8
            a = read8(data);
            b = a;
            break;
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
            // len <= 16
            a = read8(data);
            b = read8(reinterpret_cast<unsigned char*>(data) + (length - 8));
            break;
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
        case 24:
            // len <= 24
            a = read8(data) * P1 ^ read8(reinterpret_cast<unsigned char*>(data) + 8);
            b = read8(reinterpret_cast<unsigned char*>(data) + (length - 8));
            break;
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
            // len < 32
            a = (read8(data) * P1) ^ read8(reinterpret_cast<unsigned char*>(data) + 8);
            b = read8(reinterpret_cast<unsigned char*>(data) + 16) + read8(reinterpret_cast<unsigned char*>(data) + (length - 8));
            break;
    }

    hb_uint64_t first_mix = mix(a ^ P2, b ^ seed);
    return mix(first_mix, seed ^ P3);
}

HB_INLINE hb_uint64_t hashbrownbig(hb_uint64_t seed, size_t length, void *input) {
    hb_uint64_t state[4];
    state[0] = seed + P1 + P2;
    state[1] = seed + P3;
    state[2] = seed;
    state[3] = seed - P1;

    const unsigned char *data = (const unsigned char*) input;

    // run hash to find result
    hb_uint64_t res;
    res = rotLeft(state[0], 1) + rotLeft(state[1], 7) + rotLeft(state[2], 12) + rotLeft(state[3], 18);
    int round_count = length / 32;
    for (; round_count > 0; round_count--) {
        hash_round(data, state[0], state[1], state[2], state[3]);
        data += 32;
    }
    res = (res ^ state[0]) * P1 + P5;
    res = (res ^ state[1]) * P2 + P5;
    res = (res ^ state[2]) * P3 + P5;
    res = (res ^ state[3]) * P4 + P5;

    // Deal with remainder bytes. This last part *might* be slow. 
    // If so, will need to figure out a way to speed it up. 
    hb_uint64_t remainder = hashbrownsmall(seed, length % 32, (void *) data);

    remainder ^= (remainder >> 33) * P2;
    res = mix(res, remainder);
    return res;
}

// Main frontend function for hashing
hb_uint64_t hashbrown(hb_uint64_t seed, size_t length, void* data) {
    if (length < 32) {
        return hashbrownsmall(seed, length, data);
    }

    return hashbrownbig(seed, length, data);
}
