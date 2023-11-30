#pragma once

#include "Types.h"

void BulkSpeedTest ( pfHash hash, uint32_t seed );
double TinySpeedTest ( pfHash hash, int hashsize, int keysize, uint32_t seed, bool verbose );
double HashMapSpeedTest ( pfHash pfhash, int hashbits, std::vector<std::string> words,
                          const uint32_t seed, const int trials, bool verbose );

void BulkSpeedTestCustom ( pfHash hash, uint32_t seed, string input_filepath );
double HashMapSpeedTestCustom ( pfHash pfhash, const int hashbits, string input_filepath,
                          const uint32_t seed, const int trials, bool verbose );
//-----------------------------------------------------------------------------
