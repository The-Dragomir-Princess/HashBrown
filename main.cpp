#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include "hashbrown.h"
#include "xxhash.c"
using namespace std;
//using namespace std::chrono;

void simple_test_run()
{
    cout << "Running..." << endl;
    uint64_t seed = 1609587929392839161ULL;
    
    string temp = "Hello World! Let's hash.";
    void* tempPtr = static_cast<void*>(const_cast<char*>(temp.c_str()));

    cout << "trying to small hash " << temp << " | Size " << temp.size() << endl;
    auto res = hashbrown(seed, temp.size(), tempPtr);
    cout << "Resultant Hash is " << res << endl;

    temp = "The quick brown fox jumps over the lazy dog and Pack my box with five dozen liquor jugs.";
    tempPtr = static_cast<void*>(const_cast<char*>(temp.c_str()));

    cout << "trying to big hash " << temp << " | Size " << temp.size() << endl;
    res = hashbrown(seed, temp.size(), tempPtr);
    cout << "Resultant Hash is " << res << endl;
}

void test_ips(string path)
{
    cout << "Results for " << path << endl;
    ifstream inputFile(path);
    if (!inputFile.is_open()) {
        cerr << "Could not open the file." << endl;
        return;
    }
    vector<uint32_t*> integers; // Vector to store the 32-bit integers
    string line;
    while (getline(inputFile, line)) {
        uint32_t num = (uint32_t) stol(line);
        integers.push_back(&num);
    }
    inputFile.close(); // Close the file after reading

    // Hash table to store elements in. "Linked list" collision 
    const int table_size = 65535;
    vector<int> table[table_size];

    // Let's hash
    uint64_t seed = 18446744073709551557;
    auto start = chrono::high_resolution_clock::now();
    for (auto num_ptr : integers) {
        uint64_t hashed_num = hashbrown(seed, 32, num_ptr);
        //uint64_t hashed_num = XXH64(num_ptr, 32, seed);
        table[hashed_num % table_size].push_back(hashed_num);
    }
    cout << "Computing hashes took " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start).count() << " ms" << endl;

    // Let's find some statistics
    // Note: we round up b/c to allow for remainder items
    int expected_per_bkt = (integers.size() + table_size - 1) / table_size;
    int total_bkts_overflowed = 0;
    for (auto bkt : table) {
        if (bkt.size() > expected_per_bkt) {
            total_bkts_overflowed++;
            //cout << "Overflowed by " << bkt.size() - expected_per_bkt << endl;
        }
    }
    cout << "Number of buckets: " << table_size << endl;
    cout << "Number of items inserted: " << integers.size() << endl;
    cout << "Buckets overflowed: " << total_bkts_overflowed << endl;
    cout << endl;
}

void test_hash_time(string path) {
    cout << "Hash Time Results for " << path << endl;
    ifstream inputFile(path);
    if (!inputFile.is_open()) {
        cerr << "Could not open the file." << endl;
        return;
    }
    vector<uint32_t*> integers; // Vector to store the 32-bit integers
    string line;
    while (getline(inputFile, line)) {
        uint32_t num = (uint32_t) stol(line);
        integers.push_back(&num);
    }
    inputFile.close(); // Close the file after reading

    // Let's hash
    uint64_t seed = 18446744073709551557;
    auto start = chrono::high_resolution_clock::now();
    for (auto num_ptr : integers) {
        hashbrown(seed, 32, num_ptr);
        //XXH64(num_ptr, 32, seed);
        //wyhash(num_ptr, 32);
    }
    auto dur = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start).count();
    cout << "Computing hashes took " << dur << " ms" << endl;
    cout << "Average " << (float) dur / (float) integers.size() << " ms per hash" << endl;
}

int main()
{
    test_hash_time("data/random_ips.txt");
    return 0;
}
