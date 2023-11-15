#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include "hashbrown.cpp"
using namespace std;

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
    for (auto num_ptr : integers) {
        uint64_t hashed_num = hashbrown(seed, 32, num_ptr);
        table[hashed_num % table_size].push_back(hashed_num);
    }

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
}

int main()
{
    test_ips("data/random_ips.txt");
    return 0;
}
