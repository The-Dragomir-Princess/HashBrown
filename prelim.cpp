#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

class HashTable {
private:
    std::vector<int> table;
    int capacity;

public:
    HashTable(int size) {
        capacity = size;
        table.resize(capacity, -1);
    }

    // Hash function 1
    int hashFunction1(int key) {
        return key % capacity;
    }

    // Hash function 2
    int hashFunction2(int key) {
        // Use a prime number smaller than the table size
        return 7 - (key % 7);
    }

    // Insert a key into the hash table
    void insert(int key) {
        int index = hashFunction1(key);
        int step = hashFunction2(key);

        while (table[index] != -1) {
            // Collision occurred, use the second hash function to find the next position
            index = (index + step) % capacity;
        }

        // Insert the key into the found position
        table[index] = key;
    }

    // Search for a key in the hash table
    bool search(int key) {
        int index = hashFunction1(key);
        int step = hashFunction2(key);

        while (table[index] != -1) {
            if (table[index] == key) {
                // Key found
                return true;
            }

            // Move to the next position using the second hash function
            index = (index + step) % capacity;
        }

        // Key not found
        return false;
    }
};

void readFile(const std::string& filename) {
    // Open the file
    std::ifstream file(filename);

    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Read and output each line from the file
    std::string line;
    while (std::getline(file, line)) {
        std::cout << line << std::endl;
    }

    // Close the file
    file.close();
}

int main() {
    HashTable hashTable(10);

    // Insert keys into the hash table
    hashTable.insert(5);
    hashTable.insert(15);
    hashTable.insert(25);

    // Search for keys in the hash table
    std::cout << "Search for key 15: " << (hashTable.search(15) ? "Found" : "Not Found") << std::endl;
    std::cout << "Search for key 10: " << (hashTable.search(10) ? "Found" : "Not Found") << std::endl;

    return 0;
}

