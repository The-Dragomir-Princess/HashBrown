#include <iostream>
#include <string>
#include "hashbrown.cpp"
using namespace std;

int main()
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

    return 0;
}
