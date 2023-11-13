cmake_minimum_required(VERSION 3.15)

project(distribution)

add_executable(distribution City.cpp distribution.cpp farmhash.cpp hash_wrapper.cpp MurmurHash3.cpp xxhash.cpp)