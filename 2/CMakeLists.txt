cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(soup VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
add_executable(th third_e.cpp)

target_link_libraries(th PRIVATE OpenMP::OpenMP_CXX m)