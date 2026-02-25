cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(soup VERSION 0.0.1 LANGUAGES C)

set(CMAKE_C_STANDARD 11) 
set(CMAKE_C_STANDARD_REQUIRED ON)


find_package(OpenMP REQUIRED)
add_executable(program mxs.c)
add_executable(integra integr.c)
target_link_libraries(program PRIVATE OpenMP::OpenMP_C)
target_link_libraries(integra PRIVATE OpenMP::OpenMP_C m)
