cmake_minimum_required(VERSION 3.20)
project(gpu_emu)

set(CMAKE_CXX_STANDARD 20)

# Main library
add_library(gpu_emu_lib
    src/ptx_lexer.cpp
    src/ptx_parser.cpp
)
target_include_directories(gpu_emu_lib PUBLIC include)

# Test runner
enable_testing()
add_subdirectory(tests)