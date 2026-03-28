cmake_minimum_required(VERSION 3.22.1)
project("animex_neon")

add_library(
    animex_neon
    SHARED
    animex_neon.cpp
)

# Enable NEON and aggressive optimization
target_compile_options(animex_neon PRIVATE
    -O3
    -ftree-vectorize
    -ffast-math
    -DANDROID
)

target_link_libraries(animex_neon
    android
    log
)