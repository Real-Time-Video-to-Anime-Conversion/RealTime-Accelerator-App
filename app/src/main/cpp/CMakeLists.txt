cmake_minimum_required(VERSION 3.22.1)

project("animex")

add_library(
    ${CMAKE_PROJECT_NAME}
    SHARED
    image-processor.cpp
)

target_link_libraries(
    ${CMAKE_PROJECT_NAME}
    android
    log
    jnigraphics
)