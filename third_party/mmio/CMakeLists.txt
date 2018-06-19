cmake_minimum_required(VERSION 3.8)
project(MMIO LANGUAGES C VERSION 1.0)

add_library(mmio STATIC mmio.c)
target_include_directories(mmio PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
