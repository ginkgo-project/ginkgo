cmake_minimum_required(VERSION 2.8)
project(MMIO)

add_library(mmio STATIC mmio.c)
target_include_directories(mmio PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
