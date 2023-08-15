# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.16)
project(GinkgoSubdirTest LANGUAGES CXX)
file(CREATE_LINK "${CMAKE_CURRENT_SOURCE_DIR}/../.." "${CMAKE_CURRENT_BINARY_DIR}/ginkgo" SYMBOLIC)

add_subdirectory("${CMAKE_CURRENT_BINARY_DIR}/ginkgo")

add_executable(test_subdir ../test_install/test_install.cpp)
target_compile_features(test_subdir PUBLIC cxx_std_14)
target_link_libraries(test_subdir PRIVATE Ginkgo::ginkgo)
