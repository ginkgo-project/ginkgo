# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.16)
project(poisson-solver)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.7.0 REQUIRED)
endif()

add_executable(poisson-solver poisson-solver.cpp)
target_link_libraries(poisson-solver Ginkgo::ginkgo)
