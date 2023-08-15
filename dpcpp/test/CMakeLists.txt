# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

include(${PROJECT_SOURCE_DIR}/cmake/create_test.cmake)
set(GINKGO_COMPILING_DPCPP_TEST ON)

add_subdirectory(base)
add_subdirectory(components)
add_subdirectory(matrix)
add_subdirectory(preconditioner)
