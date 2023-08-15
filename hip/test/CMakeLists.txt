# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

include(${PROJECT_SOURCE_DIR}/cmake/create_test.cmake)

add_subdirectory(base)
add_subdirectory(components)
add_subdirectory(matrix)
add_subdirectory(solver)
add_subdirectory(utils)
