# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

add_executable(matrix matrix.cpp)
add_executable(matrix_complex matrix.cpp)
target_compile_definitions(matrix_complex PRIVATE GKO_TOOL_COMPLEX)
target_link_libraries(matrix Ginkgo::ginkgo)
target_link_libraries(matrix_complex Ginkgo::ginkgo)
add_executable(mtx_to_binary mtx_to_binary.cpp)
target_link_libraries(mtx_to_binary Ginkgo::ginkgo)
