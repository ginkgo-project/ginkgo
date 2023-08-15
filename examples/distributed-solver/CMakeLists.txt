# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

add_executable(distributed-solver distributed-solver.cpp)
target_link_libraries(distributed-solver Ginkgo::ginkgo)
