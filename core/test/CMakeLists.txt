# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

include(${PROJECT_SOURCE_DIR}/cmake/create_test.cmake)

add_subdirectory(accessor)
add_subdirectory(base)
add_subdirectory(components)
if(GINKGO_BUILD_MPI)
    add_subdirectory(mpi)
endif()
add_subdirectory(factorization)
add_subdirectory(log)
add_subdirectory(matrix)
add_subdirectory(multigrid)
add_subdirectory(preconditioner)
add_subdirectory(reorder)
add_subdirectory(solver)
add_subdirectory(stop)
add_subdirectory(utils)
