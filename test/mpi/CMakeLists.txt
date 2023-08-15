# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_create_common_and_reference_test(matrix MPI_SIZE 3)
ginkgo_create_common_and_reference_test(vector MPI_SIZE 3)

add_subdirectory(preconditioner)
add_subdirectory(solver)
