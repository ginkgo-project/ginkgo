# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

ginkgo_create_test(communicator MPI_SIZE 8)
ginkgo_create_test(exception_helpers MPI_SIZE 2)
ginkgo_create_test(bindings MPI_SIZE 4)
ginkgo_create_test(polymorphic_object MPI_SIZE 4)
if(NOT (MSVC OR WIN32))
    # This test uses some UNIX function so it's disabled on Windows
    ginkgo_create_test(rank_mapping MPI_SIZE 4)
endif()
