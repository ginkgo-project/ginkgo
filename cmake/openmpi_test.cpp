// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdio>


#include <mpi.h>


int main()
{
#if CHECK_HAS_OPEN_MPI && defined(OPEN_MPI) && OPEN_MPI
    static_assert(true, "Check availability of OpenMPI");
#elif CHECK_OPEN_MPI_VERSION && defined(OPEN_MPI) && OPEN_MPI
    static_assert(OMPI_MAJOR_VERSION > 4 ||
                      (OMPI_MAJOR_VERSION == 4 && OMPI_MINOR_VERSION >= 1),
                  "Check OpenMPI version.");
#else
    static_assert(false, "No OpenMPI available");
#endif
}
