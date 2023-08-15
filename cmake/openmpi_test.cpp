// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdio>


#include <mpi.h>


int main()
{
#if defined(OPEN_MPI) && OPEN_MPI
    std::printf("%d.%d.%d", OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
                OMPI_RELEASE_VERSION);
    return 1;
#else
    return 0;
#endif
}
