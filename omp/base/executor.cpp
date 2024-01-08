// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/executor.hpp>


#include <omp.h>


namespace gko {


int OmpExecutor::get_num_omp_threads()
{
    int num_threads;
#pragma omp parallel
#pragma omp single
    num_threads = omp_get_num_threads();
    return num_threads;
}


}  // namespace gko
