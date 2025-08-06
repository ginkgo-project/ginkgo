// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/executor.hpp"

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


std::string OmpExecutor::get_description() const
{
    return "OmpExecutor (" + std::to_string(this->get_num_omp_threads()) +
           " threads)";
}


}  // namespace gko
