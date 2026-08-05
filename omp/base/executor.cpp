// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/executor.hpp"

#include <omp.h>


namespace gko {


void OmpExecutor::populate_exec_info()
{
    this->get_exec_info().num_computing_units = omp_get_max_threads();
    this->get_exec_info().num_pu_per_cu = 1;
}


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
