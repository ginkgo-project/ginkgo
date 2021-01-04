/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_OMP_COMPONENTS_OMP_MUTEX_HPP_
#define GKO_OMP_COMPONENTS_OMP_MUTEX_HPP_


#include <omp.h>


namespace gko {
namespace kernels {
namespace omp {

/**
 * Wrapper class for the OpenMP mutex omp_lock_t.
 *
 * Fulfills the BasicLockable requirement, which means it can be used with
 * std::lock_guard<omp_mutex>, making RAII possible.
 */
struct omp_mutex {
    omp_mutex() { omp_init_lock(&lock_); }
    ~omp_mutex() { omp_destroy_lock(&lock_); }

    omp_mutex(const omp_mutex &) = delete;
    omp_mutex(omp_mutex &&) = delete;
    omp_mutex &operator=(const omp_mutex &) = delete;
    omp_mutex &operator=(omp_mutex &&) = delete;

    void lock() { omp_set_lock(&lock_); }

    void unlock() { omp_unset_lock(&lock_); }

private:
    omp_lock_t lock_;
};


}  // namespace omp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_OMP_COMPONENTS_OMP_MUTEX_HPP_
