// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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

    omp_mutex(const omp_mutex&) = delete;
    omp_mutex(omp_mutex&&) = delete;
    omp_mutex& operator=(const omp_mutex&) = delete;
    omp_mutex& operator=(omp_mutex&&) = delete;

    void lock() { omp_set_lock(&lock_); }

    void unlock() { omp_unset_lock(&lock_); }

private:
    omp_lock_t lock_;
};


}  // namespace omp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_OMP_COMPONENTS_OMP_MUTEX_HPP_
