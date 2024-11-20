// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_FACTORIZATION_INCOMPLETE_FACTORIZATION_HPP_
#define GKO_PUBLIC_CORE_FACTORIZATION_INCOMPLETE_FACTORIZATION_HPP_


namespace gko {
namespace factorization {


/**
 * An enum class for algorithm selection in the incomplete factorization.
 * `sparselib` is only available for CUDA, HIP, and reference.
 * `syncfree` is Ginkgo's implementation by using the Lu/Cholesky factorization
 * components with given sparsity.
 */
enum class incomplete_factorize_algorithm { sparselib, syncfree };


}  // namespace factorization
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_FACTORIZATION_INCOMPLETE_FACTORIZATION_HPP_
