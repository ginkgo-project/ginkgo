// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_PERMUTATION_HPP_
#define GKO_CORE_MATRIX_PERMUTATION_HPP_


#include <ginkgo/core/matrix/permutation.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace matrix {


/**
 * Checks that the given input and permutation size are consistent with
 * the given mode.
 */
void validate_permute_dimensions(dim<2> size, dim<2> permutation_size,
                                 permute_mode mode);


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_PERMUTATION_HPP_
