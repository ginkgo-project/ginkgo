// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/permutation.hpp>


namespace gko {
namespace matrix {


#define GKO_DECLARE_PERMUTATION_MATRIX(_type) class Permutation<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PERMUTATION_MATRIX);


}  // namespace matrix
}  // namespace gko
