// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_UTILS_UNSORT_MATRIX_HPP_
#define GKO_CORE_TEST_UTILS_UNSORT_MATRIX_HPP_


#include <algorithm>
#include <random>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/iterator_factory.hpp"


namespace gko {
namespace test {


// Plan for now: shuffle values and column indices to unsort the given matrix
// without changing the represented matrix.
template <typename MtxPtr, typename RandomEngine>
void unsort_matrix(MtxPtr&& mtx, RandomEngine&& engine)
{
    using value_type = typename gko::detail::pointee<MtxPtr>::value_type;
    using index_type = typename gko::detail::pointee<MtxPtr>::index_type;
    matrix_data<value_type, index_type> data;
    mtx->write(data);
    auto& nonzeros = data.nonzeros;
    using nz_type = typename decltype(data)::nonzero_type;

    std::shuffle(nonzeros.begin(), nonzeros.end(), engine);
    std::stable_sort(nonzeros.begin(), nonzeros.end(),
                     [](nz_type a, nz_type b) { return a.row < b.row; });
    mtx->read(data);
}


}  // namespace test
}  // namespace gko

#endif  // GKO_CORE_TEST_UTILS_UNSORT_MATRIX_HPP_
