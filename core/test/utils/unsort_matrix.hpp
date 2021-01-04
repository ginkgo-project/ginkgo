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

#ifndef GKO_CORE_TEST_UTILS_UNSORT_MATRIX_HPP_
#define GKO_CORE_TEST_UTILS_UNSORT_MATRIX_HPP_


#include <algorithm>
#include <random>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/iterator_factory.hpp"


namespace gko {
namespace test {


// Plan for now: shuffle values and column indices to unsort the given matrix
// without changing the represented matrix.
template <typename ValueType, typename IndexType, typename RandomEngine>
void unsort_matrix(matrix::Csr<ValueType, IndexType> *mtx,
                   RandomEngine &&engine)
{
    using value_type = ValueType;
    using index_type = IndexType;
    auto size = mtx->get_size();
    if (mtx->get_num_stored_elements() <= 0) {
        return;
    }
    const auto &exec = mtx->get_executor();
    const auto &master = exec->get_master();

    // If exec is not the master/host, extract the master and perform the
    // unsorting there, followed by copying it back
    if (exec != master) {
        auto h_mtx = mtx->clone(master);
        unsort_matrix(lend(h_mtx), engine);
        mtx->copy_from(lend(h_mtx));
        return;
    }

    auto vals = mtx->get_values();
    auto row_ptrs = mtx->get_row_ptrs();
    auto cols = mtx->get_col_idxs();

    for (index_type row = 0; row < size[0]; ++row) {
        auto start = row_ptrs[row];
        auto end = row_ptrs[row + 1];
        auto sort_wrapper = gko::detail::IteratorFactory<IndexType, ValueType>(
            cols + start, vals + start, end - start);
        std::shuffle(sort_wrapper.begin(), sort_wrapper.end(), engine);
    }
}


// Plan for now: shuffle values and column indices to unsort the given matrix
// without changing the represented matrix.
template <typename ValueType, typename IndexType, typename RandomEngine>
void unsort_matrix(matrix::Coo<ValueType, IndexType> *mtx,
                   RandomEngine &&engine)
{
    using value_type = ValueType;
    using index_type = IndexType;
    auto nnz = mtx->get_num_stored_elements();
    if (nnz <= 0) {
        return;
    }

    const auto &exec = mtx->get_executor();
    const auto &master = exec->get_master();

    // If exec is not the master/host, extract the master and perform the
    // unsorting there, followed by copying it back
    if (exec != master) {
        auto h_mtx = mtx->clone(master);
        unsort_matrix(lend(h_mtx), engine);
        mtx->copy_from(lend(h_mtx));
        return;
    }
    matrix_data<value_type, index_type> data;
    mtx->write(data);
    auto &nonzeros = data.nonzeros;
    using nz_type = typename decltype(data)::nonzero_type;

    std::shuffle(nonzeros.begin(), nonzeros.end(), engine);
    std::stable_sort(nonzeros.begin(), nonzeros.end(),
                     [](nz_type a, nz_type b) { return a.row < b.row; });
    mtx->read(data);
}


}  // namespace test
}  // namespace gko

#endif  // GKO_CORE_TEST_UTILS_UNSORT_MATRIX_HPP_
