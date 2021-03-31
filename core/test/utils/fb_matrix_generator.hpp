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


#ifndef GKO_CORE_TEST_UTILS_FIXED_BLOCK_MATRIX_GENERATOR_HPP_
#define GKO_CORE_TEST_UTILS_FIXED_BLOCK_MATRIX_GENERATOR_HPP_


#include <numeric>
#include <random>
#include <type_traits>
#include <vector>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/test/utils/unsort_matrix.hpp"


namespace gko {
namespace test {


template <typename ValueType>
std::enable_if_t<gko::is_complex<ValueType>(), ValueType>
complexify_if_possible(const ValueType x)
{
    using namespace std::complex_literals;
    ValueType y{x};
    constexpr gko::remove_complex<ValueType> eps =
        std::numeric_limits<gko::remove_complex<ValueType>>::epsilon();
    constexpr gko::remove_complex<ValueType> minval =
        std::numeric_limits<gko::remove_complex<ValueType>>::min();
    const gko::remove_complex<ValueType> absval = abs(x);
    if (absval > minval && abs(y.imag / absval) < eps) y.imag = abs(x);
    return y;
}

template <typename ValueType>
std::enable_if_t<!gko::is_complex<ValueType>(), ValueType>
complexify_if_possible(const ValueType x)
{
    return x;
}


/**
 * Generates a block CSR matrix having the same sparsity pattern as
 * a given CSR matrix.
 */
template <typename ValueType, typename IndexType, typename RandEngine>
std::unique_ptr<matrix::Fbcsr<ValueType, IndexType>> generate_fbcsr_from_csr(
    const std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *const csrmat, const int block_size,
    const bool row_diag_dominant, RandEngine &&rand_engine)
{
    const auto nbrows = static_cast<IndexType>(csrmat->get_size()[0]);
    const auto nbcols = static_cast<IndexType>(csrmat->get_size()[1]);
    const auto nbnz_temp =
        static_cast<IndexType>(csrmat->get_num_stored_elements());
    const int bs2 = block_size * block_size;

    auto fmtx = matrix::Fbcsr<ValueType, IndexType>::create(
        exec,
        dim<2>{static_cast<size_type>(nbrows * block_size),
               static_cast<size_type>(nbcols * block_size)},
        nbnz_temp * bs2, block_size);
    exec->copy(nbrows + 1, csrmat->get_const_row_ptrs(), fmtx->get_row_ptrs());
    exec->copy(nbnz_temp, csrmat->get_const_col_idxs(), fmtx->get_col_idxs());

    // We assume diagonal blocks are present for the diagonally-dominant case

    const IndexType nbnz = fmtx->get_num_stored_elements() / bs2;

    const IndexType *const row_ptrs = fmtx->get_const_row_ptrs();
    const IndexType *const col_idxs = fmtx->get_const_col_idxs();
    const IndexType nnz = nbnz * bs2;
    ValueType *const vals = fmtx->get_values();

    std::normal_distribution<gko::remove_complex<ValueType>> norm_dist(0.0,
                                                                       2.0);

    for (IndexType ibrow = 0; ibrow < nbrows; ibrow++) {
        if (row_diag_dominant) {
            const IndexType nrownz =
                (row_ptrs[ibrow + 1] - row_ptrs[ibrow]) * block_size;

            std::uniform_real_distribution<gko::remove_complex<ValueType>>
                diag_dist(1.01 * nrownz, 2 * nrownz);
            std::uniform_real_distribution<gko::remove_complex<ValueType>>
                off_diag_dist(-1.0, 1.0);

            for (IndexType ibz = row_ptrs[ibrow]; ibz < row_ptrs[ibrow + 1];
                 ibz++) {
                for (int i = 0; i < block_size * block_size; i++)
                    vals[ibz * bs2 + i] =
                        complexify_if_possible(off_diag_dist(rand_engine));
                if (col_idxs[ibz] == ibrow) {
                    for (int i = 0; i < block_size; i++)
                        vals[ibz * bs2 + i * block_size + i] =
                            pow(-1, i) *
                            complexify_if_possible(diag_dist(rand_engine));
                }
            }
        } else {
            for (IndexType ibz = row_ptrs[ibrow]; ibz < row_ptrs[ibrow + 1];
                 ibz++) {
                for (int i = 0; i < bs2; i++) {
                    vals[ibz * bs2 + i] =
                        complexify_if_possible(norm_dist(rand_engine));
                }
            }
        }
    }

    return fmtx;
}


template <typename ValueType, typename IndexType, typename RandEngine>
std::unique_ptr<matrix::Fbcsr<ValueType, IndexType>> generate_random_fbcsr(
    std::shared_ptr<const ReferenceExecutor> ref, RandEngine engine,
    const IndexType nbrows, const IndexType nbcols, const int mat_blk_sz,
    const bool diag_dominant, const bool unsort)
{
    using real_type = gko::remove_complex<ValueType>;
    std::unique_ptr<matrix::Csr<ValueType, IndexType>> rand_csr_ref =
        generate_random_matrix<matrix::Csr<ValueType, IndexType>>(
            nbrows, nbcols,
            std::uniform_int_distribution<IndexType>(0, nbcols - 1),
            std::normal_distribution<real_type>(0.0, 1.0), std::move(engine),
            ref);
    gko::kernels::reference::factorization::add_diagonal_elements(
        ref, gko::lend(rand_csr_ref), false);
    if (unsort && rand_csr_ref->is_sorted_by_column_index()) {
        unsort_matrix(rand_csr_ref.get(), engine);
    }
    return generate_fbcsr_from_csr(ref, rand_csr_ref.get(), mat_blk_sz,
                                   diag_dominant, std::move(engine));
}


}  // namespace test
}  // namespace gko

#endif
