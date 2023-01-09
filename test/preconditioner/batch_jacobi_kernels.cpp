/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include "core/preconditioner/batch_jacobi_kernels.hpp"


#include <limits>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"
#include "test/utils/executor.hpp"


namespace {


template <typename T>
std::complex<T> get_num(std::complex<T>)
{
    return {5.0, 1.5};
}

template <typename T>
T get_num(T)
{
    return 5.0;
}

template <typename T>
class BatchJacobi : public CommonTestFixture {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;

    BatchJacobi()
        : ref_mtx(gko::test::generate_uniform_batch_random_matrix<Mtx>(
              nbatch, nrows, nrows,
              std::uniform_int_distribution<>(1, nrows - 1),
              std::normal_distribution<real_type>(), std::ranlux48(34), true,
              ref)),
          d_mtx(Mtx::create(exec))
    {
        // make diagonal larger
        const int* const row_ptrs = ref_mtx->get_const_row_ptrs();
        const int* const col_idxs = ref_mtx->get_const_col_idxs();
        value_type* const vals = ref_mtx->get_values();
        const int nnz = row_ptrs[nrows];
        for (int irow = 0; irow < nrows; irow++) {
            for (int iz = row_ptrs[irow]; iz < row_ptrs[irow + 1]; iz++) {
                if (col_idxs[iz] == irow) {
                    for (size_t ibatch = 0; ibatch < nbatch; ibatch++) {
                        // TODO: take care of any padding here
                        const size_t valpos = iz + ibatch * nnz;
                        vals[valpos] =
                            get_num(T{}) + static_cast<T>(std::sin(irow));
                    }
                }
            }
        }
        d_mtx->copy_from(ref_mtx.get());
    }


    const size_t nbatch = 10;
    const int nrows = 50;
    std::unique_ptr<Mtx> ref_mtx;
    std::unique_ptr<Mtx> d_mtx;
    static constexpr real_type eps = std::numeric_limits<real_type>::epsilon();

    void check_jacobi(const int nrhs)
    {
        auto ref_b = gko::test::generate_uniform_batch_random_matrix<BDense>(
            nbatch, nrows, nrhs, std::uniform_int_distribution<>(nrhs, nrhs),
            std::normal_distribution<real_type>(), std::ranlux48(34), false,
            ref);
        auto d_b = BDense::create(exec);
        d_b->copy_from(ref_b.get());
        auto ref_x = BDense::create(
            ref, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrhs)));
        auto d_x = BDense::create(
            exec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrhs)));

        gko::kernels::EXEC_NAMESPACE::batch_jacobi::batch_jacobi_apply(
            exec, d_mtx.get(), d_b.get(), d_x.get());
        gko::kernels::reference::batch_jacobi::batch_jacobi_apply(
            ref, ref_mtx.get(), ref_b.get(), ref_x.get());

        exec->synchronize();
        GKO_ASSERT_BATCH_MTX_NEAR(ref_x, d_x, 5 * eps);
    }
};

TYPED_TEST_SUITE(BatchJacobi, gko::test::ValueTypes);


TYPED_TEST(BatchJacobi, ApplySingleIsEquivalentToReference)
{
    this->check_jacobi(1);
}


}  // namespace
