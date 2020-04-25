/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/preconditioner/isai.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/preconditioner/isai_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


enum struct matrix_type { lower, upper };
class Isai : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    Isai() : rand_engine(42) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        omp = gko::OmpExecutor::create();
    }

    std::unique_ptr<Csr> clone_allocations(const Csr *csr_mtx)
    {
        if (csr_mtx->get_executor() != ref) {
            return {nullptr};
        }
        auto size = csr_mtx->get_size();
        const auto num_elems = csr_mtx->get_num_stored_elements();
        auto sparsity = Csr::create(ref, size, num_elems);

        // All arrays are now filled with invalid data to catch potential errors
        auto begin_values = sparsity->get_values();
        auto end_values = begin_values + num_elems;
        std::fill(begin_values, end_values, -gko::one<value_type>());

        auto begin_cols = sparsity->get_col_idxs();
        auto end_cols = begin_cols + num_elems;
        std::fill(begin_cols, end_cols, -gko::one<index_type>());

        auto begin_rows = sparsity->get_row_ptrs();
        auto end_rows = begin_rows + size[0] + 1;
        std::fill(begin_rows, end_rows, -gko::one<index_type>());
        return sparsity;
    }

    void initialize_data(matrix_type type)
    {
        constexpr int n = 97;
        const bool for_lower_tm = type == matrix_type::lower;
        auto nz_dist = std::uniform_int_distribution<index_type>(1, n);
        auto val_dist = std::uniform_real_distribution<value_type>(-1., 1.);
        mtx = Csr::create(ref);
        mtx = gko::test::generate_random_triangular_matrix<Csr>(
            n, n, true, for_lower_tm, nz_dist, val_dist, rand_engine, ref,
            gko::dim<2>{n, n});
        inverse = clone_allocations(mtx.get());

        d_mtx = Csr::create(omp);
        d_mtx->copy_from(mtx.get());
        d_inverse = Csr::create(omp);
        d_inverse->copy_from(inverse.get());
    }


    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    std::default_random_engine rand_engine;

    std::unique_ptr<Csr> mtx;
    std::unique_ptr<Csr> inverse;

    std::unique_ptr<Csr> d_mtx;
    std::unique_ptr<Csr> d_inverse;
};


TEST_F(Isai, OmpIsaiGenerateLinverseIsEquivalentToRef)
{
    initialize_data(matrix_type::lower);

    gko::kernels::reference::isai::generate_l_inverse(ref, mtx.get(),
                                                      inverse.get());
    gko::kernels::omp::isai::generate_l_inverse(omp, d_mtx.get(),
                                                d_inverse.get());

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, r<value_type>::value);
}


TEST_F(Isai, OmpIsaiGenerateUinverseIsEquivalentToRef)
{
    initialize_data(matrix_type::upper);

    gko::kernels::reference::isai::generate_u_inverse(ref, mtx.get(),
                                                      inverse.get());
    gko::kernels::omp::isai::generate_u_inverse(omp, d_mtx.get(),
                                                d_inverse.get());

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, r<value_type>::value);
}


TEST_F(Isai, OmpIsaiIdentityTriangleLowerIsEquivalentToRef)
{
    initialize_data(matrix_type::lower);

    gko::kernels::reference::isai::identity_triangle(ref, mtx.get(), true);
    gko::kernels::omp::isai::identity_triangle(omp, d_mtx.get(), true);

    GKO_ASSERT_MTX_EQ_SPARSITY(mtx, d_mtx);
    GKO_ASSERT_MTX_NEAR(mtx, d_mtx, 0);
}


TEST_F(Isai, OmpIsaiIdentityTriangleUpperIsEquivalentToRef)
{
    initialize_data(matrix_type::upper);

    gko::kernels::reference::isai::identity_triangle(ref, mtx.get(), false);
    gko::kernels::omp::isai::identity_triangle(omp, d_mtx.get(), false);

    GKO_ASSERT_MTX_EQ_SPARSITY(mtx, d_mtx);
    GKO_ASSERT_MTX_NEAR(mtx, d_mtx, 0);
}


}  // namespace
