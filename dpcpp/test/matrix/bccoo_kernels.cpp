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

#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/matrix/coo_kernels.hpp"
#include "core/test/utils.hpp"
#include "dpcpp/test/utils.hpp"


namespace {


class Bccoo : public ::testing::Test {
protected:
#if GINKGO_DPCPP_SINGLE_MODE
    using vtype = float;
#else
    using vtype = double;
#endif  // GINKGO_DPCPP_SINGLE_MODE
    using Mtx = gko::matrix::Bccoo<vtype>;
    using Vec = gko::matrix::Dense<vtype>;
    using ComplexVec = gko::matrix::Dense<std::complex<vtype>>;

    Bccoo() : rand_engine(42) {}

    void SetUp()
    {
        ASSERT_GT(gko::DpcppExecutor::get_num_devices("all"), 0);
        ref = gko::ReferenceExecutor::create();
        dpcpp = gko::DpcppExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (dpcpp != nullptr) {
            ASSERT_NO_THROW(dpcpp->synchronize());
        }
    }

    template <typename MtxType = Vec>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols, std::uniform_int_distribution<>(1, num_cols),
            std::normal_distribution<vtype>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data_grp(int num_vectors = 1)
    {
        mtx_grp = Mtx::create(ref, 0, gko::matrix::bccoo::compression::block);
        mtx_grp->move_from(gen_mtx(532, 231));
        expected = gen_mtx(532, num_vectors);
        y = gen_mtx(231, num_vectors);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx_grp =
            Mtx::create(dpcpp, 128, gko::matrix::bccoo::compression::block);
        dmtx_grp->copy_from(mtx_grp.get());
        dresult = Vec::create(dpcpp);
        dresult->copy_from(expected.get());
        dy = Vec::create(dpcpp);
        dy->copy_from(y.get());
        dalpha = Vec::create(dpcpp);
        dalpha->copy_from(alpha.get());
        dbeta = Vec::create(dpcpp);
        dbeta->copy_from(beta.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::DpcppExecutor> dpcpp;

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> mtx_grp;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx_grp;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
};


TEST_F(Bccoo, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data_grp();

    mtx_grp->apply(y.get(), expected.get());
    dmtx_grp->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data_grp();

    mtx_grp->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx_grp->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, SimpleApplyAddIsEquivalentToRef)
{
    set_up_apply_data_grp();

    mtx_grp->apply2(y.get(), expected.get());
    dmtx_grp->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, AdvancedApplyAddIsEquivalentToRef)
{
    set_up_apply_data_grp();

    mtx_grp->apply2(alpha.get(), y.get(), expected.get());
    dmtx_grp->apply2(dalpha.get(), dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, SimpleApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data_grp(3);

    mtx_grp->apply(y.get(), expected.get());
    dmtx_grp->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, AdvancedApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data_grp(3);

    mtx_grp->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx_grp->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, SimpleApplyAddToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data_grp(3);

    mtx_grp->apply2(y.get(), expected.get());
    dmtx_grp->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, SimpleApplyAddToLargeDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data_grp(33);

    mtx_grp->apply2(y.get(), expected.get());
    dmtx_grp->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, AdvancedApplyAddToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data_grp(3);

    mtx_grp->apply2(alpha.get(), y.get(), expected.get());
    dmtx_grp->apply2(dalpha.get(), dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, AdvancedApplyAddToLargeDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data_grp(33);

    mtx_grp->apply2(y.get(), expected.get());
    dmtx_grp->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data_grp();
    auto complex_b = gen_mtx<ComplexVec>(231, 3);
    auto dcomplex_b = gko::clone(dpcpp, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(532, 3);
    auto dcomplex_x = gko::clone(dpcpp, complex_x);

    mtx_grp->apply(complex_b.get(), complex_x.get());
    dmtx_grp->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<vtype>::value);
}


TEST_F(Bccoo, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data_grp();
    auto complex_b = gen_mtx<ComplexVec>(231, 3);
    auto dcomplex_b = gko::clone(dpcpp, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(532, 3);
    auto dcomplex_x = gko::clone(dpcpp, complex_x);

    mtx_grp->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    dmtx_grp->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(),
                    dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<vtype>::value);
}


TEST_F(Bccoo, ApplyAddToComplexIsEquivalentToRef)
{
    set_up_apply_data_grp();
    auto complex_b = gen_mtx<ComplexVec>(231, 3);
    auto dcomplex_b = gko::clone(dpcpp, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(532, 3);
    auto dcomplex_x = gko::clone(dpcpp, complex_x);

    mtx_grp->apply2(alpha.get(), complex_b.get(), complex_x.get());
    dmtx_grp->apply2(dalpha.get(), dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<vtype>::value);
}


TEST_F(Bccoo, ConvertToDenseIsEquivalentToRef)
{
    set_up_apply_data_grp();
    auto dense_mtx_grp = gko::matrix::Dense<vtype>::create(ref);
    auto ddense_mtx_grp = gko::matrix::Dense<vtype>::create(dpcpp);

    mtx_grp->convert_to(dense_mtx_grp.get());
    dmtx_grp->convert_to(ddense_mtx_grp.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx_grp.get(), ddense_mtx_grp.get(), 0);
}


TEST_F(Bccoo, ConvertToCooIsEquivalentToRef)
{
    set_up_apply_data_grp();
    auto dense_mtx_grp = gko::matrix::Dense<vtype>::create(ref);
    auto coo_mtx_grp = gko::matrix::Coo<vtype>::create(ref);
    auto dcoo_mtx_grp = gko::matrix::Coo<vtype>::create(dpcpp);

    mtx_grp->convert_to(dense_mtx_grp.get());
    dense_mtx_grp->convert_to(coo_mtx_grp.get());
    dmtx_grp->convert_to(dcoo_mtx_grp.get());

    GKO_ASSERT_MTX_NEAR(coo_mtx_grp.get(), dcoo_mtx_grp.get(), 0);
}


TEST_F(Bccoo, ConvertToCsrIsEquivalentToRef)
{
    set_up_apply_data_grp();
    auto dense_mtx_grp = gko::matrix::Dense<vtype>::create(ref);
    auto csr_mtx_grp = gko::matrix::Csr<vtype>::create(ref);
    auto dcsr_mtx_grp = gko::matrix::Csr<vtype>::create(dpcpp);

    mtx_grp->convert_to(dense_mtx_grp.get());
    dense_mtx_grp->convert_to(csr_mtx_grp.get());
    dmtx_grp->convert_to(dcsr_mtx_grp.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx_grp.get(), dcsr_mtx_grp.get(), 0);
}


TEST_F(Bccoo, ExtractDiagonalIsEquivalentToRef)
{
    set_up_apply_data_grp();

    auto diag = mtx_grp->extract_diagonal();
    auto ddiag = dmtx_grp->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Bccoo, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data_grp();

    mtx_grp->compute_absolute_inplace();
    dmtx_grp->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(mtx_grp, dmtx_grp, r<vtype>::value);
}


TEST_F(Bccoo, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data_grp();

    auto abs_mtx_grp = mtx_grp->compute_absolute();
    auto dabs_mtx_grp = dmtx_grp->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx_grp, dabs_mtx_grp, r<vtype>::value);
}


}  // namespace
