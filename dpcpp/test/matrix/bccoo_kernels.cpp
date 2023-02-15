/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/matrix/coo.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/matrix/coo_kernels.hpp"
#include "core/test/utils.hpp"
// #include "core/test/utils/unsort_matrix.hpp"
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

    void set_up_apply_data(int num_vectors = 1)
    {
        mtx = gen_mtx<Mtx>(532, 231);
        expected = gen_mtx(532, num_vectors);
        y = gen_mtx(231, num_vectors);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = gko::clone(dpcpp, mtx);
        dresult = gko::clone(dpcpp, expected);
        dy = gko::clone(dpcpp, y);
        dalpha = gko::clone(dpcpp, alpha);
        dbeta = gko::clone(dpcpp, beta);
    }
    void set_up_apply_data_blk(int num_vectors = 1)
    {
        mtx_blk = Mtx::create(ref, 0, gko::matrix::bccoo::compression::block);
//        mtx_blk->copy_from(gen_mtx(532, 231));
/**/
//				std::unique_ptr<Vec> mat = gen_mtx(532, 231);
				auto mat = gen_mtx(532, 231);
				for(int i=0; i<532; i++) {
						int k = i % 231;
						for(int j=0; j<231; j++) {
								mat->at(i,k) = (i*231 + j + 1.0) / 1000;
								k++; if (k == 231) k = 0;
						}
				}
//        mtx_blk->copy_from(gko::share(&mat));
//        mtx_blk->copy_from(mat);
        mtx_blk->copy_from(gko::clone(ref, mat));
/**/
        expected = gen_mtx(532, num_vectors);
        y = gen_mtx(231, num_vectors);
				for(int i=0; i<231; i++) {
						int k = i % num_vectors;
						for(int j=0; j<num_vectors; j++) {
								y->at(i,k) = -(i*num_vectors + j + 1.0) / 1000;
								k++; if (k == num_vectors) k = 0;
						}
				}
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        //        dmtx_blk = Mtx::create(dpcpp, 32,
        //        gko::matrix::bccoo::compression::block);
//        dmtx_blk =
//            Mtx::create(dpcpp, 128, gko::matrix::bccoo::compression::block);
//        dmtx_blk->copy_from(mtx_blk);
				dmtx_blk = gko::clone(dpcpp, mtx_blk);
        dresult = gko::clone(dpcpp, expected);
        dy = gko::clone(dpcpp, y);
        dalpha = gko::clone(dpcpp, alpha);
        dbeta = gko::clone(dpcpp, beta);
    }
/*
    void unsort_mtx()
    {
        gko::test::unsort_matrix(mtx.get(), rand_engine);
        dmtx->copy_from(mtx.get());
    }
*/
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::DpcppExecutor> dpcpp;

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> mtx_blk;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Mtx> dmtx_blk;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
};


TEST_F(Bccoo, SimpleApplyIsEquivalentToRef)
{
//    std::cout << "SimpleApplyIsEquivalentToRef - A " << std::endl;
    set_up_apply_data_blk();

//    std::cout << "SimpleApplyIsEquivalentToRef - B " << std::endl;
    mtx_blk->apply(y.get(), expected.get());
//    std::cout << "SimpleApplyIsEquivalentToRef - C " << std::endl;
    dmtx_blk->apply(dy.get(), dresult.get());

//    std::cout << "SimpleApplyIsEquivalentToRef - D " << std::endl;
    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
//    std::cout << "SimpleApplyIsEquivalentToRef - E " << std::endl;
}

/*
TEST_F(Bccoo, SimpleApplyDoesntOverwritePadding)
{
    set_up_apply_data_blk();
    auto dresult_padded =
        Vec::create(dpcpp, dresult->get_size(), dresult->get_stride() + 1);
    dresult_padded->copy_from(dresult.get());
    vtype padding_val{1234.0};
    dpcpp->copy_from(dpcpp->get_master().get(), 1, &padding_val,
                     dresult_padded->get_values() + 1);

    mtx_blk->apply(y.get(), expected.get());
    dmtx_blk->apply(dy.get(), dresult_padded.get());

    GKO_ASSERT_MTX_NEAR(dresult_padded, expected, r<vtype>::value);
    ASSERT_EQ(dpcpp->copy_val_to_host(dresult_padded->get_values() + 1),
              1234.0);
}
*/
/*
TEST_F(Bccoo, SimpleApplyIsEquivalentToRefUnsorted)
{
    set_up_apply_data_blk();
    unsort_mtx();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}
*/

TEST_F(Bccoo, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data_blk();

    mtx_blk->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx_blk->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

//		std::cout << dresult->get_size() << std::endl;
//		std::cout << dresult->get_size()[0] << std::endl;
//		std::cout << expected->get_values()[1] << "  $$   "
//		          <<  dresult->get_values()[1] << std::endl;
//		std::cout << expected->get_values()[0] << "  $$   "
//		          << expected->get_values()[expected->get_size()[0]-1] << std::endl;
//		std::cout << dresult->get_values()[0] << "  $$   "
//		          << dresult->get_values()[dresult->get_size()[0]-1] << std::endl;
    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}

/*
TEST_F(Bccoo, AdvancedApplyDoesntOverwritePadding)
{
    set_up_apply_data_blk();
    auto dresult_padded =
        Vec::create(dpcpp, dresult->get_size(), dresult->get_stride() + 1);
    dresult_padded->copy_from(dresult.get());
    vtype padding_val{1234.0};
    dpcpp->copy_from(dpcpp->get_master().get(), 1, &padding_val,
                     dresult_padded->get_values() + 1);

    mtx_blk->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx_blk->apply(dalpha.get(), dy.get(), dbeta.get(), dresult_padded.get());

    GKO_ASSERT_MTX_NEAR(dresult_padded, expected, r<vtype>::value);
    ASSERT_EQ(dpcpp->copy_val_to_host(dresult_padded->get_values() + 1),
              1234.0);
}
*/

TEST_F(Bccoo, SimpleApplyAddIsEquivalentToRef)
{
    set_up_apply_data_blk();

    mtx_blk->apply2(y.get(), expected.get());
    dmtx_blk->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, AdvancedApplyAddIsEquivalentToRef)
{
    set_up_apply_data_blk();

    mtx_blk->apply2(alpha.get(), y.get(), expected.get());
    dmtx_blk->apply2(dalpha.get(), dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, SimpleApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data_blk(3);

    mtx_blk->apply(y.get(), expected.get());
    dmtx_blk->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, AdvancedApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data_blk(3);

    mtx_blk->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx_blk->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, SimpleApplyAddToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data_blk(3);

    mtx_blk->apply2(y.get(), expected.get());
    dmtx_blk->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, SimpleApplyAddToLargeDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data_blk(33);

    mtx_blk->apply2(y.get(), expected.get());
    dmtx_blk->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, AdvancedApplyAddToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data_blk(3);

    mtx_blk->apply2(alpha.get(), y.get(), expected.get());
    dmtx_blk->apply2(dalpha.get(), dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, AdvancedApplyAddToLargeDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data_blk(33);

    mtx_blk->apply2(y.get(), expected.get());
    dmtx_blk->apply2(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, r<vtype>::value);
}


TEST_F(Bccoo, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data_blk();
    auto complex_b = gen_mtx<ComplexVec>(231, 3);
    auto dcomplex_b = gko::clone(dpcpp, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(532, 3);
    auto dcomplex_x = gko::clone(dpcpp, complex_x);

    mtx_blk->apply(complex_b.get(), complex_x.get());
    dmtx_blk->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<vtype>::value);
}


TEST_F(Bccoo, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data_blk();
    auto complex_b = gen_mtx<ComplexVec>(231, 3);
    auto dcomplex_b = gko::clone(dpcpp, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(532, 3);
    auto dcomplex_x = gko::clone(dpcpp, complex_x);

    mtx_blk->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    dmtx_blk->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<vtype>::value);
}


TEST_F(Bccoo, ApplyAddToComplexIsEquivalentToRef)
{
    set_up_apply_data_blk();
    auto complex_b = gen_mtx<ComplexVec>(231, 3);
    auto dcomplex_b = gko::clone(dpcpp, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(532, 3);
    auto dcomplex_x = gko::clone(dpcpp, complex_x);

    mtx_blk->apply2(alpha.get(), complex_b.get(), complex_x.get());
    dmtx_blk->apply2(dalpha.get(), dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<vtype>::value);
}


TEST_F(Bccoo, ConvertToDenseIsEquivalentToRef)
{
    set_up_apply_data_blk();
    auto dense_mtx_blk = gko::matrix::Dense<vtype>::create(ref);
    auto ddense_mtx_blk = gko::matrix::Dense<vtype>::create(dpcpp);

    mtx_blk->convert_to(dense_mtx_blk.get());
    dmtx_blk->convert_to(ddense_mtx_blk.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx_blk.get(), ddense_mtx_blk.get(), 0);
}


TEST_F(Bccoo, ConvertToCsrIsEquivalentToRef)
{
    set_up_apply_data_blk();
    auto dense_mtx_blk = gko::matrix::Dense<vtype>::create(ref);
    auto csr_mtx_blk = gko::matrix::Csr<vtype>::create(ref);
    auto dcsr_mtx_blk = gko::matrix::Csr<vtype>::create(dpcpp);

    mtx_blk->convert_to(dense_mtx_blk.get());
    dense_mtx_blk->convert_to(csr_mtx_blk.get());
    dmtx_blk->convert_to(dcsr_mtx_blk.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx_blk.get(), dcsr_mtx_blk.get(), 0);
}


TEST_F(Bccoo, ExtractDiagonalIsEquivalentToRef)
{
    set_up_apply_data_blk();

    auto diag = mtx_blk->extract_diagonal();
    auto ddiag = dmtx_blk->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Bccoo, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data_blk();

    mtx_blk->compute_absolute_inplace();
    dmtx_blk->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(mtx_blk, dmtx_blk, r<vtype>::value);
}


TEST_F(Bccoo, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data_blk();

    auto abs_mtx_blk = mtx_blk->compute_absolute();
    auto dabs_mtx_blk = dmtx_blk->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx_blk, dabs_mtx_blk, r<vtype>::value);
}


}  // namespace
