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

#include "core/factorization/par_ilut_kernels.hpp"


#include <algorithm>
#include <fstream>
#include <memory>
#include <random>
#include <string>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/test/utils.hpp"
#include "matrices/config.hpp"


namespace {


template <typename ValueIndexType>
class ParIlut : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Dense = gko::matrix::Dense<value_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    ParIlut()
        : mtx_size(532, 423),
          rand_engine(1337),
          ref(gko::ReferenceExecutor::create()),
          omp(gko::OmpExecutor::create())
    {
        mtx1 = gko::test::generate_random_matrix<Csr>(
            mtx_size[0], mtx_size[1],
            std::uniform_int_distribution<index_type>(10, mtx_size[1]),
            std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                      1.0),
            rand_engine, ref);
        mtx2 = gko::test::generate_random_matrix<Csr>(
            mtx_size[0], mtx_size[1],
            std::uniform_int_distribution<index_type>(0, mtx_size[1]),
            std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                      1.0),
            rand_engine, ref);
        mtx_square = gko::test::generate_random_matrix<Csr>(
            mtx_size[0], mtx_size[0],
            std::uniform_int_distribution<index_type>(1, mtx_size[0]),
            std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                      1.0),
            rand_engine, ref);
        mtx_l = gko::test::generate_random_lower_triangular_matrix<Csr>(
            mtx_size[0], mtx_size[0], false,
            std::uniform_int_distribution<index_type>(10, mtx_size[0]),
            std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                      1.0),
            rand_engine, ref);
        mtx_l2 = gko::test::generate_random_lower_triangular_matrix<Csr>(
            mtx_size[0], mtx_size[0], true,
            std::uniform_int_distribution<index_type>(1, mtx_size[0]),
            std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                      1.0),
            rand_engine, ref);
        mtx_u = gko::test::generate_random_upper_triangular_matrix<Csr>(
            mtx_size[0], mtx_size[0], false,
            std::uniform_int_distribution<index_type>(10, mtx_size[0]),
            std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                      1.0),
            rand_engine, ref);

        dmtx1 = Csr::create(omp);
        dmtx1->copy_from(mtx1.get());
        dmtx2 = Csr::create(omp);
        dmtx2->copy_from(mtx2.get());
        dmtx_square = Csr::create(omp);
        dmtx_square->copy_from(mtx_square.get());
        dmtx_ani = Csr::create(omp);
        dmtx_l_ani = Csr::create(omp);
        dmtx_u_ani = Csr::create(omp);
        dmtx_ut_ani = Csr::create(omp);
        dmtx_l = Csr::create(omp);
        dmtx_l->copy_from(mtx_l.get());
        dmtx_l2 = Csr::create(omp);
        dmtx_l2->copy_from(mtx_l2.get());
        dmtx_u = Csr::create(omp);
        dmtx_u->copy_from(mtx_u.get());
    }

    void SetUp()
    {
        std::string file_name(gko::matrices::location_ani4_mtx);
        auto input_file = std::ifstream(file_name, std::ios::in);
        if (!input_file) {
            FAIL() << "Could not find the file \"" << file_name
                   << "\", which is required for this test.\n";
        }
        mtx_ani = gko::read<Csr>(input_file, ref);
        mtx_ani->sort_by_column_index();

        {
            mtx_l_ani = Csr::create(ref, mtx_ani->get_size());
            mtx_u_ani = Csr::create(ref, mtx_ani->get_size());
            gko::matrix::CsrBuilder<value_type, index_type> l_builder(
                mtx_l_ani.get());
            gko::matrix::CsrBuilder<value_type, index_type> u_builder(
                mtx_u_ani.get());
            gko::kernels::reference::factorization::initialize_row_ptrs_l_u(
                ref, mtx_ani.get(), mtx_l_ani->get_row_ptrs(),
                mtx_u_ani->get_row_ptrs());
            auto l_nnz =
                mtx_l_ani->get_const_row_ptrs()[mtx_ani->get_size()[0]];
            auto u_nnz =
                mtx_u_ani->get_const_row_ptrs()[mtx_ani->get_size()[0]];
            l_builder.get_col_idx_array().resize_and_reset(l_nnz);
            l_builder.get_value_array().resize_and_reset(l_nnz);
            u_builder.get_col_idx_array().resize_and_reset(u_nnz);
            u_builder.get_value_array().resize_and_reset(u_nnz);
            gko::kernels::reference::factorization::initialize_l_u(
                ref, mtx_ani.get(), mtx_l_ani.get(), mtx_u_ani.get());
            mtx_ut_ani = Csr::create(ref, mtx_ani->get_size(),
                                     mtx_u_ani->get_num_stored_elements());
            gko::kernels::reference::csr::transpose(ref, mtx_u_ani.get(),
                                                    mtx_ut_ani.get());
        }
        dmtx_ani->copy_from(mtx_ani.get());
        dmtx_l_ani->copy_from(mtx_l_ani.get());
        dmtx_u_ani->copy_from(mtx_u_ani.get());
        dmtx_ut_ani->copy_from(mtx_ut_ani.get());
    }

    void test_select(const std::unique_ptr<Csr> &mtx,
                     const std::unique_ptr<Csr> &dmtx, index_type rank,
                     gko::remove_complex<value_type> tolerance = 0.0)
    {
        auto size = index_type(mtx->get_num_stored_elements());

        gko::remove_complex<value_type> res{};
        gko::remove_complex<value_type> dres{};
        gko::Array<value_type> tmp(ref);
        gko::Array<gko::remove_complex<value_type>> tmp2(ref);
        gko::Array<value_type> dtmp(omp);
        gko::Array<gko::remove_complex<value_type>> dtmp2(omp);

        gko::kernels::reference::par_ilut_factorization::threshold_select(
            ref, mtx.get(), rank, tmp, tmp2, res);
        gko::kernels::omp::par_ilut_factorization::threshold_select(
            omp, dmtx.get(), rank, dtmp, dtmp2, dres);

        ASSERT_EQ(res, dres);
    }

    void test_filter(const std::unique_ptr<Csr> &mtx,
                     const std::unique_ptr<Csr> &dmtx,
                     gko::remove_complex<value_type> threshold, bool lower)
    {
        auto res = Csr::create(ref, mtx_size);
        auto dres = Csr::create(omp, mtx_size);
        auto res_coo = Coo::create(ref, mtx_size);
        auto dres_coo = Coo::create(omp, mtx_size);
        auto local_mtx = gko::as<Csr>(lower ? mtx->clone() : mtx->transpose());
        auto local_dmtx =
            gko::as<Csr>(lower ? dmtx->clone() : dmtx->transpose());

        gko::kernels::reference::par_ilut_factorization::threshold_filter(
            ref, local_mtx.get(), threshold, res.get(), res_coo.get(), lower);
        gko::kernels::omp::par_ilut_factorization::threshold_filter(
            omp, local_dmtx.get(), threshold, dres.get(), dres_coo.get(),
            lower);

        GKO_ASSERT_MTX_NEAR(res, dres, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(res, dres);
        GKO_ASSERT_MTX_NEAR(res, res_coo, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(res, res_coo);
        GKO_ASSERT_MTX_NEAR(dres, dres_coo, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(dres, dres_coo);
    }

    void test_filter_approx(const std::unique_ptr<Csr> &mtx,
                            const std::unique_ptr<Csr> &dmtx, index_type rank)
    {
        auto res = Csr::create(ref, mtx_size);
        auto dres = Csr::create(omp, mtx_size);
        auto res_coo = Coo::create(ref, mtx_size);
        auto dres_coo = Coo::create(omp, mtx_size);

        gko::Array<value_type> tmp(ref);
        gko::Array<value_type> dtmp(omp);
        gko::remove_complex<value_type> threshold{};
        gko::remove_complex<value_type> dthreshold{};

        gko::kernels::reference::par_ilut_factorization::
            threshold_filter_approx(ref, mtx.get(), rank, tmp, threshold,
                                    res.get(), res_coo.get());
        gko::kernels::omp::par_ilut_factorization::threshold_filter_approx(
            omp, dmtx.get(), rank, dtmp, dthreshold, dres.get(),
            dres_coo.get());

        GKO_ASSERT_MTX_NEAR(res, dres, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(res, dres);
        GKO_ASSERT_MTX_NEAR(res, res_coo, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(res, res_coo);
        GKO_ASSERT_MTX_NEAR(dres, dres_coo, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(dres, dres_coo);
        ASSERT_EQ(threshold, dthreshold);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::OmpExecutor> omp;

    const gko::dim<2> mtx_size;
    std::default_random_engine rand_engine;

    std::unique_ptr<Csr> mtx1;
    std::unique_ptr<Csr> mtx2;
    std::unique_ptr<Csr> mtx_square;
    std::unique_ptr<Csr> mtx_ani;
    std::unique_ptr<Csr> mtx_l_ani;
    std::unique_ptr<Csr> mtx_u_ani;
    std::unique_ptr<Csr> mtx_ut_ani;
    std::unique_ptr<Csr> mtx_l;
    std::unique_ptr<Csr> mtx_l2;
    std::unique_ptr<Csr> mtx_u;

    std::unique_ptr<Csr> dmtx1;
    std::unique_ptr<Csr> dmtx2;
    std::unique_ptr<Csr> dmtx_square;
    std::unique_ptr<Csr> dmtx_ani;
    std::unique_ptr<Csr> dmtx_l_ani;
    std::unique_ptr<Csr> dmtx_u_ani;
    std::unique_ptr<Csr> dmtx_ut_ani;
    std::unique_ptr<Csr> dmtx_l;
    std::unique_ptr<Csr> dmtx_l2;
    std::unique_ptr<Csr> dmtx_u;
};

TYPED_TEST_CASE(ParIlut, gko::test::ValueIndexTypes);


TYPED_TEST(ParIlut, KernelThresholdSelectIsEquivalentToRef)
{
    this->test_select(this->mtx_l, this->dmtx_l,
                      this->mtx_l->get_num_stored_elements() / 3);
}


TYPED_TEST(ParIlut, KernelThresholdSelectMinIsEquivalentToRef)
{
    this->test_select(this->mtx_l, this->dmtx_l, 0);
}


TYPED_TEST(ParIlut, KernelThresholdSelectMaxIsEquivalentToRef)
{
    this->test_select(this->mtx_l, this->dmtx_l,
                      this->mtx_l->get_num_stored_elements() - 1);
}


TYPED_TEST(ParIlut, KernelThresholdFilterNullptrCooIsEquivalentToRef)
{
    using Csr = typename TestFixture::Csr;
    using Coo = typename TestFixture::Coo;
    auto res = Csr::create(this->ref, this->mtx_size);
    auto dres = Csr::create(this->omp, this->mtx_size);
    Coo *null_coo = nullptr;

    gko::kernels::reference::par_ilut_factorization::threshold_filter(
        this->ref, this->mtx_l.get(), 0.5, res.get(), null_coo, true);
    gko::kernels::omp::par_ilut_factorization::threshold_filter(
        this->omp, this->dmtx_l.get(), 0.5, dres.get(), null_coo, true);

    GKO_ASSERT_MTX_NEAR(res, dres, 0);
    GKO_ASSERT_MTX_EQ_SPARSITY(res, dres);
}


TYPED_TEST(ParIlut, KernelThresholdFilterLowerIsEquivalentToRef)
{
    this->test_filter(this->mtx_l, this->dmtx_l, 0.5, true);
}


TYPED_TEST(ParIlut, KernelThresholdFilterUpperIsEquivalentToRef)
{
    this->test_filter(this->mtx_l, this->dmtx_l, 0.5, false);
}


TYPED_TEST(ParIlut, KernelThresholdFilterNoneLowerIsEquivalentToRef)
{
    this->test_filter(this->mtx_l, this->dmtx_l, 0, true);
}


TYPED_TEST(ParIlut, KernelThresholdFilterNoneUpperIsEquivalentToRef)
{
    this->test_filter(this->mtx_l, this->dmtx_l, 0, false);
}


TYPED_TEST(ParIlut, KernelThresholdFilterAllLowerIsEquivalentToRef)
{
    this->test_filter(this->mtx_l, this->dmtx_l, 1e6, true);
}


TYPED_TEST(ParIlut, KernelThresholdFilterAllUpperIsEquivalentToRef)
{
    this->test_filter(this->mtx_l, this->dmtx_l, 1e6, false);
}


TYPED_TEST(ParIlut, KernelThresholdFilterApproxNullptrCooIsEquivalentToRef)
{
    using Csr = typename TestFixture::Csr;
    using Coo = typename TestFixture::Coo;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->test_filter(this->mtx_l, this->dmtx_l, 0.5, true);
    auto res = Csr::create(this->ref, this->mtx_size);
    auto dres = Csr::create(this->omp, this->mtx_size);
    Coo *null_coo = nullptr;
    gko::Array<value_type> tmp(this->ref);
    gko::Array<value_type> dtmp(this->omp);
    gko::remove_complex<value_type> threshold{};
    gko::remove_complex<value_type> dthreshold{};
    index_type rank{};

    gko::kernels::reference::par_ilut_factorization::threshold_filter_approx(
        this->ref, this->mtx_l.get(), rank, tmp, threshold, res.get(),
        null_coo);
    gko::kernels::omp::par_ilut_factorization::threshold_filter_approx(
        this->omp, this->dmtx_l.get(), rank, dtmp, dthreshold, dres.get(),
        null_coo);

    GKO_ASSERT_MTX_NEAR(res, dres, 0);
    GKO_ASSERT_MTX_EQ_SPARSITY(res, dres);
    ASSERT_EQ(threshold, dthreshold);
}


TYPED_TEST(ParIlut, KernelThresholdFilterApproxLowerIsEquivalentToRef)
{
    this->test_filter_approx(this->mtx_l, this->dmtx_l,
                             this->mtx_l->get_num_stored_elements() / 2);
}


TYPED_TEST(ParIlut, KernelThresholdFilterApproxNoneLowerIsEquivalentToRef)
{
    this->test_filter_approx(this->mtx_l, this->dmtx_l, 0);
}


TYPED_TEST(ParIlut, KernelThresholdFilterApproxAllLowerIsEquivalentToRef)
{
    this->test_filter_approx(this->mtx_l, this->dmtx_l,
                             this->mtx_l->get_num_stored_elements() - 1);
}


TYPED_TEST(ParIlut, KernelAddCandidatesIsEquivalentToRef)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    auto square_size = this->mtx_square->get_size();
    auto mtx_lu = Csr::create(this->ref, square_size);
    this->mtx_l2->apply(this->mtx_u.get(), mtx_lu.get());
    auto dmtx_lu = Csr::create(this->omp, square_size);
    dmtx_lu->copy_from(mtx_lu.get());
    auto res_mtx_l = Csr::create(this->ref, square_size);
    auto res_mtx_u = Csr::create(this->ref, square_size);
    auto dres_mtx_l = Csr::create(this->omp, square_size);
    auto dres_mtx_u = Csr::create(this->omp, square_size);

    gko::kernels::reference::par_ilut_factorization::add_candidates(
        this->ref, mtx_lu.get(), this->mtx_square.get(), this->mtx_l2.get(),
        this->mtx_u.get(), res_mtx_l.get(), res_mtx_u.get());
    gko::kernels::omp::par_ilut_factorization::add_candidates(
        this->omp, dmtx_lu.get(), this->dmtx_square.get(), this->dmtx_l2.get(),
        this->dmtx_u.get(), dres_mtx_l.get(), dres_mtx_u.get());

    GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx_l, dres_mtx_l);
    GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx_u, dres_mtx_u);
    GKO_ASSERT_MTX_NEAR(res_mtx_l, dres_mtx_l, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(res_mtx_u, dres_mtx_u, r<value_type>::value);
}


TYPED_TEST(ParIlut, KernelComputeLUIsEquivalentToRef)
{
    using Csr = typename TestFixture::Csr;
    using Coo = typename TestFixture::Coo;
    auto square_size = this->mtx_ani->get_size();
    auto mtx_l_coo = Coo::create(this->ref, square_size);
    auto mtx_u_coo = Coo::create(this->ref, square_size);
    this->mtx_l_ani->convert_to(mtx_l_coo.get());
    this->mtx_u_ani->convert_to(mtx_u_coo.get());
    auto dmtx_l_coo = Coo::create(this->omp, square_size);
    auto dmtx_u_coo = Coo::create(this->omp, square_size);
    dmtx_l_coo->copy_from(mtx_l_coo.get());
    dmtx_u_coo->copy_from(mtx_u_coo.get());

    gko::kernels::reference::par_ilut_factorization::compute_l_u_factors(
        this->ref, this->mtx_ani.get(), this->mtx_l_ani.get(), mtx_l_coo.get(),
        this->mtx_u_ani.get(), mtx_u_coo.get(), this->mtx_ut_ani.get());
    for (int i = 0; i < 20; ++i) {
        gko::kernels::omp::par_ilut_factorization::compute_l_u_factors(
            this->omp, this->dmtx_ani.get(), this->dmtx_l_ani.get(),
            dmtx_l_coo.get(), this->dmtx_u_ani.get(), dmtx_u_coo.get(),
            this->dmtx_ut_ani.get());
    }
    auto dmtx_utt_ani = gko::as<Csr>(this->dmtx_ut_ani->transpose());

    GKO_ASSERT_MTX_NEAR(this->mtx_l_ani, this->dmtx_l_ani, 1e-2);
    GKO_ASSERT_MTX_NEAR(this->mtx_u_ani, this->dmtx_u_ani, 1e-2);
    GKO_ASSERT_MTX_NEAR(this->dmtx_u_ani, dmtx_utt_ani, 0);
}


}  // namespace
