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
#include "core/factorization/par_ilu_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "hip/test/utils.hip.hpp"
#include "matrices/config.hpp"


namespace {


class ParIlut : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Dense = gko::matrix::Dense<value_type>;
    using ComplexDense = gko::matrix::Dense<std::complex<value_type>>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using ComplexCsr = gko::matrix::Csr<std::complex<value_type>, index_type>;

    ParIlut()
        : mtx_size(500, 700),
          rand_engine(1337),
          ref(gko::ReferenceExecutor::create()),
          hip(gko::HipExecutor::create(0, ref))
    {
        mtx1 = gko::test::generate_random_matrix<Csr>(
            mtx_size[0], mtx_size[1],
            std::uniform_int_distribution<>(10, mtx_size[1]),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        mtx2 = gko::test::generate_random_matrix<Csr>(
            mtx_size[0], mtx_size[1],
            std::uniform_int_distribution<>(0, mtx_size[1]),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        mtx_square = gko::test::generate_random_matrix<Csr>(
            mtx_size[0], mtx_size[0],
            std::uniform_int_distribution<>(1, mtx_size[0]),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        mtx_l = gko::test::generate_random_lower_triangular_matrix<Csr>(
            mtx_size[0], mtx_size[0], false,
            std::uniform_int_distribution<>(1, mtx_size[0]),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        mtx_l2 = gko::test::generate_random_lower_triangular_matrix<Csr>(
            mtx_size[0], mtx_size[0], true,
            std::uniform_int_distribution<>(1, mtx_size[0]),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        mtx_l_complex =
            gko::test::generate_random_lower_triangular_matrix<ComplexCsr>(
                mtx_size[0], mtx_size[0], false,
                std::uniform_int_distribution<>(10, mtx_size[0]),
                std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        mtx_u = gko::test::generate_random_upper_triangular_matrix<Csr>(
            mtx_size[0], mtx_size[0], false,
            std::uniform_int_distribution<>(10, mtx_size[0]),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
        mtx_u_complex =
            gko::test::generate_random_upper_triangular_matrix<ComplexCsr>(
                mtx_size[0], mtx_size[0], false,
                std::uniform_int_distribution<>(10, mtx_size[0]),
                std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);


        dmtx1 = Csr::create(hip);
        dmtx1->copy_from(mtx1.get());
        dmtx2 = Csr::create(hip);
        dmtx2->copy_from(mtx2.get());
        dmtx_square = Csr::create(hip);
        dmtx_square->copy_from(mtx_square.get());
        dmtx_ani = Csr::create(hip);
        dmtx_l_ani = Csr::create(hip);
        dmtx_u_ani = Csr::create(hip);
        dmtx_ut_ani = Csr::create(hip);
        dmtx_l = Csr::create(hip);
        dmtx_l->copy_from(mtx_l.get());
        dmtx_l2 = Csr::create(hip);
        dmtx_l2->copy_from(mtx_l2.get());
        dmtx_u = Csr::create(hip);
        dmtx_u->copy_from(mtx_u.get());
        dmtx_l_complex = ComplexCsr::create(hip);
        dmtx_l_complex->copy_from(mtx_l_complex.get());
        dmtx_u_complex = ComplexCsr::create(hip);
        dmtx_u_complex->copy_from(mtx_u_complex.get());
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

    template <typename Mtx>
    void test_select(const std::unique_ptr<Mtx> &mtx,
                     const std::unique_ptr<Mtx> &dmtx, index_type rank,
                     value_type tolerance = 0.0)
    {
        auto size = index_type(mtx->get_num_stored_elements());
        using ValueType = typename Mtx::value_type;

        gko::remove_complex<ValueType> res{};
        gko::remove_complex<ValueType> dres{};
        gko::Array<ValueType> tmp(ref);
        gko::Array<gko::remove_complex<ValueType>> tmp2(ref);
        gko::Array<ValueType> dtmp(hip);
        gko::Array<gko::remove_complex<ValueType>> dtmp2(hip);

        gko::kernels::reference::par_ilut_factorization::threshold_select(
            ref, mtx.get(), rank, tmp, tmp2, res);
        gko::kernels::hip::par_ilut_factorization::threshold_select(
            hip, dmtx.get(), rank, dtmp, dtmp2, dres);

        ASSERT_NEAR(res, dres, tolerance);
    }

    template <typename Mtx,
              typename Coo = gko::matrix::Coo<typename Mtx::value_type,
                                              typename Mtx::index_type>>
    void test_filter(const std::unique_ptr<Mtx> &mtx,
                     const std::unique_ptr<Mtx> &dmtx, value_type threshold,
                     bool lower)
    {
        auto res = Mtx::create(ref, mtx_size);
        auto dres = Mtx::create(hip, mtx_size);
        auto res_coo = Coo::create(ref, mtx_size);
        auto dres_coo = Coo::create(hip, mtx_size);
        auto local_mtx = gko::as<Mtx>(lower ? mtx->clone() : mtx->transpose());
        auto local_dmtx =
            gko::as<Mtx>(lower ? dmtx->clone() : dmtx->transpose());

        gko::kernels::reference::par_ilut_factorization::threshold_filter(
            ref, local_mtx.get(), threshold, res.get(), res_coo.get(), lower);
        gko::kernels::hip::par_ilut_factorization::threshold_filter(
            hip, local_dmtx.get(), threshold, dres.get(), dres_coo.get(),
            lower);

        GKO_ASSERT_MTX_NEAR(res, dres, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(res, dres);
        GKO_ASSERT_MTX_NEAR(res, res_coo, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(res, res_coo);
        GKO_ASSERT_MTX_NEAR(dres, dres_coo, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(dres, dres_coo);
    }

    template <typename Mtx,
              typename Coo = gko::matrix::Coo<typename Mtx::value_type,
                                              typename Mtx::index_type>>
    void test_filter_approx(const std::unique_ptr<Mtx> &mtx,
                            const std::unique_ptr<Mtx> &dmtx, index_type rank,
                            value_type tolerance = 0.0)
    {
        auto res = Mtx::create(ref, mtx_size);
        auto dres = Mtx::create(hip, mtx_size);
        auto res_coo = Coo::create(ref, mtx_size);
        auto dres_coo = Coo::create(hip, mtx_size);
        using ValueType = typename Mtx::value_type;

        gko::Array<ValueType> tmp(ref);
        gko::Array<ValueType> dtmp(hip);
        gko::remove_complex<ValueType> threshold{};
        gko::remove_complex<ValueType> dthreshold{};

        gko::kernels::reference::par_ilut_factorization::
            threshold_filter_approx(ref, mtx.get(), rank, tmp, threshold,
                                    res.get(), res_coo.get());
        gko::kernels::hip::par_ilut_factorization::threshold_filter_approx(
            hip, dmtx.get(), rank, dtmp, dthreshold, dres.get(),
            dres_coo.get());

        GKO_ASSERT_MTX_NEAR(res, dres, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(res, dres);
        GKO_ASSERT_MTX_NEAR(res, res_coo, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(res, res_coo);
        GKO_ASSERT_MTX_NEAR(dres, dres_coo, 0);
        GKO_ASSERT_MTX_EQ_SPARSITY(dres, dres_coo);
        ASSERT_NEAR(threshold, dthreshold, tolerance);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::HipExecutor> hip;

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
    std::unique_ptr<ComplexCsr> mtx_l_complex;
    std::unique_ptr<Csr> mtx_u;
    std::unique_ptr<ComplexCsr> mtx_u_complex;

    std::unique_ptr<Csr> dmtx1;
    std::unique_ptr<Csr> dmtx2;
    std::unique_ptr<Csr> dmtx_square;
    std::unique_ptr<Csr> dmtx_ani;
    std::unique_ptr<Csr> dmtx_l_ani;
    std::unique_ptr<Csr> dmtx_u_ani;
    std::unique_ptr<Csr> dmtx_ut_ani;
    std::unique_ptr<Csr> dmtx_l;
    std::unique_ptr<Csr> dmtx_l2;
    std::unique_ptr<ComplexCsr> dmtx_l_complex;
    std::unique_ptr<Csr> dmtx_u;
    std::unique_ptr<ComplexCsr> dmtx_u_complex;
};


TEST_F(ParIlut, KernelThresholdSelectIsEquivalentToRef)
{
    test_select(mtx_l, dmtx_l, mtx_l->get_num_stored_elements() / 3);
}


TEST_F(ParIlut, KernelThresholdSelectMinIsEquivalentToRef)
{
    test_select(mtx_l, dmtx_l, 0);
}


TEST_F(ParIlut, KernelThresholdSelectMaxIsEquivalentToRef)
{
    test_select(mtx_l, dmtx_l, mtx_l->get_num_stored_elements() - 1);
}


TEST_F(ParIlut, KernelComplexThresholdSelectIsEquivalentToRef)
{
    test_select(mtx_l_complex, dmtx_l_complex,
                mtx_l_complex->get_num_stored_elements() / 3, 1e-14);
}


TEST_F(ParIlut, KernelComplexThresholdSelectMinIsEquivalentToRef)
{
    test_select(mtx_l_complex, dmtx_l_complex, 0, 1e-14);
}


TEST_F(ParIlut, KernelComplexThresholdSelectMaxLowerIsEquivalentToRef)
{
    test_select(mtx_l_complex, dmtx_l_complex,
                mtx_l_complex->get_num_stored_elements() - 1, 1e-14);
}


TEST_F(ParIlut, KernelThresholdFilterNullptrCooIsEquivalentToRef)
{
    auto res = Csr::create(ref, mtx_size);
    auto dres = Csr::create(hip, mtx_size);
    Coo *null_coo = nullptr;

    gko::kernels::reference::par_ilut_factorization::threshold_filter(
        ref, mtx_l.get(), 0.5, res.get(), null_coo, true);
    gko::kernels::hip::par_ilut_factorization::threshold_filter(
        hip, dmtx_l.get(), 0.5, dres.get(), null_coo, true);

    GKO_ASSERT_MTX_NEAR(res, dres, 0);
    GKO_ASSERT_MTX_EQ_SPARSITY(res, dres);
}


TEST_F(ParIlut, KernelThresholdFilterLowerIsEquivalentToRef)
{
    test_filter(mtx_l, dmtx_l, 0.5, true);
}


TEST_F(ParIlut, KernelThresholdFilterUpperIsEquivalentToRef)
{
    test_filter(mtx_l, dmtx_l, 0.5, false);
}


TEST_F(ParIlut, KernelThresholdFilterNoneLowerIsEquivalentToRef)
{
    test_filter(mtx_l, dmtx_l, 0, true);
}


TEST_F(ParIlut, KernelThresholdFilterNoneUpperIsEquivalentToRef)
{
    test_filter(mtx_l, dmtx_l, 0, false);
}


TEST_F(ParIlut, KernelThresholdFilterAllLowerIsEquivalentToRef)
{
    test_filter(mtx_l, dmtx_l, 1e6, true);
}


TEST_F(ParIlut, KernelThresholdFilterAllUpperIsEquivalentToRef)
{
    test_filter(mtx_l, dmtx_l, 1e6, false);
}


TEST_F(ParIlut, KernelComplexThresholdFilterLowerIsEquivalentToRef)
{
    test_filter(mtx_l_complex, dmtx_l_complex, 0.5, true);
}


TEST_F(ParIlut, KernelComplexThresholdFilterNoneLowerIsEquivalentToRef)
{
    test_filter(mtx_l_complex, dmtx_l_complex, 0, true);
}


TEST_F(ParIlut, KernelComplexThresholdFilterAllLowerIsEquivalentToRef)
{
    test_filter(mtx_l_complex, dmtx_l_complex, 1e6, true);
}


#if defined(hipsparseVersionMajor) && defined(hipsparseVersionMinor) && \
    ((hipsparseVersionMajor > 1) ||                                     \
     (hipsparseVersionMajor == 1 && hipsparseVersionMinor >= 4))
TEST_F(ParIlut, KernelComplexThresholdFilterUpperIsEquivalentToRef)
{
    test_filter(mtx_l_complex, dmtx_l_complex, 0.5, false);
}


TEST_F(ParIlut, KernelComplexThresholdFilterNoneUpperIsEquivalentToRef)
{
    test_filter(mtx_l_complex, dmtx_l_complex, 0, false);
}


TEST_F(ParIlut, KernelComplexThresholdFilterAllUppererIsEquivalentToRef)
{
    test_filter(mtx_l_complex, dmtx_l_complex, 1e6, false);
}
#endif  // hipsparse version >= 1.4


TEST_F(ParIlut, KernelThresholdFilterApproxNullptrCooIsEquivalentToRef)
{
    test_filter(mtx_l, dmtx_l, 0.5, true);
    auto res = Csr::create(ref, mtx_size);
    auto dres = Csr::create(hip, mtx_size);
    Coo *null_coo = nullptr;
    gko::Array<value_type> tmp(ref);
    gko::Array<value_type> dtmp(hip);
    gko::remove_complex<value_type> threshold{};
    gko::remove_complex<value_type> dthreshold{};
    index_type rank{};

    gko::kernels::reference::par_ilut_factorization::threshold_filter_approx(
        ref, mtx_l.get(), rank, tmp, threshold, res.get(), null_coo);
    gko::kernels::hip::par_ilut_factorization::threshold_filter_approx(
        hip, dmtx_l.get(), rank, dtmp, dthreshold, dres.get(), null_coo);

    GKO_ASSERT_MTX_NEAR(res, dres, 0);
    GKO_ASSERT_MTX_EQ_SPARSITY(res, dres);
    ASSERT_EQ(threshold, dthreshold);
}


TEST_F(ParIlut, KernelThresholdFilterApproxLowerIsEquivalentToRef)
{
    test_filter_approx(mtx_l, dmtx_l, mtx_l->get_num_stored_elements() / 2);
}


TEST_F(ParIlut, KernelThresholdFilterApproxNoneLowerIsEquivalentToRef)
{
    test_filter_approx(mtx_l, dmtx_l, 0);
}


TEST_F(ParIlut, KernelThresholdFilterApproxAllLowerIsEquivalentToRef)
{
    test_filter_approx(mtx_l, dmtx_l, mtx_l->get_num_stored_elements() - 1);
}


TEST_F(ParIlut, KernelComplexThresholdFilterApproxLowerIsEquivalentToRef)
{
    test_filter_approx(mtx_l_complex, dmtx_l_complex,
                       mtx_l_complex->get_num_stored_elements() / 2,
                       r<value_type>::value);
}


TEST_F(ParIlut, KernelComplexThresholdFilterApproxNoneLowerIsEquivalentToRef)
{
    test_filter_approx(mtx_l_complex, dmtx_l_complex, 0, r<value_type>::value);
}


TEST_F(ParIlut, KernelComplexThresholdFilterApproxAllLowerIsEquivalentToRef)
{
    test_filter_approx(mtx_l_complex, dmtx_l_complex,
                       mtx_l_complex->get_num_stored_elements() - 1,
                       r<value_type>::value);
}


TEST_F(ParIlut, KernelAddCandidatesIsEquivalentToRef)
{
    auto square_size = mtx_square->get_size();
    auto mtx_lu = Csr::create(ref, square_size);
    mtx_l2->apply(mtx_u.get(), mtx_lu.get());
    auto dmtx_lu = Csr::create(hip, square_size);
    dmtx_lu->copy_from(mtx_lu.get());
    auto res_mtx_l = Csr::create(ref, square_size);
    auto res_mtx_u = Csr::create(ref, square_size);
    auto dres_mtx_l = Csr::create(hip, square_size);
    auto dres_mtx_u = Csr::create(hip, square_size);

    gko::kernels::reference::par_ilut_factorization::add_candidates(
        ref, mtx_lu.get(), mtx_square.get(), mtx_l2.get(), mtx_u.get(),
        res_mtx_l.get(), res_mtx_u.get());
    gko::kernels::hip::par_ilut_factorization::add_candidates(
        hip, dmtx_lu.get(), dmtx_square.get(), dmtx_l2.get(), dmtx_u.get(),
        dres_mtx_l.get(), dres_mtx_u.get());

    GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx_l, dres_mtx_l);
    GKO_ASSERT_MTX_EQ_SPARSITY(res_mtx_u, dres_mtx_u);
    GKO_ASSERT_MTX_NEAR(res_mtx_l, dres_mtx_l, 1e-14);
    GKO_ASSERT_MTX_NEAR(res_mtx_u, dres_mtx_u, 1e-14);
}


TEST_F(ParIlut, KernelComputeLUIsEquivalentToRef)
{
    auto square_size = mtx_ani->get_size();
    auto mtx_l_coo = Coo::create(ref, square_size);
    auto mtx_u_coo = Coo::create(ref, square_size);
    mtx_l_ani->convert_to(mtx_l_coo.get());
    mtx_u_ani->convert_to(mtx_u_coo.get());
    auto dmtx_l_coo = Coo::create(hip, square_size);
    auto dmtx_u_coo = Coo::create(hip, square_size);
    dmtx_l_coo->copy_from(mtx_l_coo.get());
    dmtx_u_coo->copy_from(mtx_u_coo.get());

    gko::kernels::reference::par_ilut_factorization::compute_l_u_factors(
        ref, mtx_ani.get(), mtx_l_ani.get(), mtx_l_coo.get(), mtx_u_ani.get(),
        mtx_u_coo.get(), mtx_ut_ani.get());
    for (int i = 0; i < 20; ++i) {
        gko::kernels::hip::par_ilut_factorization::compute_l_u_factors(
            hip, dmtx_ani.get(), dmtx_l_ani.get(), dmtx_l_coo.get(),
            dmtx_u_ani.get(), dmtx_u_coo.get(), dmtx_ut_ani.get());
    }
    auto dmtx_utt_ani = gko::as<Csr>(dmtx_ut_ani->transpose());

    GKO_ASSERT_MTX_NEAR(mtx_l_ani, dmtx_l_ani, 1e-2);
    GKO_ASSERT_MTX_NEAR(mtx_u_ani, dmtx_u_ani, 1e-2);
    GKO_ASSERT_MTX_NEAR(dmtx_u_ani, dmtx_utt_ani, 0);
}


}  // namespace
