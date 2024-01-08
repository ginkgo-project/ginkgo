// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
#include "test/utils/executor.hpp"


template <typename ValueIndexType>
class ParIlut : public CommonTestFixture {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Dense = gko::matrix::Dense<value_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    ParIlut()
#ifdef GINKGO_FAST_TESTS
        : mtx_size(152, 231),
#else
        : mtx_size(532, 423),
#endif
          rand_engine(1337)
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
            mtx_size[0], false,
            std::uniform_int_distribution<index_type>(10, mtx_size[0]),
            std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                      1.0),
            rand_engine, ref);
        mtx_l2 = gko::test::generate_random_lower_triangular_matrix<Csr>(
            mtx_size[0], true,
            std::uniform_int_distribution<index_type>(1, mtx_size[0]),
            std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                      1.0),
            rand_engine, ref);
        mtx_u = gko::test::generate_random_upper_triangular_matrix<Csr>(
            mtx_size[0], false,
            std::uniform_int_distribution<index_type>(10, mtx_size[0]),
            std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                      1.0),
            rand_engine, ref);

        dmtx1 = gko::clone(exec, mtx1);
        dmtx2 = gko::clone(exec, mtx2);
        dmtx_square = gko::clone(exec, mtx_square);
        dmtx_ani = Csr::create(exec);
        dmtx_l_ani = Csr::create(exec);
        dmtx_u_ani = Csr::create(exec);
        dmtx_ut_ani = Csr::create(exec);
        dmtx_l = gko::clone(exec, mtx_l);
        dmtx_l2 = gko::clone(exec, mtx_l2);
        dmtx_u = gko::clone(exec, mtx_u);

        std::string file_name(gko::matrices::location_ani4_mtx);
        auto input_file = std::ifstream(file_name, std::ios::in);
        mtx_ani = gko::read<Csr>(input_file, ref);
        mtx_ani->sort_by_column_index();

        {
            mtx_l_ani = Csr::create(ref, mtx_ani->get_size());
            mtx_u_ani = Csr::create(ref, mtx_ani->get_size());
            gko::matrix::CsrBuilder<value_type, index_type> l_builder(
                mtx_l_ani);
            gko::matrix::CsrBuilder<value_type, index_type> u_builder(
                mtx_u_ani);
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
        dmtx_ani->copy_from(mtx_ani);
        dmtx_l_ani->copy_from(mtx_l_ani);
        dmtx_u_ani->copy_from(mtx_u_ani);
        dmtx_ut_ani->copy_from(mtx_ut_ani);
    }

    template <typename Mtx>
    void test_select(const std::unique_ptr<Mtx>& mtx,
                     const std::unique_ptr<Mtx>& dmtx, index_type rank)
    {
        double tolerance =
            gko::is_complex<value_type>() ? r<value_type>::value : 0.0;
        auto size = index_type(mtx->get_num_stored_elements());
        using ValueType = typename Mtx::value_type;

        gko::remove_complex<ValueType> res{};
        gko::remove_complex<ValueType> dres{};
        gko::array<ValueType> tmp(ref);
        gko::array<gko::remove_complex<ValueType>> tmp2(ref);
        gko::array<ValueType> dtmp(exec);
        gko::array<gko::remove_complex<ValueType>> dtmp2(exec);

        gko::kernels::reference::par_ilut_factorization::threshold_select(
            ref, mtx.get(), rank, tmp, tmp2, res);
        gko::kernels::EXEC_NAMESPACE::par_ilut_factorization::threshold_select(
            exec, dmtx.get(), rank, dtmp, dtmp2, dres);

        ASSERT_NEAR(res, dres, tolerance);
    }

    template <typename Mtx,
              typename Coo = gko::matrix::Coo<typename Mtx::value_type,
                                              typename Mtx::index_type>>
    void test_filter(const std::unique_ptr<Mtx>& mtx,
                     const std::unique_ptr<Mtx>& dmtx,
                     gko::remove_complex<value_type> threshold, bool lower)
    {
        auto res = Mtx::create(ref, mtx_size);
        auto dres = Mtx::create(exec, mtx_size);
        auto res_coo = Coo::create(ref, mtx_size);
        auto dres_coo = Coo::create(exec, mtx_size);
        auto local_mtx = gko::as<Mtx>(lower ? mtx->clone() : mtx->transpose());
        auto local_dmtx =
            gko::as<Mtx>(lower ? dmtx->clone() : dmtx->transpose());

        gko::kernels::reference::par_ilut_factorization::threshold_filter(
            ref, local_mtx.get(), threshold, res.get(), res_coo.get(), lower);
        gko::kernels::EXEC_NAMESPACE::par_ilut_factorization::threshold_filter(
            exec, local_dmtx.get(), threshold, dres.get(), dres_coo.get(),
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
    void test_filter_approx(const std::unique_ptr<Mtx>& mtx,
                            const std::unique_ptr<Mtx>& dmtx, index_type rank)
    {
        double tolerance =
            gko::is_complex<value_type>() ? r<value_type>::value : 0.0;
        auto res = Mtx::create(ref, mtx_size);
        auto dres = Mtx::create(exec, mtx_size);
        auto res_coo = Coo::create(ref, mtx_size);
        auto dres_coo = Coo::create(exec, mtx_size);
        using ValueType = typename Mtx::value_type;

        gko::array<ValueType> tmp(ref);
        gko::array<ValueType> dtmp(exec);
        gko::remove_complex<ValueType> threshold{};
        gko::remove_complex<ValueType> dthreshold{};

        gko::kernels::reference::par_ilut_factorization::
            threshold_filter_approx(ref, mtx.get(), rank, tmp, threshold,
                                    res.get(), res_coo.get());
        gko::kernels::EXEC_NAMESPACE::par_ilut_factorization::
            threshold_filter_approx(exec, dmtx.get(), rank, dtmp, dthreshold,
                                    dres.get(), dres_coo.get());

        GKO_ASSERT_MTX_NEAR(res, dres, 0);
        GKO_ASSERT_MTX_NEAR(res, res_coo, 0);
        GKO_ASSERT_MTX_NEAR(dres, dres_coo, 0);
        if (tolerance > 0.0) {
            GKO_ASSERT_MTX_EQ_SPARSITY(res, dres);
            GKO_ASSERT_MTX_EQ_SPARSITY(res, res_coo);
            GKO_ASSERT_MTX_EQ_SPARSITY(dres, dres_coo);
        }
        ASSERT_NEAR(threshold, dthreshold, tolerance);
    }

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

TYPED_TEST_SUITE(ParIlut, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


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
    auto dres = Csr::create(this->exec, this->mtx_size);
    Coo* null_coo = nullptr;

    gko::kernels::reference::par_ilut_factorization::threshold_filter(
        this->ref, this->mtx_l.get(), 0.5, res.get(), null_coo, true);
    gko::kernels::EXEC_NAMESPACE::par_ilut_factorization::threshold_filter(
        this->exec, this->dmtx_l.get(), 0.5, dres.get(), null_coo, true);

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
    auto dres = Csr::create(this->exec, this->mtx_size);
    Coo* null_coo = nullptr;
    gko::array<value_type> tmp(this->ref);
    gko::array<value_type> dtmp(this->exec);
    gko::remove_complex<value_type> threshold{};
    gko::remove_complex<value_type> dthreshold{};
    index_type rank{};

    gko::kernels::reference::par_ilut_factorization::threshold_filter_approx(
        this->ref, this->mtx_l.get(), rank, tmp, threshold, res.get(),
        null_coo);
    gko::kernels::EXEC_NAMESPACE::par_ilut_factorization::
        threshold_filter_approx(this->exec, this->dmtx_l.get(), rank, dtmp,
                                dthreshold, dres.get(), null_coo);

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
    this->mtx_l2->apply(this->mtx_u, mtx_lu);
    auto dmtx_lu = Csr::create(this->exec, square_size);
    dmtx_lu->copy_from(mtx_lu);
    auto res_mtx_l = Csr::create(this->ref, square_size);
    auto res_mtx_u = Csr::create(this->ref, square_size);
    auto dres_mtx_l = Csr::create(this->exec, square_size);
    auto dres_mtx_u = Csr::create(this->exec, square_size);

    gko::kernels::reference::par_ilut_factorization::add_candidates(
        this->ref, mtx_lu.get(), this->mtx_square.get(), this->mtx_l2.get(),
        this->mtx_u.get(), res_mtx_l.get(), res_mtx_u.get());
    gko::kernels::EXEC_NAMESPACE::par_ilut_factorization::add_candidates(
        this->exec, dmtx_lu.get(), this->dmtx_square.get(), this->dmtx_l2.get(),
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
    this->mtx_l_ani->convert_to(mtx_l_coo);
    this->mtx_u_ani->convert_to(mtx_u_coo);
    auto dmtx_l_coo = Coo::create(this->exec, square_size);
    auto dmtx_u_coo = Coo::create(this->exec, square_size);
    dmtx_l_coo->copy_from(mtx_l_coo);
    dmtx_u_coo->copy_from(mtx_u_coo);

    gko::kernels::reference::par_ilut_factorization::compute_l_u_factors(
        this->ref, this->mtx_ani.get(), this->mtx_l_ani.get(), mtx_l_coo.get(),
        this->mtx_u_ani.get(), mtx_u_coo.get(), this->mtx_ut_ani.get());
    for (int i = 0; i < 20; ++i) {
        gko::kernels::EXEC_NAMESPACE::par_ilut_factorization::
            compute_l_u_factors(this->exec, this->dmtx_ani.get(),
                                this->dmtx_l_ani.get(), dmtx_l_coo.get(),
                                this->dmtx_u_ani.get(), dmtx_u_coo.get(),
                                this->dmtx_ut_ani.get());
    }
    auto dmtx_utt_ani = gko::as<Csr>(this->dmtx_ut_ani->transpose());

    GKO_ASSERT_MTX_NEAR(this->mtx_l_ani, this->dmtx_l_ani, 1e-2);
    GKO_ASSERT_MTX_NEAR(this->mtx_u_ani, this->dmtx_u_ani, 1e-2);
    GKO_ASSERT_MTX_NEAR(this->dmtx_u_ani, dmtx_utt_ani, 0);
}
