// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/isai_kernels.hpp"


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>


#include "core/test/utils.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


enum struct matrix_type { lower, upper, general, spd };


class Isai : public CommonTestFixture {
protected:
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    Isai() : rand_engine(42) {}

    std::unique_ptr<Csr> clone_allocations(const Csr* csr_mtx)
    {
        if (csr_mtx->get_executor() != ref) {
            return {nullptr};
        }
        const auto num_elems = csr_mtx->get_num_stored_elements();
        auto sparsity = csr_mtx->clone();

        // values are now filled with invalid data to catch potential errors
        auto begin_values = sparsity->get_values();
        auto end_values = begin_values + num_elems;
        std::fill(begin_values, end_values, -gko::one<value_type>());
        return sparsity;
    }

    void initialize_data(matrix_type type, gko::size_type n,
                         gko::size_type row_limit)
    {
        const bool for_lower_tm = type == matrix_type::lower;
        auto nz_dist = std::uniform_int_distribution<index_type>(1, row_limit);
        auto val_dist = std::uniform_real_distribution<value_type>(-1., 1.);
        mtx = Csr::create(ref);
        if (type == matrix_type::general) {
            auto dense_mtx = gko::test::generate_random_matrix<Dense>(
                n, n, nz_dist, val_dist, rand_engine, ref, gko::dim<2>{n, n});
            ensure_diagonal(dense_mtx.get());
            mtx->copy_from(dense_mtx);
        } else if (type == matrix_type::spd) {
            auto dense_mtx = gko::test::generate_random_band_matrix<Dense>(
                n, row_limit / 4, row_limit / 4, val_dist, rand_engine, ref,
                gko::dim<2>{n, n});
            auto transp = gko::as<Dense>(dense_mtx->transpose());
            auto spd_mtx = Dense::create(ref, gko::dim<2>{n, n});
            dense_mtx->apply(transp, spd_mtx);
            mtx->copy_from(spd_mtx);
        } else {
            mtx = gko::test::generate_random_triangular_matrix<Csr>(
                n, true, for_lower_tm, nz_dist, val_dist, rand_engine, ref,
                gko::dim<2>{n, n});
        }
        inverse = clone_allocations(mtx.get());

        d_mtx = gko::clone(exec, mtx);
        d_inverse = gko::clone(exec, inverse);
    }

    void initialize_tridiag_data(matrix_type type, gko::size_type n)
    {
        auto val_dist = std::uniform_real_distribution<value_type>(0., 1.);
        mtx = Csr::create(ref);
        auto dense_mtx = gko::test::generate_random_band_matrix<Dense>(
            n, 1, 1, val_dist, rand_engine, ref);
        ensure_diagonal(dense_mtx.get());
        mtx->copy_from(dense_mtx);
        inverse = clone_allocations(mtx.get());

        d_mtx = gko::clone(exec, mtx);
        d_inverse = gko::clone(exec, inverse);
    }

    void ensure_diagonal(Dense* mtx)
    {
        for (int i = 0; i < mtx->get_size()[0]; ++i) {
            mtx->at(i, i) = gko::one<Dense::value_type>();
        }
    }

    std::default_random_engine rand_engine;

    std::unique_ptr<Csr> mtx;
    std::unique_ptr<Csr> inverse;

    std::unique_ptr<Csr> d_mtx;
    std::unique_ptr<Csr> d_inverse;
};


TEST_F(Isai, IsaiGenerateLinverseShortIsEquivalentToRef)
{
    initialize_data(matrix_type::lower, 536, 31);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(exec, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::kernels::EXEC_NAMESPACE::isai::generate_tri_inverse(
        exec, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        true);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_EQ(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, IsaiGenerateUinverseShortIsEquivalentToRef)
{
    initialize_data(matrix_type::upper, 615, 31);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(exec, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::kernels::EXEC_NAMESPACE::isai::generate_tri_inverse(
        exec, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        false);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_EQ(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, IsaiGenerateAinverseShortIsEquivalentToRef)
{
    initialize_data(matrix_type::general, 615, 15);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(exec, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::kernels::EXEC_NAMESPACE::isai::generate_general_inverse(
        exec, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        false);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_EQ(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, IsaiGenerateSpdinverseShortIsEquivalentToRef)
{
    initialize_data(matrix_type::spd, 100, 15);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(exec, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::kernels::EXEC_NAMESPACE::isai::generate_general_inverse(
        exec, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        true);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 15 * r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_EQ(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, IsaiGenerateLinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::lower, 554, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(exec, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::kernels::EXEC_NAMESPACE::isai::generate_tri_inverse(
        exec, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        true);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_GT(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, IsaiGenerateUinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::upper, 695, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(exec, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::kernels::EXEC_NAMESPACE::isai::generate_tri_inverse(
        exec, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        false);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_GT(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, IsaiGenerateAinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::general, 695, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(exec, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::kernels::EXEC_NAMESPACE::isai::generate_general_inverse(
        exec, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        false);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 100 * r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_GT(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, IsaiGenerateSpdinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::spd, 100, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(exec, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::kernels::EXEC_NAMESPACE::isai::generate_general_inverse(
        exec, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        false);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 10 * r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_GT(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, IsaiGenerateExcessLinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::lower, 518, 40);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::array<index_type> da1(exec, a1);
    gko::array<index_type> da2(exec, a2);
    auto e_dim = a1.get_data()[num_rows];
    auto e_nnz = a2.get_data()[num_rows];
    auto excess = Csr::create(ref, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    auto dexcess = Csr::create(exec, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto de_rhs = Dense::create(exec, gko::dim<2>(e_dim, 1));

    gko::kernels::reference::isai::generate_excess_system(
        ref, mtx.get(), inverse.get(), a1.get_const_data(), a2.get_const_data(),
        excess.get(), e_rhs.get(), 0, num_rows);
    gko::kernels::EXEC_NAMESPACE::isai::generate_excess_system(
        exec, d_mtx.get(), d_inverse.get(), da1.get_const_data(),
        da2.get_const_data(), dexcess.get(), de_rhs.get(), 0, num_rows);

    GKO_ASSERT_MTX_EQ_SPARSITY(excess, dexcess);
    GKO_ASSERT_MTX_NEAR(excess, dexcess, 0);
    GKO_ASSERT_MTX_NEAR(e_rhs, de_rhs, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, IsaiGenerateExcessUinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::upper, 673, 51);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::array<index_type> da1(exec, a1);
    gko::array<index_type> da2(exec, a2);
    auto e_dim = a1.get_data()[num_rows];
    auto e_nnz = a2.get_data()[num_rows];
    auto excess = Csr::create(ref, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    auto dexcess = Csr::create(exec, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto de_rhs = Dense::create(exec, gko::dim<2>(e_dim, 1));

    gko::kernels::reference::isai::generate_excess_system(
        ref, mtx.get(), inverse.get(), a1.get_const_data(), a2.get_const_data(),
        excess.get(), e_rhs.get(), 0, num_rows);
    gko::kernels::EXEC_NAMESPACE::isai::generate_excess_system(
        exec, d_mtx.get(), d_inverse.get(), da1.get_const_data(),
        da2.get_const_data(), dexcess.get(), de_rhs.get(), 0, num_rows);

    GKO_ASSERT_MTX_EQ_SPARSITY(excess, dexcess);
    GKO_ASSERT_MTX_NEAR(excess, dexcess, 0);
    GKO_ASSERT_MTX_NEAR(e_rhs, de_rhs, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, IsaiGenerateExcessAinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::general, 100, 51);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::array<index_type> da1(exec, a1);
    gko::array<index_type> da2(exec, a2);
    auto e_dim = a1.get_data()[num_rows];
    auto e_nnz = a2.get_data()[num_rows];
    auto excess = Csr::create(ref, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    auto dexcess = Csr::create(exec, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto de_rhs = Dense::create(exec, gko::dim<2>(e_dim, 1));

    gko::kernels::reference::isai::generate_excess_system(
        ref, mtx.get(), inverse.get(), a1.get_const_data(), a2.get_const_data(),
        excess.get(), e_rhs.get(), 0, num_rows);
    gko::kernels::EXEC_NAMESPACE::isai::generate_excess_system(
        exec, d_mtx.get(), d_inverse.get(), da1.get_const_data(),
        da2.get_const_data(), dexcess.get(), de_rhs.get(), 0, num_rows);

    GKO_ASSERT_MTX_EQ_SPARSITY(excess, dexcess);
    GKO_ASSERT_MTX_NEAR(excess, dexcess, 0);
    GKO_ASSERT_MTX_NEAR(e_rhs, de_rhs, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, IsaiGenerateExcessSpdinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::spd, 100, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::array<index_type> da1(exec, a1);
    gko::array<index_type> da2(exec, a2);
    auto e_dim = a1.get_data()[num_rows];
    auto e_nnz = a2.get_data()[num_rows];
    auto excess = Csr::create(ref, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    auto dexcess = Csr::create(exec, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto de_rhs = Dense::create(exec, gko::dim<2>(e_dim, 1));

    gko::kernels::reference::isai::generate_excess_system(
        ref, mtx.get(), inverse.get(), a1.get_const_data(), a2.get_const_data(),
        excess.get(), e_rhs.get(), 0, num_rows);
    gko::kernels::EXEC_NAMESPACE::isai::generate_excess_system(
        exec, d_mtx.get(), d_inverse.get(), da1.get_const_data(),
        da2.get_const_data(), dexcess.get(), de_rhs.get(), 0, num_rows);

    GKO_ASSERT_MTX_EQ_SPARSITY(excess, dexcess);
    GKO_ASSERT_MTX_NEAR(excess, dexcess, 0);
    GKO_ASSERT_MTX_NEAR(e_rhs, de_rhs, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, IsaiGeneratePartialExcessIsEquivalentToRef)
{
    initialize_data(matrix_type::general, 100, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::array<index_type> da1(exec, a1);
    gko::array<index_type> da2(exec, a2);
    auto e_dim = a1.get_data()[10] - a1.get_data()[5];
    auto e_nnz = a2.get_data()[10] - a2.get_data()[5];
    auto excess = Csr::create(ref, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    auto dexcess = Csr::create(exec, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto de_rhs = Dense::create(exec, gko::dim<2>(e_dim, 1));

    gko::kernels::reference::isai::generate_excess_system(
        ref, mtx.get(), inverse.get(), a1.get_const_data(), a2.get_const_data(),
        excess.get(), e_rhs.get(), 5u, 10u);
    gko::kernels::EXEC_NAMESPACE::isai::generate_excess_system(
        exec, d_mtx.get(), d_inverse.get(), da1.get_const_data(),
        da2.get_const_data(), dexcess.get(), de_rhs.get(), 5u, 10u);

    GKO_ASSERT_MTX_EQ_SPARSITY(excess, dexcess);
    GKO_ASSERT_MTX_NEAR(excess, dexcess, 0);
    GKO_ASSERT_MTX_NEAR(e_rhs, de_rhs, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, IsaiScaleExcessSolutionIsEquivalentToRef)
{
    initialize_data(matrix_type::spd, 100, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::array<index_type> da1(exec, a1);
    auto e_dim = a1.get_data()[num_rows];
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    std::fill_n(e_rhs->get_values(), e_dim, 123456);
    auto de_rhs = gko::clone(exec, e_rhs);
    d_inverse->copy_from(inverse);

    gko::kernels::reference::isai::scale_excess_solution(
        ref, a1.get_const_data(), e_rhs.get(), 0, num_rows);
    gko::kernels::EXEC_NAMESPACE::isai::scale_excess_solution(
        exec, da1.get_const_data(), de_rhs.get(), 0, num_rows);

    GKO_ASSERT_MTX_NEAR(e_rhs, de_rhs, 0);
}


TEST_F(Isai, IsaiScalePartialExcessSolutionIsEquivalentToRef)
{
    initialize_data(matrix_type::spd, 100, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::array<index_type> da1(exec, a1);
    auto e_dim = a1.get_data()[10] - a1.get_data()[5];
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    std::fill_n(e_rhs->get_values(), e_dim, 123456);
    auto de_rhs = gko::clone(exec, e_rhs);

    gko::kernels::reference::isai::scale_excess_solution(
        ref, a1.get_const_data(), e_rhs.get(), 5u, 10u);
    gko::kernels::EXEC_NAMESPACE::isai::scale_excess_solution(
        exec, da1.get_const_data(), de_rhs.get(), 5u, 10u);

    GKO_ASSERT_MTX_NEAR(e_rhs, de_rhs, 0);
}


TEST_F(Isai, IsaiScatterExcessSolutionLIsEquivalentToRef)
{
    initialize_data(matrix_type::lower, 572, 52);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::array<index_type> da1(exec, a1);
    auto e_dim = a1.get_data()[num_rows];
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    std::fill_n(e_rhs->get_values(), e_dim, 123456);
    auto de_rhs = gko::clone(exec, e_rhs);
    d_inverse->copy_from(inverse);

    gko::kernels::reference::isai::scatter_excess_solution(
        ref, a1.get_const_data(), e_rhs.get(), inverse.get(), 0, num_rows);
    gko::kernels::EXEC_NAMESPACE::isai::scatter_excess_solution(
        exec, da1.get_const_data(), de_rhs.get(), d_inverse.get(), 0, num_rows);

    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, IsaiScatterExcessSolutionUIsEquivalentToRef)
{
    initialize_data(matrix_type::upper, 702, 45);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::array<index_type> da1(exec, a1);
    auto e_dim = a1.get_data()[num_rows];
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    std::fill_n(e_rhs->get_values(), e_dim, 123456);
    auto de_rhs = gko::clone(exec, e_rhs);
    // overwrite -1 values with inverse
    d_inverse->copy_from(inverse);

    gko::kernels::reference::isai::scatter_excess_solution(
        ref, a1.get_const_data(), e_rhs.get(), inverse.get(), 0, num_rows);
    gko::kernels::EXEC_NAMESPACE::isai::scatter_excess_solution(
        exec, da1.get_const_data(), de_rhs.get(), d_inverse.get(), 0, num_rows);

    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, IsaiScatterExcessSolutionAIsEquivalentToRef)
{
    initialize_data(matrix_type::general, 702, 45);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::array<index_type> da1(exec, a1);
    auto e_dim = a1.get_data()[num_rows];
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    std::fill_n(e_rhs->get_values(), e_dim, 123456);
    auto de_rhs = gko::clone(exec, e_rhs);
    // overwrite -1 values with inverse
    d_inverse->copy_from(inverse);

    gko::kernels::reference::isai::scatter_excess_solution(
        ref, a1.get_const_data(), e_rhs.get(), inverse.get(), 0, num_rows);
    gko::kernels::EXEC_NAMESPACE::isai::scatter_excess_solution(
        exec, da1.get_const_data(), de_rhs.get(), d_inverse.get(), 0, num_rows);

    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, IsaiScatterExcessSolutionSpdIsEquivalentToRef)
{
    initialize_data(matrix_type::spd, 100, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::array<index_type> da1(exec, a1);
    auto e_dim = a1.get_data()[num_rows];
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    std::fill_n(e_rhs->get_values(), e_dim, 123456);
    auto de_rhs = gko::clone(exec, e_rhs);
    // overwrite -1 values with inverse
    d_inverse->copy_from(inverse);

    gko::kernels::reference::isai::scatter_excess_solution(
        ref, a1.get_const_data(), e_rhs.get(), inverse.get(), 0, num_rows);
    gko::kernels::EXEC_NAMESPACE::isai::scatter_excess_solution(
        exec, da1.get_const_data(), de_rhs.get(), d_inverse.get(), 0, num_rows);

    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, IsaiScatterPartialExcessSolutionIsEquivalentToRef)
{
    initialize_data(matrix_type::spd, 100, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::array<index_type> da1(exec, a1);
    auto e_dim = a1.get_data()[10] - a1.get_data()[5];
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    std::fill_n(e_rhs->get_values(), e_dim, 123456);
    auto de_rhs = gko::clone(exec, e_rhs);
    // overwrite -1 values with inverse
    d_inverse->copy_from(inverse);

    gko::kernels::reference::isai::scatter_excess_solution(
        ref, a1.get_const_data(), e_rhs.get(), inverse.get(), 5u, 10u);
    gko::kernels::EXEC_NAMESPACE::isai::scatter_excess_solution(
        exec, da1.get_const_data(), de_rhs.get(), d_inverse.get(), 5u, 10u);

    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, IsaiGenerateGeneralWithSparsityPower8IsEquivalentToRef)
{
    using Isai =
        gko::preconditioner::Isai<gko::preconditioner::isai_type::general,
                                  value_type, index_type>;
    initialize_tridiag_data(matrix_type::general, 65);

    auto isai =
        Isai::build().with_sparsity_power(8).on(ref)->generate(mtx->clone());
    auto d_isai =
        Isai::build().with_sparsity_power(8).on(exec)->generate(d_mtx->clone());

    GKO_ASSERT_MTX_NEAR(isai->get_approximate_inverse(),
                        d_isai->get_approximate_inverse(),
                        5 * r<value_type>::value);
}


TEST_F(Isai, IsaiGenerateGeneralSparsityPowerNIsEquivalentToRef)
{
    using Isai =
        gko::preconditioner::Isai<gko::preconditioner::isai_type::general,
                                  value_type, index_type>;
    initialize_tridiag_data(matrix_type::general, 65);

    auto isai = Isai::build()
                    .with_sparsity_power(static_cast<int>(mtx->get_size()[0]))
                    .with_excess_solver_reduction(r<value_type>::value)
                    .on(ref)
                    ->generate(mtx->clone());
    auto d_isai =
        Isai::build()
            .with_sparsity_power(static_cast<int>(d_mtx->get_size()[0]))
            .with_excess_solver_reduction(r<value_type>::value)
            .on(exec)
            ->generate(d_mtx->clone());

    GKO_ASSERT_MTX_NEAR(isai->get_approximate_inverse(),
                        d_isai->get_approximate_inverse(),
                        10 * r<value_type>::value);
}
