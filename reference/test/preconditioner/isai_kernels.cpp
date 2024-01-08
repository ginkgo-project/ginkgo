// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/preconditioner/isai.hpp>


#include <algorithm>
#include <fstream>
#include <memory>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/gmres.hpp>


#include "core/base/utils.hpp"
#include "core/preconditioner/isai_kernels.hpp"
#include "core/test/utils.hpp"
#include "matrices/config.hpp"


namespace {


template <typename ValueIndexType>
class Isai : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using excess_solver_type = gko::solver::Gmres<value_type>;
    using bj = gko::preconditioner::Jacobi<value_type, index_type>;
    using LowerIsai = gko::preconditioner::LowerIsai<value_type, index_type>;
    using UpperIsai = gko::preconditioner::UpperIsai<value_type, index_type>;
    using GeneralIsai =
        gko::preconditioner::GeneralIsai<value_type, index_type>;
    using SpdIsai = gko::preconditioner::SpdIsai<value_type, index_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    Isai()
        : exec{gko::ReferenceExecutor::create()},
          excess_solver_factory(
              excess_solver_type::build()
                  .with_preconditioner(bj::build().with_max_block_size(16u))
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(1000u),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_baseline(gko::stop::mode::rhs_norm)
                          .with_reduction_factor(
                              gko::remove_complex<value_type>{1e-6}))
                  .on(exec)),
          a_dense{gko::initialize<Dense>({{2, 1, 2}, {1, -2, 3}, {-1, 1, 1}},
                                         exec)},
          a_dense_inv{gko::initialize<Dense>({{0.3125, -0.0625, -0.4375},
                                              {0.25, -0.25, 0.25},
                                              {0.0625, 0.1875, 0.3125}},
                                             exec)},
          l_dense{gko::initialize<Dense>(
              {{2., 0., 0.}, {1., -2., 0.}, {-1., 1., -1.}}, exec)},
          l_dense_inv{gko::initialize<Dense>(
              {{.5, 0., 0.}, {.25, -.5, 0.}, {-.25, -.5, -1.}}, exec)},
          u_dense{gko::initialize<Dense>(
              {{4., 1., -1.}, {0., -2., 4.}, {0., 0., 8.}}, exec)},
          u_dense_inv{gko::initialize<Dense>(
              {{.25, .125, -0.03125}, {0., -.5, .25}, {0., 0., .125}}, exec)},
          spd_dense{gko::initialize<Dense>(
              {{.0625, -.0625, .25}, {-.0625, .3125, -.5}, {.25, -.5, 2.25}},
              exec)},
          spd_dense_inv{gko::initialize<Dense>(
              {{4., 0., 0.}, {2., 2., 0.}, {-3., 1., 1.}}, exec)},
          a_csr{Csr::create(exec)},
          a_csr_inv{Csr::create(exec)},
          l_csr{Csr::create(exec)},
          l_csr_inv{Csr::create(exec)},
          u_csr{Csr::create(exec)},
          u_csr_inv{Csr::create(exec)},
          spd_csr{Csr::create(exec)},
          spd_csr_inv{Csr::create(exec)},
          l_sparse{Csr::create(exec, gko::dim<2>(4, 4),
                               I<value_type>{-1., 2., 4., 5., -4., 8., -8.},
                               I<index_type>{0, 0, 1, 1, 2, 2, 3},
                               I<index_type>{0, 1, 3, 5, 7})},
          l_s_unsorted{Csr::create(exec, gko::dim<2>(4, 4),
                                   I<value_type>{-1., 4., 2., 5., -4., -8., 8.},
                                   I<index_type>{0, 1, 0, 1, 2, 3, 2},
                                   I<index_type>{0, 1, 3, 5, 7})},
          l_sparse_inv{
              Csr::create(exec, gko::dim<2>(4, 4),
                          I<value_type>{-1., .5, .25, .3125, -.25, -.25, -.125},
                          I<index_type>{0, 0, 1, 1, 2, 2, 3},
                          I<index_type>{0, 1, 3, 5, 7})},
          l_sparse_inv2{Csr::create(exec, gko::dim<2>(4, 4),
                                    I<value_type>{-1., .5, .25, .625, .3125,
                                                  -.25, .3125, -.25, -.125},
                                    I<index_type>{0, 0, 1, 0, 1, 2, 1, 2, 3},
                                    I<index_type>{0, 1, 3, 6, 9})},
          l_sparse_inv3{
              Csr::create(exec, gko::dim<2>(4, 4),
                          I<value_type>{-1., .5, .25, .625, .3125, -.25, .625,
                                        .3125, -.25, -.125},
                          I<index_type>{0, 0, 1, 0, 1, 2, 0, 1, 2, 3},
                          I<index_type>{0, 1, 3, 6, 10})},
          l_sparse2{Csr::create(exec, gko::dim<2>(4, 4),
                                I<value_type>{-2, 1, 4, 1, -2, 1, -1, 1, 2},
                                I<index_type>{0, 0, 1, 1, 2, 0, 1, 2, 3},
                                I<index_type>{0, 1, 3, 5, 9})},
          l_sparse2_inv{Csr::create(exec, gko::dim<2>(4, 4),
                                    I<value_type>{-.5, .125, .25, .125, -.5,
                                                  .28125, .0625, 0.25, 0.5},
                                    I<index_type>{0, 0, 1, 1, 2, 0, 1, 2, 3},
                                    I<index_type>{0, 1, 3, 5, 9})},
          u_sparse{
              Csr::create(exec, gko::dim<2>(4, 4),
                          I<value_type>{-2., 1., -1., 1., 4., 1., -2., 1., 2.},
                          I<index_type>{0, 1, 2, 3, 1, 2, 2, 3, 3},
                          I<index_type>{0, 4, 6, 8, 9})},
          u_s_unsorted{
              Csr::create(exec, gko::dim<2>(4, 4),
                          I<value_type>{-2., -1., 1., 1., 1., 4., -2., 1., 2.},
                          I<index_type>{0, 2, 1, 3, 2, 1, 2, 3, 3},
                          I<index_type>{0, 4, 6, 8, 9})},
          u_sparse_inv{Csr::create(
              exec, gko::dim<2>(4, 4),
              I<value_type>{-.5, .125, .3125, .09375, .25, .125, -.5, .25, .5},
              I<index_type>{0, 1, 2, 3, 1, 2, 2, 3, 3},
              I<index_type>{0, 4, 6, 8, 9})},
          u_sparse_inv2{Csr::create(exec, gko::dim<2>(4, 4),
                                    I<value_type>{-.5, .125, .3125, .09375, .25,
                                                  .125, -.0625, -.5, .25, .5},
                                    I<index_type>{0, 1, 2, 3, 1, 2, 3, 2, 3, 3},
                                    I<index_type>{0, 4, 7, 9, 10})},
          a_sparse{
              Csr::create(exec, gko::dim<2>{4, 4},
                          I<value_type>{1., 4., 1., 4., 2., 1., 2., 1., 1.},
                          I<index_type>{0, 3, 0, 1, 3, 1, 2, 2, 3},
                          I<index_type>{0, 2, 5, 7, 9})},
          a_s_unsorted{
              Csr::create(exec, gko::dim<2>{4, 4},
                          I<value_type>{4., 1., 1., 2., 4., 2., 1., 1., 1.},
                          I<index_type>{3, 0, 0, 3, 1, 2, 1, 2, 3},
                          I<index_type>{0, 2, 5, 7, 9})},
          a_sparse_inv{Csr::create(
              exec, gko::dim<2>{4, 4},
              I<value_type>{1., -4, -.25, .25, .5, -.125, .5, -.5, 1},
              I<index_type>{0, 3, 0, 1, 3, 1, 2, 2, 3},
              I<index_type>{0, 2, 5, 7, 9})},
          spd_sparse{Csr::create(exec, gko::dim<2>{4, 4},
                                 I<value_type>{.25, -.25, -.25, .5, 4., 4.},
                                 I<index_type>{0, 1, 0, 1, 2, 3},
                                 I<index_type>{0, 2, 4, 5, 6})},
          spd_s_unsorted{Csr::create(exec, gko::dim<2>{4, 4},
                                     I<value_type>{-.25, .25, .5, -.25, 4., 4.},
                                     I<index_type>{1, 0, 1, 0, 2, 3},
                                     I<index_type>{0, 2, 4, 5, 6})},
          spd_sparse_inv{Csr::create(
              exec, gko::dim<2>{4, 4}, I<value_type>{2., 2., 2., .5, .5},
              I<index_type>{0, 0, 1, 2, 3}, I<index_type>{0, 1, 3, 4, 5})}
    {
        lower_isai_factory = LowerIsai::build().on(exec);
        upper_isai_factory = UpperIsai::build().on(exec);
        general_isai_factory = GeneralIsai::build().on(exec);
        spd_isai_factory = SpdIsai::build().on(exec);
        a_dense->convert_to(a_csr);
        a_dense_inv->convert_to(a_csr_inv);
        l_dense->convert_to(l_csr);
        l_dense_inv->convert_to(l_csr_inv);
        u_dense->convert_to(u_csr);
        u_dense_inv->convert_to(u_csr_inv);
        spd_dense->convert_to(spd_csr);
        spd_dense_inv->convert_to(spd_csr_inv);
        l_csr_longrow = read<Csr>("isai_l.mtx");
        l_csr_longrow_e = read<Csr>("isai_l_excess.mtx");
        l_csr_longrow_e_rhs = read<Dense>("isai_l_excess_rhs.mtx");
        l_csr_longrow_inv_partial = read<Csr>("isai_l_inv_partial.mtx");
        l_csr_longrow_inv = read<Csr>("isai_l_inv.mtx");
        u_csr_longrow = read<Csr>("isai_u.mtx");
        u_csr_longrow_e = read<Csr>("isai_u_excess.mtx");
        u_csr_longrow_e_rhs = read<Dense>("isai_u_excess_rhs.mtx");
        u_csr_longrow_inv_partial = read<Csr>("isai_u_inv_partial.mtx");
        u_csr_longrow_inv = read<Csr>("isai_u_inv.mtx");
        a_csr_longrow = read<Csr>("isai_a.mtx");
        a_csr_longrow_e = read<Csr>("isai_a_excess.mtx");
        a_csr_longrow_e_rhs = read<Dense>("isai_a_excess_rhs.mtx");
        a_csr_longrow_inv_partial = read<Csr>("isai_a_inv_partial.mtx");
        a_csr_longrow_inv = read<Csr>("isai_a_inv.mtx");
        spd_csr_longrow = read<Csr>("isai_spd.mtx");
        spd_csr_longrow_e = read<Csr>("isai_spd_excess.mtx");
        spd_csr_longrow_e_rhs = read<Dense>("isai_spd_excess_rhs.mtx");
        spd_csr_longrow_inv_partial = read<Csr>("isai_spd_inv_partial.mtx");
        spd_csr_longrow_inv = read<Csr>("isai_spd_inv.mtx");
    }

    template <typename ReadMtx>
    std::unique_ptr<ReadMtx> read(const char* name)
    {
        std::ifstream mtxstream{std::string{gko::matrices::location_isai_mtxs} +
                                name};
        auto result = gko::read<ReadMtx>(mtxstream, exec);
        // to avoid removing 0s, the matrices store 12345 instead
        for (gko::size_type i = 0; i < result->get_num_stored_elements(); ++i) {
            auto& val = result->get_values()[i];
            if (val == static_cast<value_type>(12345.0)) {
                val = 0;
            }
        }
        return std::move(result);
    }

    std::unique_ptr<Csr> clone_allocations(gko::ptr_param<const Csr> csr_mtx)
    {
        const auto num_elems = csr_mtx->get_num_stored_elements();
        auto sparsity = csr_mtx->clone();

        // values are now filled with invalid data to catch potential errors
        std::fill_n(sparsity->get_values(), num_elems, -gko::one<value_type>());
        return sparsity;
    }

    std::unique_ptr<Csr> transpose(gko::ptr_param<const Csr> mtx)
    {
        return gko::as<Csr>(mtx->transpose());
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<typename excess_solver_type::Factory> excess_solver_factory;
    std::unique_ptr<typename LowerIsai::Factory> lower_isai_factory;
    std::unique_ptr<typename UpperIsai::Factory> upper_isai_factory;
    std::unique_ptr<typename GeneralIsai::Factory> general_isai_factory;
    std::unique_ptr<typename SpdIsai::Factory> spd_isai_factory;
    std::shared_ptr<Dense> a_dense;
    std::shared_ptr<Dense> a_dense_inv;
    std::shared_ptr<Dense> l_dense;
    std::shared_ptr<Dense> l_dense_inv;
    std::shared_ptr<Dense> u_dense;
    std::shared_ptr<Dense> u_dense_inv;
    std::shared_ptr<Dense> spd_dense;
    std::shared_ptr<Dense> spd_dense_inv;
    std::shared_ptr<Csr> a_csr;
    std::shared_ptr<Csr> a_csr_inv;
    std::shared_ptr<Csr> l_csr;
    std::shared_ptr<Csr> l_csr_inv;
    std::shared_ptr<Csr> l_csr_longrow;
    std::shared_ptr<Csr> l_csr_longrow_e;
    std::shared_ptr<Dense> l_csr_longrow_e_rhs;
    std::shared_ptr<Csr> l_csr_longrow_inv_partial;
    std::shared_ptr<Csr> l_csr_longrow_inv;
    std::shared_ptr<Csr> u_csr;
    std::shared_ptr<Csr> u_csr_inv;
    std::shared_ptr<Csr> u_csr_longrow;
    std::shared_ptr<Csr> u_csr_longrow_e;
    std::shared_ptr<Dense> u_csr_longrow_e_rhs;
    std::shared_ptr<Csr> u_csr_longrow_inv_partial;
    std::shared_ptr<Csr> u_csr_longrow_inv;
    std::shared_ptr<Csr> l_sparse;
    std::shared_ptr<Csr> l_s_unsorted;
    std::shared_ptr<Csr> l_sparse_inv;
    std::shared_ptr<Csr> l_sparse_inv2;
    std::shared_ptr<Csr> l_sparse_inv3;
    std::shared_ptr<Csr> l_sparse2;
    std::shared_ptr<Csr> l_sparse2_inv;
    std::shared_ptr<Csr> u_sparse;
    std::shared_ptr<Csr> u_s_unsorted;
    std::shared_ptr<Csr> u_sparse_inv;
    std::shared_ptr<Csr> u_sparse_inv2;
    std::shared_ptr<Csr> a_sparse;
    std::shared_ptr<Csr> a_s_unsorted;
    std::shared_ptr<Csr> a_sparse_inv;
    std::shared_ptr<Csr> a_csr_longrow;
    std::shared_ptr<Csr> a_csr_longrow_e;
    std::shared_ptr<Dense> a_csr_longrow_e_rhs;
    std::shared_ptr<Csr> a_csr_longrow_inv_partial;
    std::shared_ptr<Csr> a_csr_longrow_inv;
    std::shared_ptr<Csr> spd_csr;
    std::shared_ptr<Csr> spd_csr_inv;
    std::shared_ptr<Csr> spd_csr_longrow;
    std::shared_ptr<Csr> spd_csr_longrow_e;
    std::shared_ptr<Dense> spd_csr_longrow_e_rhs;
    std::shared_ptr<Csr> spd_csr_longrow_inv_partial;
    std::shared_ptr<Csr> spd_csr_longrow_inv;
    std::shared_ptr<Csr> spd_sparse;
    std::shared_ptr<Csr> spd_s_unsorted;
    std::shared_ptr<Csr> spd_sparse_inv;
};

TYPED_TEST_SUITE(Isai, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Isai, KernelGenerateA)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto result = this->clone_allocations(this->a_csr);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_general_inverse(
        this->exec, this->a_csr.get(), result.get(), a1.get_data(),
        a2.get_data(), false);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->a_csr_inv);
    GKO_ASSERT_MTX_NEAR(result, this->a_csr_inv, r<value_type>::value);
    // no row above the size limit -> zero array
    GKO_ASSERT_ARRAY_EQ(a1, zeros);
    GKO_ASSERT_ARRAY_EQ(a2, zeros);
}


TYPED_TEST(Isai, KernelGenerateA2)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto a_transpose = this->transpose(this->a_csr);
    auto result = this->clone_allocations(a_transpose);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_general_inverse(
        this->exec, a_transpose.get(), result.get(), a1.get_data(),
        a2.get_data(), false);

    const auto expected = this->transpose(this->a_csr_inv);
    GKO_ASSERT_MTX_EQ_SPARSITY(result, expected);
    GKO_ASSERT_MTX_NEAR(result, expected, r<value_type>::value);
    // no row above the size limit -> zero array
    GKO_ASSERT_ARRAY_EQ(a1, zeros);
    GKO_ASSERT_ARRAY_EQ(a2, zeros);
}


TYPED_TEST(Isai, KernelGenerateAsparse)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto result = this->clone_allocations(this->a_sparse);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_general_inverse(
        this->exec, this->a_sparse.get(), result.get(), a1.get_data(),
        a2.get_data(), false);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->a_sparse_inv);
    GKO_ASSERT_MTX_NEAR(result, this->a_sparse_inv, r<value_type>::value);
    // no row above the size limit -> zero array
    GKO_ASSERT_ARRAY_EQ(a1, zeros);
    GKO_ASSERT_ARRAY_EQ(a2, zeros);
}


TYPED_TEST(Isai, KernelGenerateALongrow)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto result = this->clone_allocations(this->a_csr_longrow);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);
    // only the 32nd row has some excess storage
    auto a1_expect = zeros;
    std::fill_n(a1_expect.get_data() + 15, 21, 86);
    std::fill_n(a1_expect.get_data() + 36, 65, 122);
    auto a2_expect = zeros;
    std::fill_n(a2_expect.get_data() + 15, 21, 355);
    std::fill_n(a2_expect.get_data() + 36, 65, 509);

    gko::kernels::reference::isai::generate_general_inverse(
        this->exec, this->a_csr_longrow.get(), result.get(), a1.get_data(),
        a2.get_data(), false);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->a_csr_longrow_inv_partial);
    GKO_ASSERT_MTX_NEAR(result, this->a_csr_longrow_inv_partial,
                        r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, a1_expect);
    GKO_ASSERT_ARRAY_EQ(a2, a2_expect);
}


TYPED_TEST(Isai, KernelGenerateExcessALongrow)
{
    using Csr = typename TestFixture::Csr;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto num_rows = this->a_csr_longrow->get_size()[0];
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);
    // only the 32nd row has some excess storage
    auto a1 = zeros;
    std::fill_n(a1.get_data() + 15, 21, 86);
    std::fill_n(a1.get_data() + 36, 65, 122);
    auto a2 = zeros;
    std::fill_n(a2.get_data() + 15, 21, 355);
    std::fill_n(a2.get_data() + 36, 65, 509);
    auto result = Csr::create(this->exec, gko::dim<2>(122, 122), 509);
    auto result_rhs = Dense::create(this->exec, gko::dim<2>(122, 1));

    gko::kernels::reference::isai::generate_excess_system(
        this->exec, this->a_csr_longrow.get(), this->a_csr_longrow.get(),
        a1.get_const_data(), a2.get_const_data(), result.get(),
        result_rhs.get(), 0, num_rows);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->a_csr_longrow_e);
    GKO_ASSERT_MTX_NEAR(result, this->a_csr_longrow_e, 0);
    GKO_ASSERT_MTX_NEAR(result_rhs, this->a_csr_longrow_e_rhs, 0);
}


TYPED_TEST(Isai, KernelGenerateL1)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto result = this->clone_allocations(this->l_csr);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, this->l_csr.get(), result.get(), a1.get_data(),
        a2.get_data(), true);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->l_csr_inv);
    GKO_ASSERT_MTX_NEAR(result, this->l_csr_inv, r<value_type>::value);
    // no row above the size limit -> zero array
    GKO_ASSERT_ARRAY_EQ(a1, zeros);
    GKO_ASSERT_ARRAY_EQ(a2, zeros);
}


TYPED_TEST(Isai, KernelGenerateL2)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const auto l_mtx = this->transpose(this->u_csr);
    auto result = this->clone_allocations(l_mtx);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, l_mtx.get(), result.get(), a1.get_data(), a2.get_data(),
        true);

    const auto expected = this->transpose(this->u_csr_inv);
    GKO_ASSERT_MTX_EQ_SPARSITY(result, expected);
    GKO_ASSERT_MTX_NEAR(result, expected, r<value_type>::value);
    // no row above the size limit -> zero array
    GKO_ASSERT_ARRAY_EQ(a1, zeros);
    GKO_ASSERT_ARRAY_EQ(a2, zeros);
}


TYPED_TEST(Isai, KernelGenerateLsparse1)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto result = this->clone_allocations(this->l_sparse);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, this->l_sparse.get(), result.get(), a1.get_data(),
        a2.get_data(), true);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->l_sparse_inv);
    GKO_ASSERT_MTX_NEAR(result, this->l_sparse_inv, r<value_type>::value);
    // no row above the size limit -> zero array
    GKO_ASSERT_ARRAY_EQ(a1, zeros);
    GKO_ASSERT_ARRAY_EQ(a2, zeros);
}


TYPED_TEST(Isai, KernelGenerateLsparse2)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto result = this->clone_allocations(this->l_sparse2);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, this->l_sparse2.get(), result.get(), a1.get_data(),
        a2.get_data(), true);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->l_sparse2_inv);
    GKO_ASSERT_MTX_NEAR(result, this->l_sparse2_inv, r<value_type>::value);
    // no row above the size limit -> zero array
    GKO_ASSERT_ARRAY_EQ(a1, zeros);
    GKO_ASSERT_ARRAY_EQ(a2, zeros);
}


TYPED_TEST(Isai, KernelGenerateLsparse3)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const auto l_mtx = this->transpose(this->u_sparse);
    auto result = this->clone_allocations(l_mtx);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, l_mtx.get(), result.get(), a1.get_data(), a2.get_data(),
        true);

    // Results in a slightly different version than u_sparse_inv->transpose()
    // because a different row-sparsity pattern is used in u_sparse vs. l_mtx
    // (only one value changes compared to u_sparse_inv->transpose())
    const auto expected = Csr::create(
        this->exec, gko::dim<2>(4, 4),
        I<value_type>{-.5, .125, .25, .3125, .125, -.5, .125, .25, .5},
        I<index_type>{0, 0, 1, 0, 1, 2, 0, 2, 3}, I<index_type>{0, 1, 3, 6, 9});
    GKO_ASSERT_MTX_EQ_SPARSITY(result, expected);
    GKO_ASSERT_MTX_NEAR(result, expected, r<value_type>::value);
    // no row above the size limit -> zero array
    GKO_ASSERT_ARRAY_EQ(a1, zeros);
    GKO_ASSERT_ARRAY_EQ(a2, zeros);
}


TYPED_TEST(Isai, KernelGenerateLLongrow)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto result = this->clone_allocations(this->l_csr_longrow);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);
    // only the 32nd row has some excess storage
    auto a1_expect = zeros;
    a1_expect.get_data()[33] = 33;
    a1_expect.get_data()[34] = 33;
    a1_expect.get_data()[35] = 66;
    auto a2_expect = zeros;
    a2_expect.get_data()[33] = 124;
    a2_expect.get_data()[34] = 124;
    a2_expect.get_data()[35] = 248;

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, this->l_csr_longrow.get(), result.get(), a1.get_data(),
        a2.get_data(), true);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->l_csr_longrow_inv_partial);
    GKO_ASSERT_MTX_NEAR(result, this->l_csr_longrow_inv_partial,
                        r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, a1_expect);
    GKO_ASSERT_ARRAY_EQ(a2, a2_expect);
}


TYPED_TEST(Isai, KernelGenerateExcessLLongrow)
{
    using Csr = typename TestFixture::Csr;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto num_rows = this->l_csr_longrow->get_size()[0];
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);
    // only the 32nd row has some excess storage
    auto a1 = zeros;
    a1.get_data()[33] = 33;
    a1.get_data()[34] = 33;
    a1.get_data()[35] = 66;
    auto a2 = zeros;
    a2.get_data()[33] = 124;
    a2.get_data()[34] = 124;
    a2.get_data()[35] = 248;
    auto result = Csr::create(this->exec, gko::dim<2>(66, 66), 248);
    auto result_rhs = Dense::create(this->exec, gko::dim<2>(66, 1));

    gko::kernels::reference::isai::generate_excess_system(
        this->exec, this->l_csr_longrow.get(), this->l_csr_longrow.get(),
        a1.get_const_data(), a2.get_const_data(), result.get(),
        result_rhs.get(), 0, num_rows);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->l_csr_longrow_e);
    GKO_ASSERT_MTX_NEAR(result, this->l_csr_longrow_e, 0);
    GKO_ASSERT_MTX_NEAR(result_rhs, this->l_csr_longrow_e_rhs, 0);
}


TYPED_TEST(Isai, KernelGenerateU1)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const auto u_mtx = this->transpose(this->l_csr);
    auto result = this->clone_allocations(u_mtx);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, u_mtx.get(), result.get(), a1.get_data(), a2.get_data(),
        false);

    auto expected = this->transpose(this->l_csr_inv);
    GKO_ASSERT_MTX_EQ_SPARSITY(result, expected);
    GKO_ASSERT_MTX_NEAR(result, expected, r<value_type>::value);
    // no row above the size limit -> zero array
    GKO_ASSERT_ARRAY_EQ(a1, zeros);
    GKO_ASSERT_ARRAY_EQ(a2, zeros);
}


TYPED_TEST(Isai, KernelGenerateU2)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto result = this->clone_allocations(this->u_csr);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, this->u_csr.get(), result.get(), a1.get_data(),
        a2.get_data(), false);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->u_csr_inv);
    GKO_ASSERT_MTX_NEAR(result, this->u_csr_inv, r<value_type>::value);
    // no row above the size limit -> zero array
    GKO_ASSERT_ARRAY_EQ(a1, zeros);
    GKO_ASSERT_ARRAY_EQ(a2, zeros);
}


TYPED_TEST(Isai, KernelGenerateUsparse1)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const auto u_mtx = this->transpose(this->l_sparse);
    auto result = this->clone_allocations(u_mtx);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, u_mtx.get(), result.get(), a1.get_data(), a2.get_data(),
        false);

    const auto expected = this->transpose(this->l_sparse_inv);
    GKO_ASSERT_MTX_EQ_SPARSITY(result, expected);
    GKO_ASSERT_MTX_NEAR(result, expected, r<value_type>::value);
    // no row above the size limit -> zero array
    GKO_ASSERT_ARRAY_EQ(a1, zeros);
    GKO_ASSERT_ARRAY_EQ(a2, zeros);
}


TYPED_TEST(Isai, KernelGenerateUsparse2)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const auto u_mtx = this->transpose(this->l_sparse2);
    auto result = this->clone_allocations(u_mtx);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, u_mtx.get(), result.get(), a1.get_data(), a2.get_data(),
        false);

    // Results in a slightly different version than l_sparse2_inv->transpose()
    // because a different row-sparsity pattern is used in l_sparse2 vs. u_mtx
    // (only one value changes compared to l_sparse2_inv->transpose())
    const auto expected = Csr::create(
        this->exec, gko::dim<2>(4, 4),
        I<value_type>{-.5, .125, .3125, .25, .125, .0625, -.5, .25, .5},
        I<index_type>{0, 1, 3, 1, 2, 3, 2, 3, 3}, I<index_type>{0, 3, 6, 8, 9});
    GKO_ASSERT_MTX_EQ_SPARSITY(result, expected);
    GKO_ASSERT_MTX_NEAR(result, expected, r<value_type>::value);
    // no row above the size limit -> zero array
    GKO_ASSERT_ARRAY_EQ(a1, zeros);
    GKO_ASSERT_ARRAY_EQ(a2, zeros);
}


TYPED_TEST(Isai, KernelGenerateUsparse3)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto result = this->clone_allocations(this->u_sparse);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, this->u_sparse.get(), result.get(), a1.get_data(),
        a2.get_data(), false);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->u_sparse_inv);
    GKO_ASSERT_MTX_NEAR(result, this->u_sparse_inv, r<value_type>::value);
    // no row above the size limit -> zero array
    GKO_ASSERT_ARRAY_EQ(a1, zeros);
    GKO_ASSERT_ARRAY_EQ(a2, zeros);
}


TYPED_TEST(Isai, KernelGenerateULongrow)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto result = this->clone_allocations(this->u_csr_longrow);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);
    // only the 32nd row has some excess storage
    auto a1_expect = zeros;
    std::fill_n(a1_expect.get_data() + 3, 33, 33);
    auto a2_expect = zeros;
    std::fill_n(a2_expect.get_data() + 3, 33, 153);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, this->u_csr_longrow.get(), result.get(), a1.get_data(),
        a2.get_data(), false);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->u_csr_longrow_inv_partial);
    GKO_ASSERT_MTX_NEAR(result, this->u_csr_longrow_inv_partial,
                        r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, a1_expect);
    GKO_ASSERT_ARRAY_EQ(a2, a2_expect);
}


TYPED_TEST(Isai, KernelGenerateExcessULongrow)
{
    using Csr = typename TestFixture::Csr;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto num_rows = this->u_csr_longrow->get_size()[0];
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);
    // only the 32nd row has some excess storage
    auto a1 = zeros;
    std::fill_n(a1.get_data() + 3, 33, 33);
    auto a2 = zeros;
    std::fill_n(a2.get_data() + 3, 33, 153);
    auto result = Csr::create(this->exec, gko::dim<2>(33, 33), 153);
    auto result_rhs = Dense::create(this->exec, gko::dim<2>(33, 1));

    gko::kernels::reference::isai::generate_excess_system(
        this->exec, this->u_csr_longrow.get(), this->u_csr_longrow.get(),
        a1.get_const_data(), a2.get_const_data(), result.get(),
        result_rhs.get(), 0, num_rows);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->u_csr_longrow_e);
    GKO_ASSERT_MTX_NEAR(result, this->u_csr_longrow_e, 0);
    GKO_ASSERT_MTX_NEAR(result_rhs, this->u_csr_longrow_e_rhs, 0);
}


TYPED_TEST(Isai, KernelGenerateSpd)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto result = this->clone_allocations(this->spd_csr_inv);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_general_inverse(
        this->exec, this->spd_csr.get(), result.get(), a1.get_data(),
        a2.get_data(), true);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->spd_csr_inv);
    GKO_ASSERT_MTX_NEAR(result, this->spd_csr_inv, r<value_type>::value);
    // no row above the size limit -> zero array
    GKO_ASSERT_ARRAY_EQ(a1, zeros);
    GKO_ASSERT_ARRAY_EQ(a2, zeros);
}


TYPED_TEST(Isai, KernelGenerateSpdsparse)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto result = this->clone_allocations(this->spd_sparse_inv);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_general_inverse(
        this->exec, this->spd_sparse.get(), result.get(), a1.get_data(),
        a2.get_data(), true);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->spd_sparse_inv);
    GKO_ASSERT_MTX_NEAR(result, this->spd_sparse_inv, r<value_type>::value);
    // no row above the size limit -> zero array
    GKO_ASSERT_ARRAY_EQ(a1, zeros);
    GKO_ASSERT_ARRAY_EQ(a2, zeros);
}


TYPED_TEST(Isai, KernelGenerateSpdLongrow)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto result = this->clone_allocations(this->spd_csr_longrow_inv);
    auto num_rows = result->get_size()[0];
    gko::array<index_type> a1(this->exec, num_rows + 1);
    gko::array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);
    // only the 36th row has some excess storage
    auto a1_expect = zeros;
    std::fill_n(a1_expect.get_data() + 36, 65, 36);
    auto a2_expect = zeros;
    std::fill_n(a2_expect.get_data() + 36, 65, 338);

    gko::kernels::reference::isai::generate_general_inverse(
        this->exec, this->spd_csr_longrow.get(), result.get(), a1.get_data(),
        a2.get_data(), true);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->spd_csr_longrow_inv_partial);
    GKO_ASSERT_MTX_NEAR(result, this->spd_csr_longrow_inv_partial,
                        r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, a1_expect);
    GKO_ASSERT_ARRAY_EQ(a2, a2_expect);
}


TYPED_TEST(Isai, KernelGenerateExcessSpdLongrow)
{
    using Csr = typename TestFixture::Csr;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto num_rows = this->spd_csr_longrow->get_size()[0];
    gko::array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);
    // only the 36th row has some excess storage
    auto a1 = zeros;
    std::fill_n(a1.get_data() + 36, 65, 36);
    auto a2 = zeros;
    std::fill_n(a2.get_data() + 36, 65, 338);
    auto result = Csr::create(this->exec, gko::dim<2>(36, 36), 338);
    auto result_rhs = Dense::create(this->exec, gko::dim<2>(36, 1));

    gko::kernels::reference::isai::generate_excess_system(
        this->exec, this->spd_csr_longrow.get(),
        this->spd_csr_longrow_inv_partial.get(), a1.get_const_data(),
        a2.get_const_data(), result.get(), result_rhs.get(), 0, num_rows);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->spd_csr_longrow_e);
    GKO_ASSERT_MTX_NEAR(result, this->spd_csr_longrow_e, 0);
    GKO_ASSERT_MTX_NEAR(result_rhs, this->spd_csr_longrow_e_rhs, 0);
}


TYPED_TEST(Isai, KernelScatterExcessSolution)
{
    using Csr = typename TestFixture::Csr;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    gko::array<index_type> ptrs{this->exec, I<index_type>{0, 0, 2, 2, 5, 7, 7}};
    auto mtx = Csr::create(this->exec, gko::dim<2>{6, 6},
                           I<value_type>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                           I<index_type>{0, 0, 1, 0, 0, 1, 2, 0, 1, 0},
                           I<index_type>{0, 1, 3, 4, 7, 9, 10});
    auto expect =
        Csr::create(this->exec, gko::dim<2>{6, 6},
                    I<value_type>{1, 11, 12, 4, 13, 14, 15, 16, 17, 10},
                    I<index_type>{0, 0, 1, 0, 0, 1, 2, 0, 1, 0},
                    I<index_type>{0, 1, 3, 4, 7, 9, 10});
    auto sol = Dense::create(this->exec, gko::dim<2>(7, 1),
                             I<value_type>{11, 12, 13, 14, 15, 16, 17}, 1);

    gko::kernels::reference::isai::scatter_excess_solution(
        this->exec, ptrs.get_const_data(), sol.get(), mtx.get(), 0, 6);

    GKO_ASSERT_MTX_NEAR(mtx, expect, 0);
}


TYPED_TEST(Isai, ReturnsCorrectInverseA)
{
    using value_type = typename TestFixture::value_type;
    const auto isai = this->general_isai_factory->generate(this->a_sparse);

    auto l_inv = isai->get_approximate_inverse();

    GKO_ASSERT_MTX_EQ_SPARSITY(l_inv, this->a_sparse_inv);
    GKO_ASSERT_MTX_NEAR(l_inv, this->a_sparse_inv, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseALongrow)
{
    using value_type = typename TestFixture::value_type;
    const auto isai = this->general_isai_factory->generate(this->a_csr_longrow);

    auto a_inv = isai->get_approximate_inverse();

    GKO_ASSERT_MTX_EQ_SPARSITY(a_inv, this->a_csr_longrow_inv);
    // need to reduce precision due to general ISAI using GMRES instead of
    // direct solve
    GKO_ASSERT_MTX_NEAR(a_inv, this->a_csr_longrow_inv,
                        20 * r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseALongrowWithExcessSolver)
{
    using value_type = typename TestFixture::value_type;
    using GeneralIsai = typename TestFixture::GeneralIsai;
    auto general_isai_factory =
        GeneralIsai::build()
            .with_excess_solver_factory(this->excess_solver_factory)
            .on(this->exec);
    const auto isai = general_isai_factory->generate(this->a_csr_longrow);

    auto a_inv = isai->get_approximate_inverse();

    GKO_ASSERT_MTX_EQ_SPARSITY(a_inv, this->a_csr_longrow_inv);
    // need to drastically reduce precision due to using different excess solver
    // factory.
    GKO_ASSERT_MTX_NEAR(a_inv, this->a_csr_longrow_inv,
                        2e4 * r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseL)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->lower_isai_factory->generate(this->l_sparse);

    auto l_inv = isai->get_approximate_inverse();

    GKO_ASSERT_MTX_EQ_SPARSITY(l_inv, this->l_sparse_inv);
    GKO_ASSERT_MTX_NEAR(l_inv, this->l_sparse_inv, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseLLongrow)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->lower_isai_factory->generate(this->l_csr_longrow);

    auto l_inv = isai->get_approximate_inverse();

    GKO_ASSERT_MTX_EQ_SPARSITY(l_inv, this->l_csr_longrow_inv);
    GKO_ASSERT_MTX_NEAR(l_inv, this->l_csr_longrow_inv, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseLLongrowWithExcessSolver)
{
    using Csr = typename TestFixture::Csr;
    using LowerIsai = typename TestFixture::LowerIsai;
    using value_type = typename TestFixture::value_type;
    auto lower_isai_factory =
        LowerIsai::build()
            .with_excess_solver_factory(this->excess_solver_factory)
            .on(this->exec);
    const auto isai = lower_isai_factory->generate(this->l_csr_longrow);

    auto l_inv = isai->get_approximate_inverse();

    GKO_ASSERT_MTX_EQ_SPARSITY(l_inv, this->l_csr_longrow_inv);
    // need to drastically reduce precision due to using different excess solver
    // factory.
    GKO_ASSERT_MTX_NEAR(l_inv, this->l_csr_longrow_inv,
                        1e3 * r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseU)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->upper_isai_factory->generate(this->u_sparse);

    auto u_inv = isai->get_approximate_inverse();

    GKO_ASSERT_MTX_EQ_SPARSITY(u_inv, this->u_sparse_inv);
    GKO_ASSERT_MTX_NEAR(u_inv, this->u_sparse_inv, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseULongrow)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->upper_isai_factory->generate(this->u_csr_longrow);

    auto u_inv = isai->get_approximate_inverse();

    GKO_ASSERT_MTX_EQ_SPARSITY(u_inv, this->u_csr_longrow_inv);
    GKO_ASSERT_MTX_NEAR(u_inv, this->u_csr_longrow_inv, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseULongrowWithExcessSolver)
{
    using Csr = typename TestFixture::Csr;
    using UpperIsai = typename TestFixture::UpperIsai;
    using value_type = typename TestFixture::value_type;
    auto upper_isai_factory =
        UpperIsai::build()
            .with_excess_solver_factory(this->excess_solver_factory)
            .on(this->exec);
    const auto isai = upper_isai_factory->generate(this->u_csr_longrow);

    auto u_inv = isai->get_approximate_inverse();

    GKO_ASSERT_MTX_EQ_SPARSITY(u_inv, this->u_csr_longrow_inv);
    // need to drastically reduce precision due to using different excess solver
    // factory.
    GKO_ASSERT_MTX_NEAR(u_inv, this->u_csr_longrow_inv,
                        1e3 * r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseLWithL2)
{
    using value_type = typename TestFixture::value_type;
    const auto isai = TestFixture::LowerIsai::build()
                          .with_sparsity_power(2)
                          .on(this->exec)
                          ->generate(this->l_sparse);

    auto l_inv = isai->get_approximate_inverse();

    GKO_ASSERT_MTX_EQ_SPARSITY(l_inv, this->l_sparse_inv2);
    GKO_ASSERT_MTX_NEAR(l_inv, this->l_sparse_inv2, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseUWithU2)
{
    using value_type = typename TestFixture::value_type;
    const auto isai = TestFixture::UpperIsai::build()
                          .with_sparsity_power(2)
                          .on(this->exec)
                          ->generate(this->u_sparse);

    auto u_inv = isai->get_approximate_inverse();

    GKO_ASSERT_MTX_EQ_SPARSITY(u_inv, this->u_sparse_inv2);
    GKO_ASSERT_MTX_NEAR(u_inv, this->u_sparse_inv2, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseLWithL3)
{
    using value_type = typename TestFixture::value_type;
    const auto isai = TestFixture::LowerIsai::build()
                          .with_sparsity_power(3)
                          .on(this->exec)
                          ->generate(this->l_sparse);

    auto l_inv = isai->get_approximate_inverse();

    GKO_ASSERT_MTX_EQ_SPARSITY(l_inv, this->l_sparse_inv3);
    GKO_ASSERT_MTX_NEAR(l_inv, this->l_sparse_inv3, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseUWithU3)
{
    using value_type = typename TestFixture::value_type;
    const auto isai = TestFixture::UpperIsai::build()
                          .with_sparsity_power(3)
                          .on(this->exec)
                          ->generate(this->u_sparse);

    auto u_inv = isai->get_approximate_inverse();

    GKO_ASSERT_MTX_EQ_SPARSITY(u_inv, this->u_sparse_inv2);
    GKO_ASSERT_MTX_NEAR(u_inv, this->u_sparse_inv2, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseSpd)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->spd_isai_factory->generate(this->spd_sparse);
    const auto expected_transpose =
        gko::as<Csr>(this->spd_sparse_inv->transpose());

    // In the spd case, the approximate inverse is a composition of L^T and L.
    const auto composition = isai->get_approximate_inverse()->get_operators();
    const auto lower_t = gko::as<Csr>(composition[0]);
    const auto lower = gko::as<Csr>(composition[1]);

    GKO_ASSERT_MTX_EQ_SPARSITY(lower, this->spd_sparse_inv);
    GKO_ASSERT_MTX_EQ_SPARSITY(lower_t, expected_transpose);
    GKO_ASSERT_MTX_NEAR(lower, this->spd_sparse_inv, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(lower_t, expected_transpose, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseSpdLongrow)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->spd_isai_factory->generate(this->spd_csr_longrow);
    const auto expected_transpose =
        gko::as<Csr>(this->spd_csr_longrow_inv->transpose());

    const auto composition = isai->get_approximate_inverse()->get_operators();
    const auto lower_t = gko::as<Csr>(composition[0]);
    const auto lower = gko::as<Csr>(composition[1]);

    GKO_ASSERT_MTX_EQ_SPARSITY(lower, this->spd_csr_longrow_inv);
    GKO_ASSERT_MTX_EQ_SPARSITY(lower_t, expected_transpose);
    // need to reduce precision due to spd ISAI using GMRES instead of
    // direct solve
    GKO_ASSERT_MTX_NEAR(lower, this->spd_csr_longrow_inv,
                        10 * r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(lower_t, expected_transpose, 10 * r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseSpdLongrowWithExcessSolver)
{
    using Csr = typename TestFixture::Csr;
    using SpdIsai = typename TestFixture::SpdIsai;
    using value_type = typename TestFixture::value_type;
    const auto expected_transpose =
        gko::as<Csr>(this->spd_csr_longrow_inv->transpose());
    auto spd_isai_factory =
        SpdIsai::build()
            .with_excess_solver_factory(this->excess_solver_factory)
            .on(this->exec);
    const auto isai = spd_isai_factory->generate(this->spd_csr_longrow);

    const auto composition = isai->get_approximate_inverse()->get_operators();
    const auto lower_t = gko::as<Csr>(composition[0]);
    const auto lower = gko::as<Csr>(composition[1]);

    GKO_ASSERT_MTX_EQ_SPARSITY(lower, this->spd_csr_longrow_inv);
    GKO_ASSERT_MTX_EQ_SPARSITY(lower_t, expected_transpose);
    // need to drastically reduce precision due to using different excess solver
    // factory.
    GKO_ASSERT_MTX_NEAR(lower, this->spd_csr_longrow_inv,
                        1e3 * r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(lower_t, expected_transpose,
                        1e3 * r<value_type>::value);
}


TYPED_TEST(Isai, GeneratesWithUnsortedCsr)
{
    using Csr = typename TestFixture::Csr;
    using T = typename TestFixture::value_type;

    const auto a_isai =
        this->general_isai_factory->generate(this->a_s_unsorted);
    const auto l_isai = this->lower_isai_factory->generate(this->l_s_unsorted);
    const auto u_isai = this->upper_isai_factory->generate(this->u_s_unsorted);
    const auto spd_isai =
        this->spd_isai_factory->generate(this->spd_s_unsorted);
    auto a_inv = a_isai->get_approximate_inverse();
    auto l_inv = l_isai->get_approximate_inverse();
    auto u_inv = u_isai->get_approximate_inverse();
    auto spd_l_t_inv =
        gko::as<Csr>(spd_isai->get_approximate_inverse()->get_operators()[0]);
    auto spd_l_inv =
        gko::as<Csr>(spd_isai->get_approximate_inverse()->get_operators()[1]);

    GKO_ASSERT_MTX_NEAR(a_inv, this->a_sparse_inv, r<T>::value);
    GKO_ASSERT_MTX_NEAR(l_inv, this->l_sparse_inv, r<T>::value);
    GKO_ASSERT_MTX_NEAR(u_inv, this->u_sparse_inv, r<T>::value);
    GKO_ASSERT_MTX_NEAR(spd_l_inv, this->spd_sparse_inv, r<T>::value);
    GKO_ASSERT_MTX_NEAR(spd_l_t_inv,
                        gko::as<Csr>(this->spd_sparse_inv->transpose()),
                        r<T>::value);
}


TYPED_TEST(Isai, ApplyWithAMtx)
{
    using Dense = typename TestFixture::Dense;
    using T = typename TestFixture::value_type;
    const auto vec = gko::initialize<Dense>({18., 16., 12.}, this->exec);
    auto result = Dense::create_with_config_of(vec);
    const auto a_isai = this->general_isai_factory->generate(this->a_dense);

    a_isai->apply(vec, result);

    GKO_ASSERT_MTX_NEAR(result, l({-0.625, 3.5, 7.875}), r<T>::value);
}


TYPED_TEST(Isai, ApplyWithLMtx)
{
    using Dense = typename TestFixture::Dense;
    using T = typename TestFixture::value_type;
    const auto vec = gko::initialize<Dense>({18., 16., 12.}, this->exec);
    auto result = Dense::create_with_config_of(vec);
    const auto l_isai = this->lower_isai_factory->generate(this->l_dense);

    l_isai->apply(vec, result);

    GKO_ASSERT_MTX_NEAR(result, l({9., -3.5, -24.5}), r<T>::value);
}


TYPED_TEST(Isai, ApplyWithUMtx)
{
    using Dense = typename TestFixture::Dense;
    using T = typename TestFixture::value_type;
    const auto vec = gko::initialize<Dense>({18., 16., 12.}, this->exec);
    auto result = Dense::create_with_config_of(vec);
    const auto u_isai = this->upper_isai_factory->generate(this->u_dense);

    u_isai->apply(vec, result);

    GKO_ASSERT_MTX_NEAR(result, l({6.125, -5., 1.5}), r<T>::value);
}


TYPED_TEST(Isai, ApplyWithSpdComposition)
{
    using Dense = typename TestFixture::Dense;
    using T = typename TestFixture::value_type;
    const auto vec = gko::initialize<Dense>({18., 16., 12.}, this->exec);
    auto result = Dense::create_with_config_of(vec);
    const auto spd_isai = this->spd_isai_factory->generate(this->spd_dense);

    spd_isai->apply(vec, result);

    GKO_ASSERT_MTX_NEAR(result, l({502., 110., -26.}), r<T>::value);
}


TYPED_TEST(Isai, AdvancedApplyAMtx)
{
    using Dense = typename TestFixture::Dense;
    using T = typename TestFixture::value_type;
    const auto alpha = gko::initialize<Dense>({3.}, this->exec);
    const auto beta = gko::initialize<Dense>({-4.}, this->exec);
    const auto vec = gko::initialize<Dense>({18., 16., 12}, this->exec);
    auto result = gko::initialize<Dense>({2., -3., 1.}, this->exec);
    const auto a_isai = this->general_isai_factory->generate(this->a_dense);

    a_isai->apply(alpha, vec, beta, result);

    GKO_ASSERT_MTX_NEAR(result, l({-9.875, 22.5, 19.625}), r<T>::value);
}


TYPED_TEST(Isai, AdvancedApplyLMtx)
{
    using Dense = typename TestFixture::Dense;
    using T = typename TestFixture::value_type;
    const auto alpha = gko::initialize<Dense>({3.}, this->exec);
    const auto beta = gko::initialize<Dense>({-4.}, this->exec);
    const auto vec = gko::initialize<Dense>({18., 16., 12}, this->exec);
    auto result = gko::initialize<Dense>({2., -3., 1.}, this->exec);
    const auto l_isai = this->lower_isai_factory->generate(this->l_dense);

    l_isai->apply(alpha, vec, beta, result);

    GKO_ASSERT_MTX_NEAR(result, l({19., 1.5, -77.5}), r<T>::value);
}


TYPED_TEST(Isai, AdvancedApplyUMtx)
{
    using Dense = typename TestFixture::Dense;
    using T = typename TestFixture::value_type;
    const auto alpha = gko::initialize<Dense>({3.}, this->exec);
    const auto beta = gko::initialize<Dense>({-4.}, this->exec);
    const auto vec = gko::initialize<Dense>({18., 16., 12}, this->exec);
    auto result = gko::initialize<Dense>({2., -3., 1.}, this->exec);
    const auto u_isai = this->upper_isai_factory->generate(this->u_dense);

    u_isai->apply(alpha, vec, beta, result);

    GKO_ASSERT_MTX_NEAR(result, l({10.375, -3., 0.5}), r<T>::value);
}


TYPED_TEST(Isai, AdvancedApplySpdComposition)
{
    using Dense = typename TestFixture::Dense;
    using T = typename TestFixture::value_type;
    const auto alpha = gko::initialize<Dense>({3.}, this->exec);
    const auto beta = gko::initialize<Dense>({-4.}, this->exec);
    const auto vec = gko::initialize<Dense>({18., 16., 12}, this->exec);
    auto result = gko::initialize<Dense>({2., -3., 1.}, this->exec);
    const auto spd_isai = this->spd_isai_factory->generate(this->spd_dense);

    spd_isai->apply(alpha, vec, beta, result);

    GKO_ASSERT_MTX_NEAR(result, l({1498., 342., -82.}), r<T>::value);
}


TYPED_TEST(Isai, UseWithIluPreconditioner)
{
    using Dense = typename TestFixture::Dense;
    using index_type = typename TestFixture::index_type;
    using T = typename TestFixture::value_type;
    using LowerIsai = typename TestFixture::LowerIsai;
    using UpperIsai = typename TestFixture::UpperIsai;
    const auto vec = gko::initialize<Dense>({128, -64, 32}, this->exec);
    auto result = Dense::create(this->exec, vec->get_size());
    auto mtx = gko::share(Dense::create_with_config_of(this->l_dense));
    this->l_dense->apply(this->u_dense, mtx);
    auto ilu_factory = gko::preconditioner::Ilu<LowerIsai, UpperIsai, false,
                                                index_type>::build()
                           .on(this->exec);
    auto ilu = ilu_factory->generate(mtx);

    ilu->apply(vec, result);

    GKO_ASSERT_MTX_NEAR(result, l({25., -40., -4.}), r<T>::value);
}


TYPED_TEST(Isai, ReturnsTransposedCorrectInverseA)
{
    using GeneralIsai = typename TestFixture::GeneralIsai;
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->general_isai_factory->generate(this->a_sparse);

    auto a_inv = gko::as<Csr>(gko::as<GeneralIsai>(isai->transpose())
                                  ->get_approximate_inverse()
                                  ->transpose());

    GKO_ASSERT_MTX_EQ_SPARSITY(a_inv, this->a_sparse_inv);
    GKO_ASSERT_MTX_NEAR(a_inv, this->a_sparse_inv, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsTransposedCorrectInverseL)
{
    using UpperIsai = typename TestFixture::UpperIsai;
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->lower_isai_factory->generate(this->l_sparse);

    auto l_inv = gko::as<Csr>(gko::as<UpperIsai>(isai->transpose())
                                  ->get_approximate_inverse()
                                  ->transpose());

    GKO_ASSERT_MTX_EQ_SPARSITY(l_inv, this->l_sparse_inv);
    GKO_ASSERT_MTX_NEAR(l_inv, this->l_sparse_inv, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsTransposedCorrectInverseU)
{
    using LowerIsai = typename TestFixture::LowerIsai;
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->upper_isai_factory->generate(this->u_sparse);

    auto u_inv = gko::as<Csr>(gko::as<LowerIsai>(isai->transpose())
                                  ->get_approximate_inverse()
                                  ->transpose());

    GKO_ASSERT_MTX_EQ_SPARSITY(u_inv, this->u_sparse_inv);
    GKO_ASSERT_MTX_NEAR(u_inv, this->u_sparse_inv, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsTransposedCorrectInverseSpd)
{
    using SpdIsai = typename TestFixture::SpdIsai;
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->spd_isai_factory->generate(this->spd_sparse);

    auto transpose =
        gko::as<SpdIsai>(isai->transpose())->get_approximate_inverse();
    auto lower_t = gko::as<Csr>(transpose->get_operators()[0]);
    auto lower = gko::as<Csr>(transpose->get_operators()[1]);

    GKO_ASSERT_MTX_EQ_SPARSITY(lower, this->spd_sparse_inv);
    GKO_ASSERT_MTX_EQ_SPARSITY(lower_t,
                               gko::as<Csr>(this->spd_sparse_inv->transpose()));
    GKO_ASSERT_MTX_NEAR(lower, this->spd_sparse_inv, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(lower_t,
                        gko::as<Csr>(this->spd_sparse_inv->transpose()),
                        r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsConjTransposedCorrectInverseA)
{
    using GeneralIsai = typename TestFixture::GeneralIsai;
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->general_isai_factory->generate(this->a_sparse);

    auto a_inv = gko::as<Csr>(gko::as<GeneralIsai>(isai->conj_transpose())
                                  ->get_approximate_inverse()
                                  ->conj_transpose());

    GKO_ASSERT_MTX_EQ_SPARSITY(a_inv, this->a_sparse_inv);
    GKO_ASSERT_MTX_NEAR(a_inv, this->a_sparse_inv, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsConjTransposedCorrectInverseL)
{
    using UpperIsai = typename TestFixture::UpperIsai;
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->lower_isai_factory->generate(this->l_sparse);

    auto l_inv = gko::as<Csr>(gko::as<UpperIsai>(isai->conj_transpose())
                                  ->get_approximate_inverse()
                                  ->conj_transpose());

    GKO_ASSERT_MTX_EQ_SPARSITY(l_inv, this->l_sparse_inv);
    GKO_ASSERT_MTX_NEAR(l_inv, this->l_sparse_inv, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsConjTransposedCorrectInverseU)
{
    using LowerIsai = typename TestFixture::LowerIsai;
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->upper_isai_factory->generate(this->u_sparse);

    auto u_inv = gko::as<Csr>(gko::as<LowerIsai>(isai->conj_transpose())
                                  ->get_approximate_inverse()
                                  ->conj_transpose());

    GKO_ASSERT_MTX_EQ_SPARSITY(u_inv, this->u_sparse_inv);
    GKO_ASSERT_MTX_NEAR(u_inv, this->u_sparse_inv, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsConjTransposedCorrectInverseSpd)
{
    using SpdIsai = typename TestFixture::SpdIsai;
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->spd_isai_factory->generate(this->spd_sparse);

    auto transpose =
        gko::as<SpdIsai>(isai->conj_transpose())->get_approximate_inverse();
    auto lower_t = gko::as<Csr>(transpose->get_operators()[0]);
    auto lower = gko::as<Csr>(transpose->get_operators()[1]);

    GKO_ASSERT_MTX_EQ_SPARSITY(lower, this->spd_sparse_inv);
    GKO_ASSERT_MTX_EQ_SPARSITY(
        lower_t, gko::as<Csr>(this->spd_sparse_inv->conj_transpose()));
    GKO_ASSERT_MTX_NEAR(lower, this->spd_sparse_inv, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(lower_t,
                        gko::as<Csr>(this->spd_sparse_inv->conj_transpose()),
                        r<value_type>::value);
}


TYPED_TEST(Isai, IsExactInverseOnFullSparsitySet)
{
    using Isai = typename TestFixture::GeneralIsai;
    using Csr = typename TestFixture::Csr;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    auto mtx = gko::share(gko::test::generate_tridiag_matrix<Csr>(
        12, gko::to_std_array<value_type>(-1, 2, -1), this->exec));
    auto inv_mtx = gko::test::generate_tridiag_inverse_matrix<Dense>(
        12, gko::to_std_array<value_type>(-1, 2, -1), this->exec);

    auto isai = Isai::build()
                    .with_sparsity_power(static_cast<int>(mtx->get_size()[0]))
                    .on(this->exec)
                    ->generate(mtx);

    GKO_ASSERT_MTX_NEAR(inv_mtx, isai->get_approximate_inverse(),
                        r<value_type>::value);
}


TYPED_TEST(Isai, IsExactInverseOnFullSparsitySetLarge)
{
    using Isai = typename TestFixture::GeneralIsai;
    using Csr = typename TestFixture::Csr;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    auto mtx = gko::share(gko::test::generate_tridiag_matrix<Csr>(
        33, gko::to_std_array<value_type>(-1, 2, -1), this->exec));
    auto inv_mtx = gko::test::generate_tridiag_inverse_matrix<Dense>(
        33, gko::to_std_array<value_type>(-1, 2, -1), this->exec);

    auto isai = Isai::build()
                    .with_sparsity_power(static_cast<int>(mtx->get_size()[0]))
                    .with_excess_solver_reduction(r<value_type>::value)
                    .on(this->exec)
                    ->generate(mtx);

    GKO_ASSERT_MTX_NEAR(inv_mtx, isai->get_approximate_inverse(),
                        5 * r<value_type>::value);
}


}  // namespace
