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

#include "reference/preconditioner/isai_kernels.cpp"


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
#include <ginkgo/core/preconditioner/isai.hpp>


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
    using LowerIsai = gko::preconditioner::LowerIsai<value_type, index_type>;
    using UpperIsai = gko::preconditioner::UpperIsai<value_type, index_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    Isai()
        : exec{gko::ReferenceExecutor::create()},
          l_dense{gko::initialize<Dense>(
              {{2., 0., 0.}, {1., -2., 0.}, {-1., 1., -1.}}, exec)},
          l_dense_inv{gko::initialize<Dense>(
              {{.5, 0., 0.}, {.25, -.5, 0.}, {-.25, -.5, -1.}}, exec)},
          u_dense{gko::initialize<Dense>(
              {{4., 1., -1.}, {0., -2., 4.}, {0., 0., 8.}}, exec)},
          u_dense_inv{gko::initialize<Dense>(
              {{.25, .125, -0.03125}, {0., -.5, .25}, {0., 0., .125}}, exec)},
          l_csr{Csr::create(exec)},
          l_csr_inv{Csr::create(exec)},
          u_csr{Csr::create(exec)},
          u_csr_inv{Csr::create(exec)},
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
                                    I<index_type>{0, 4, 7, 9, 10})}
    {
        lower_isai_factory = LowerIsai::build().on(exec);
        upper_isai_factory = UpperIsai::build().on(exec);
        l_dense->convert_to(lend(l_csr));
        l_dense_inv->convert_to(lend(l_csr_inv));
        u_dense->convert_to(lend(u_csr));
        u_dense_inv->convert_to(lend(u_csr_inv));
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
    }

    template <typename ReadMtx>
    std::unique_ptr<ReadMtx> read(const char *name)
    {
        std::ifstream mtxstream{std::string{gko::matrices::location_isai_mtxs} +
                                name};
        auto result = gko::read<ReadMtx>(mtxstream, exec);
        // to avoid removing 0s, the matrices store 12345 instead
        for (gko::size_type i = 0; i < result->get_num_stored_elements(); ++i) {
            auto &val = result->get_values()[i];
            if (val == static_cast<value_type>(12345.0)) {
                val = 0;
            }
        }
        return std::move(result);
    }

    std::unique_ptr<Csr> clone_allocations(const Csr *csr_mtx)
    {
        const auto num_elems = csr_mtx->get_num_stored_elements();
        auto sparsity = csr_mtx->clone();

        // values are now filled with invalid data to catch potential errors
        std::fill_n(sparsity->get_values(), num_elems, -gko::one<value_type>());
        return sparsity;
    }

    std::unique_ptr<Csr> transpose(const Csr *mtx)
    {
        return gko::as<Csr>(mtx->transpose());
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<typename LowerIsai::Factory> lower_isai_factory;
    std::unique_ptr<typename UpperIsai::Factory> upper_isai_factory;
    std::shared_ptr<Dense> l_dense;
    std::shared_ptr<Dense> l_dense_inv;
    std::shared_ptr<Dense> u_dense;
    std::shared_ptr<Dense> u_dense_inv;
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
};

TYPED_TEST_CASE(Isai, gko::test::ValueIndexTypes);


TYPED_TEST(Isai, KernelGenerateL1)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto result = this->clone_allocations(lend(this->l_csr));
    auto num_rows = result->get_size()[0];
    gko::Array<index_type> a1(this->exec, num_rows + 1);
    gko::Array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::Array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, lend(this->l_csr), lend(result), a1.get_data(),
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
    const auto l_mtx = this->transpose(lend(this->u_csr));
    auto result = this->clone_allocations(lend(l_mtx));
    auto num_rows = result->get_size()[0];
    gko::Array<index_type> a1(this->exec, num_rows + 1);
    gko::Array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::Array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, lend(l_mtx), lend(result), a1.get_data(), a2.get_data(),
        true);

    const auto expected = this->transpose(lend(this->u_csr_inv));
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
    auto result = this->clone_allocations(lend(this->l_sparse));
    auto num_rows = result->get_size()[0];
    gko::Array<index_type> a1(this->exec, num_rows + 1);
    gko::Array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::Array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, lend(this->l_sparse), lend(result), a1.get_data(),
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
    auto result = this->clone_allocations(lend(this->l_sparse2));
    auto num_rows = result->get_size()[0];
    gko::Array<index_type> a1(this->exec, num_rows + 1);
    gko::Array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::Array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, lend(this->l_sparse2), lend(result), a1.get_data(),
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
    const auto l_mtx = this->transpose(lend(this->u_sparse));
    auto result = this->clone_allocations(lend(l_mtx));
    auto num_rows = result->get_size()[0];
    gko::Array<index_type> a1(this->exec, num_rows + 1);
    gko::Array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::Array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, lend(l_mtx), lend(result), a1.get_data(), a2.get_data(),
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
    auto result = this->clone_allocations(lend(this->l_csr_longrow));
    auto num_rows = result->get_size()[0];
    gko::Array<index_type> a1(this->exec, num_rows + 1);
    gko::Array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::Array<index_type> zeros(this->exec, num_rows + 1);
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
        this->exec, lend(this->l_csr_longrow), lend(result), a1.get_data(),
        a2.get_data(), true);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->l_csr_longrow_inv_partial);
    GKO_ASSERT_MTX_NEAR(result, this->l_csr_longrow_inv_partial,
                        r<value_type>::value);
    // no row above the size limit -> zero array
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
    gko::Array<index_type> zeros(this->exec, num_rows + 1);
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
        this->exec, lend(this->l_csr_longrow), lend(this->l_csr_longrow),
        a1.get_const_data(), a2.get_const_data(), lend(result),
        lend(result_rhs));

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->l_csr_longrow_e);
    GKO_ASSERT_MTX_NEAR(result, this->l_csr_longrow_e, 0);
    GKO_ASSERT_MTX_NEAR(result_rhs, this->l_csr_longrow_e_rhs, 0);
}


TYPED_TEST(Isai, KernelGenerateU1)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const auto u_mtx = this->transpose(lend(this->l_csr));
    auto result = this->clone_allocations(lend(u_mtx));
    auto num_rows = result->get_size()[0];
    gko::Array<index_type> a1(this->exec, num_rows + 1);
    gko::Array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::Array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, lend(u_mtx), lend(result), a1.get_data(), a2.get_data(),
        false);

    auto expected = this->transpose(lend(this->l_csr_inv));
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
    auto result = this->clone_allocations(lend(this->u_csr));
    auto num_rows = result->get_size()[0];
    gko::Array<index_type> a1(this->exec, num_rows + 1);
    gko::Array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::Array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, lend(this->u_csr), lend(result), a1.get_data(),
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
    const auto u_mtx = this->transpose(lend(this->l_sparse));
    auto result = this->clone_allocations(lend(u_mtx));
    auto num_rows = result->get_size()[0];
    gko::Array<index_type> a1(this->exec, num_rows + 1);
    gko::Array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::Array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, lend(u_mtx), lend(result), a1.get_data(), a2.get_data(),
        false);

    const auto expected = this->transpose(lend(this->l_sparse_inv));
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
    const auto u_mtx = this->transpose(this->l_sparse2.get());
    auto result = this->clone_allocations(lend(u_mtx));
    auto num_rows = result->get_size()[0];
    gko::Array<index_type> a1(this->exec, num_rows + 1);
    gko::Array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::Array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, lend(u_mtx), lend(result), a1.get_data(), a2.get_data(),
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
    auto result = this->clone_allocations(lend(this->u_sparse));
    auto num_rows = result->get_size()[0];
    gko::Array<index_type> a1(this->exec, num_rows + 1);
    gko::Array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::Array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, lend(this->u_sparse), lend(result), a1.get_data(),
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
    auto result = this->clone_allocations(lend(this->u_csr_longrow));
    auto num_rows = result->get_size()[0];
    gko::Array<index_type> a1(this->exec, num_rows + 1);
    gko::Array<index_type> a2(this->exec, num_rows + 1);
    // zero-filled array
    gko::Array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);
    // only the 32nd row has some excess storage
    auto a1_expect = zeros;
    std::fill_n(a1_expect.get_data() + 3, 33, 33);
    auto a2_expect = zeros;
    std::fill_n(a2_expect.get_data() + 3, 33, 153);

    gko::kernels::reference::isai::generate_tri_inverse(
        this->exec, lend(this->u_csr_longrow), lend(result), a1.get_data(),
        a2.get_data(), false);

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->u_csr_longrow_inv_partial);
    GKO_ASSERT_MTX_NEAR(result, this->u_csr_longrow_inv_partial,
                        r<value_type>::value);
    // no row above the size limit -> zero array
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
    gko::Array<index_type> zeros(this->exec, num_rows + 1);
    std::fill_n(zeros.get_data(), num_rows + 1, 0);
    // only the 32nd row has some excess storage
    auto a1 = zeros;
    std::fill_n(a1.get_data() + 3, 33, 33);
    auto a2 = zeros;
    std::fill_n(a2.get_data() + 3, 33, 153);
    auto result = Csr::create(this->exec, gko::dim<2>(33, 33), 153);
    auto result_rhs = Dense::create(this->exec, gko::dim<2>(33, 1));

    gko::kernels::reference::isai::generate_excess_system(
        this->exec, lend(this->u_csr_longrow), lend(this->u_csr_longrow),
        a1.get_const_data(), a2.get_const_data(), lend(result),
        lend(result_rhs));

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->u_csr_longrow_e);
    GKO_ASSERT_MTX_NEAR(result, this->u_csr_longrow_e, 0);
    GKO_ASSERT_MTX_NEAR(result_rhs, this->u_csr_longrow_e_rhs, 0);
}


TYPED_TEST(Isai, KernelScatterExcessSolution)
{
    using Csr = typename TestFixture::Csr;
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    gko::Array<index_type> ptrs{this->exec, I<index_type>{0, 0, 2, 2, 5, 7, 7}};
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
        this->exec, ptrs.get_const_data(), sol.get(), mtx.get());

    GKO_ASSERT_MTX_NEAR(mtx, expect, 0);
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


TYPED_TEST(Isai, GeneratesWithUnsortedCsr)
{
    using Csr = typename TestFixture::Csr;
    using T = typename TestFixture::value_type;

    const auto l_isai = this->lower_isai_factory->generate(this->l_s_unsorted);
    const auto u_isai = this->upper_isai_factory->generate(this->u_s_unsorted);
    auto l_inv = l_isai->get_approximate_inverse();
    auto u_inv = u_isai->get_approximate_inverse();

    GKO_ASSERT_MTX_NEAR(l_inv, this->l_sparse_inv, r<T>::value);
    GKO_ASSERT_MTX_NEAR(u_inv, this->u_sparse_inv, r<T>::value);
}


TYPED_TEST(Isai, ApplyWithLMtx)
{
    using Dense = typename TestFixture::Dense;
    using T = typename TestFixture::value_type;
    const auto vec = gko::initialize<Dense>({18., 16., 12.}, this->exec);
    auto result = Dense::create_with_config_of(lend(vec));
    const auto l_isai = this->lower_isai_factory->generate(this->l_dense);

    l_isai->apply(lend(vec), lend(result));

    GKO_ASSERT_MTX_NEAR(result, l({9., -3.5, -24.5}), r<T>::value);
}


TYPED_TEST(Isai, ApplyWithUMtx)
{
    using Dense = typename TestFixture::Dense;
    using T = typename TestFixture::value_type;
    const auto vec = gko::initialize<Dense>({18., 16., 12.}, this->exec);
    auto result = Dense::create_with_config_of(lend(vec));
    const auto u_isai = this->upper_isai_factory->generate(this->u_dense);

    u_isai->apply(lend(vec), lend(result));

    GKO_ASSERT_MTX_NEAR(result, l({6.125, -5., 1.5}), r<T>::value);
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

    l_isai->apply(lend(alpha), lend(vec), lend(beta), lend(result));

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

    u_isai->apply(lend(alpha), lend(vec), lend(beta), lend(result));

    GKO_ASSERT_MTX_NEAR(result, l({10.375, -3., 0.5}), r<T>::value);
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
    auto mtx = gko::share(Dense::create_with_config_of(lend(this->l_dense)));
    this->l_dense->apply(lend(this->u_dense), lend(mtx));
    auto ilu_factory = gko::preconditioner::Ilu<LowerIsai, UpperIsai, false,
                                                index_type>::build()
                           .on(this->exec);
    auto ilu = ilu_factory->generate(mtx);

    ilu->apply(lend(vec), lend(result));

    GKO_ASSERT_MTX_NEAR(result, l({25., -40., -4.}), r<T>::value);
}


}  // namespace
