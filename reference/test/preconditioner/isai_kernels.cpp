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


#include <algorithm>
#include <memory>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"
#include "reference/preconditioner/isai_kernels.cpp"


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
              I<index_type>{0, 4, 6, 8, 9})}
    {
        lower_isai_factory = LowerIsai::build().on(exec);
        upper_isai_factory = UpperIsai::build().on(exec);
        l_dense->convert_to(gko::lend(l_csr));
        l_dense_inv->convert_to(gko::lend(l_csr_inv));
        u_dense->convert_to(gko::lend(u_csr));
        u_dense_inv->convert_to(gko::lend(u_csr_inv));
    }

    std::unique_ptr<Csr> clone_allocations(const Csr *csr_mtx)
    {
        auto size = csr_mtx->get_size();
        const auto num_elems = csr_mtx->get_num_stored_elements();
        auto sparsity = Csr::create(exec, size, num_elems);

        // All arrays are now filled with invalid data to catch potential errors
        std::fill_n(sparsity->get_values(), num_elems, -gko::one<value_type>());
        std::fill_n(sparsity->get_col_idxs(), num_elems,
                    -gko::one<index_type>());
        std::fill_n(sparsity->get_row_ptrs(), size[0] + 1,
                    -gko::one<index_type>());
        return sparsity;
    }

    template <typename To, typename From>
    static std::unique_ptr<To> unique_static_cast(std::unique_ptr<From> from)
    {
        return std::unique_ptr<To>{static_cast<To *>(from.release())};
    }

    std::unique_ptr<Csr> transpose(const Csr *mtx)
    {
        return unique_static_cast<Csr>(mtx->transpose());
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
    std::shared_ptr<Csr> u_csr;
    std::shared_ptr<Csr> u_csr_inv;
    std::shared_ptr<Csr> l_sparse;
    std::shared_ptr<Csr> l_s_unsorted;
    std::shared_ptr<Csr> l_sparse_inv;
    std::shared_ptr<Csr> l_sparse2;
    std::shared_ptr<Csr> l_sparse2_inv;
    std::shared_ptr<Csr> u_sparse;
    std::shared_ptr<Csr> u_s_unsorted;
    std::shared_ptr<Csr> u_sparse_inv;
};

TYPED_TEST_CASE(Isai, gko::test::ValueIndexTypes);


TYPED_TEST(Isai, KernelGenerateL1)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    auto result = this->clone_allocations(gko::lend(this->l_csr));

    gko::kernels::reference::isai::generate_l_inverse(
        this->exec, gko::lend(this->l_csr), gko::lend(result));

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->l_csr_inv);
    GKO_ASSERT_MTX_NEAR(result, this->l_csr_inv, r<value_type>::value);
}


TYPED_TEST(Isai, KernelGenerateL2)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto l_mtx = this->transpose(gko::lend(this->u_csr));
    auto result = this->clone_allocations(gko::lend(l_mtx));

    gko::kernels::reference::isai::generate_l_inverse(
        this->exec, gko::lend(l_mtx), gko::lend(result));

    const auto expected = this->transpose(gko::lend(this->u_csr_inv));
    GKO_ASSERT_MTX_EQ_SPARSITY(result, expected);
    GKO_ASSERT_MTX_NEAR(result, expected, r<value_type>::value);
}


TYPED_TEST(Isai, KernelGenerateLsparse1)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    auto result = this->clone_allocations(gko::lend(this->l_sparse));

    gko::kernels::reference::isai::generate_l_inverse(
        this->exec, gko::lend(this->l_sparse), gko::lend(result));

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->l_sparse_inv);
    GKO_ASSERT_MTX_NEAR(result, this->l_sparse_inv, r<value_type>::value);
}


TYPED_TEST(Isai, KernelGenerateLsparse2)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto result = this->clone_allocations(gko::lend(this->l_sparse2));

    gko::kernels::reference::isai::generate_l_inverse(
        this->exec, gko::lend(this->l_sparse2), gko::lend(result));

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->l_sparse2_inv);
    GKO_ASSERT_MTX_NEAR(result, this->l_sparse2_inv, r<value_type>::value);
}


TYPED_TEST(Isai, KernelGenerateLsparse3)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const auto l_mtx = this->transpose(gko::lend(this->u_sparse));
    auto result = this->clone_allocations(gko::lend(l_mtx));

    gko::kernels::reference::isai::generate_l_inverse(
        this->exec, gko::lend(l_mtx), gko::lend(result));

    // Results in a slightly different version than u_sparse_inv->transpose()
    // because a different row-sparsity pattern is used in u_sparse vs. l_mtx
    // (only one value changes compared to u_sparse_inv->transpose())
    const auto expected = Csr::create(
        this->exec, gko::dim<2>(4, 4),
        I<value_type>{-.5, .125, .25, .3125, .125, -.5, .125, .25, .5},
        I<index_type>{0, 0, 1, 0, 1, 2, 0, 2, 3}, I<index_type>{0, 1, 3, 6, 9});
    GKO_ASSERT_MTX_EQ_SPARSITY(result, expected);
    GKO_ASSERT_MTX_NEAR(result, expected, r<value_type>::value);
}


TYPED_TEST(Isai, KernelGenerateU1)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto u_mtx = this->transpose(gko::lend(this->l_csr));
    auto result = this->clone_allocations(gko::lend(u_mtx));

    gko::kernels::reference::isai::generate_u_inverse(
        this->exec, gko::lend(u_mtx), gko::lend(result));

    auto expected = this->transpose(gko::lend(this->l_csr_inv));
    GKO_ASSERT_MTX_EQ_SPARSITY(result, expected);
    GKO_ASSERT_MTX_NEAR(result, expected, r<value_type>::value);
}


TYPED_TEST(Isai, KernelGenerateU2)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    auto result = this->clone_allocations(gko::lend(this->u_csr));

    gko::kernels::reference::isai::generate_u_inverse(
        this->exec, gko::lend(this->u_csr), gko::lend(result));

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->u_csr_inv);
    GKO_ASSERT_MTX_NEAR(result, this->u_csr_inv, r<value_type>::value);
}


TYPED_TEST(Isai, KernelGenerateUsparse1)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto u_mtx = this->transpose(gko::lend(this->l_sparse));
    auto result = this->clone_allocations(gko::lend(u_mtx));

    gko::kernels::reference::isai::generate_u_inverse(
        this->exec, gko::lend(u_mtx), gko::lend(result));

    const auto expected = this->transpose(gko::lend(this->l_sparse_inv));
    GKO_ASSERT_MTX_EQ_SPARSITY(result, expected);
    GKO_ASSERT_MTX_NEAR(result, expected, r<value_type>::value);
}


TYPED_TEST(Isai, KernelGenerateUsparse2)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const auto u_mtx = this->transpose(this->l_sparse2.get());
    auto result = this->clone_allocations(gko::lend(u_mtx));

    gko::kernels::reference::isai::generate_u_inverse(
        this->exec, gko::lend(u_mtx), gko::lend(result));

    // Results in a slightly different version than l_sparse2_inv->transpose()
    // because a different row-sparsity pattern is used in l_sparse2 vs. u_mtx
    // (only one value changes compared to l_sparse2_inv->transpose())
    const auto expected = Csr::create(
        this->exec, gko::dim<2>(4, 4),
        I<value_type>{-.5, .125, .3125, .25, .125, .0625, -.5, .25, .5},
        I<index_type>{0, 1, 3, 1, 2, 3, 2, 3, 3}, I<index_type>{0, 3, 6, 8, 9});
    GKO_ASSERT_MTX_EQ_SPARSITY(result, expected);
    GKO_ASSERT_MTX_NEAR(result, expected, r<value_type>::value);
}


TYPED_TEST(Isai, KernelGenerateUsparse3)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    auto result = this->clone_allocations(gko::lend(this->u_sparse));

    gko::kernels::reference::isai::generate_u_inverse(
        this->exec, gko::lend(this->u_sparse), gko::lend(result));

    GKO_ASSERT_MTX_EQ_SPARSITY(result, this->u_sparse_inv);
    GKO_ASSERT_MTX_NEAR(result, this->u_sparse_inv, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseL)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->lower_isai_factory->generate(this->l_sparse);

    auto l_inv = isai->get_system_matrix();

    GKO_ASSERT_MTX_NEAR(l_inv, this->l_sparse_inv, r<value_type>::value);
}


TYPED_TEST(Isai, ReturnsCorrectInverseU)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    const auto isai = this->upper_isai_factory->generate(this->u_sparse);

    auto u_inv = isai->get_system_matrix();

    GKO_ASSERT_MTX_NEAR(u_inv, this->u_sparse_inv, r<value_type>::value);
}


TYPED_TEST(Isai, GeneratesWithUnsortedCsr)
{
    using Csr = typename TestFixture::Csr;
    using T = typename TestFixture::value_type;

    const auto l_isai = this->lower_isai_factory->generate(this->l_s_unsorted);
    const auto u_isai = this->upper_isai_factory->generate(this->u_s_unsorted);
    auto l_inv = l_isai->get_system_matrix();
    auto u_inv = u_isai->get_system_matrix();

    GKO_ASSERT_MTX_NEAR(l_inv, this->l_sparse_inv, r<T>::value);
    GKO_ASSERT_MTX_NEAR(u_inv, this->u_sparse_inv, r<T>::value);
}


TYPED_TEST(Isai, ApplyWithLMtx)
{
    using Dense = typename TestFixture::Dense;
    using T = typename TestFixture::value_type;
    const auto vec = gko::initialize<Dense>({18., 16., 12.}, this->exec);
    auto result = Dense::create_with_config_of(gko::lend(vec));
    const auto l_isai = this->lower_isai_factory->generate(this->l_dense);

    l_isai->apply(gko::lend(vec), gko::lend(result));

    GKO_ASSERT_MTX_NEAR(result, l({9., -3.5, -24.5}), r<T>::value);
}


TYPED_TEST(Isai, ApplyWithUMtx)
{
    using Dense = typename TestFixture::Dense;
    using T = typename TestFixture::value_type;
    const auto vec = gko::initialize<Dense>({18., 16., 12.}, this->exec);
    auto result = Dense::create_with_config_of(gko::lend(vec));
    const auto u_isai = this->upper_isai_factory->generate(this->u_dense);

    u_isai->apply(gko::lend(vec), gko::lend(result));

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

    l_isai->apply(gko::lend(alpha), gko::lend(vec), gko::lend(beta),
                  gko::lend(result));

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

    u_isai->apply(gko::lend(alpha), gko::lend(vec), gko::lend(beta),
                  gko::lend(result));

    GKO_ASSERT_MTX_NEAR(result, l({10.375, -3., 0.5}), r<T>::value);
}


}  // namespace
