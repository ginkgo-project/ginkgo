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
    using Isai_type = gko::preconditioner::Isai<value_type, index_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    Isai()
        : exec{gko::ReferenceExecutor::create()},
          l_dense{gko::initialize<Dense>(
              {{2., 0., 0.}, {1., -2., 0.}, {-1., 1., -1.}}, exec)},
          l_dense_inv{gko::initialize<Dense>(
              {{.5, 0., 0.}, {.25, -.5, 0.}, {-.25, -.5, -1.}}, exec)},
          l_csr{Csr::create(exec)},
          l_csr_inv{Csr::create(exec)},
          l_sparse{Csr::create(exec, gko::dim<2>(4, 4),
                               I<value_type>{-1., 2., 4., 5., -4., 8., -8.},
                               I<index_type>{0, 0, 1, 1, 2, 2, 3},
                               I<index_type>{0, 1, 3, 5, 7})},
          l_sparse_inv{Csr::create(
              exec, gko::dim<2>(4, 4),
              I<value_type>{-1., .5, .25, .3125, -.25, -.25, -.125},
              I<index_type>{0, 0, 1, 1, 2, 2, 3}, I<index_type>{0, 1, 3, 5, 7})}
    {
        isai_factory = Isai_type::build().on(exec);
        l_dense->convert_to(gko::lend(l_csr));
        l_dense_inv->convert_to(gko::lend(l_csr_inv));
    }

    std::unique_ptr<Csr> clone_allocations(const Csr *csr_mtx)
    {
        auto size = csr_mtx->get_size();
        const auto num_elems = csr_mtx->get_num_stored_elements();
        auto sparsity = Csr::create(exec, size, num_elems);

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
    std::unique_ptr<typename Isai_type::Factory> isai_factory;
    std::shared_ptr<gko::matrix::Csr<value_type, index_type>> mtx;
    std::unique_ptr<Dense> l_dense;
    std::unique_ptr<Dense> l_dense_inv;
    std::unique_ptr<Csr> l_csr;
    std::unique_ptr<Csr> l_csr_inv;
    std::unique_ptr<Csr> l_sparse;
    std::unique_ptr<Csr> l_sparse_inv;
};

TYPED_TEST_CASE(Isai, gko::test::ValueIndexTypes);


TYPED_TEST(Isai, KernelGenerateL)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    auto trans_expected = this->transpose(gko::lend(this->l_csr_inv));
    auto l_trans = this->transpose(gko::lend(this->l_csr));
    auto result = this->clone_allocations(gko::lend(l_trans));

    gko::kernels::reference::isai::generate_l(this->exec, gko::lend(l_trans),
                                              gko::lend(result));

    GKO_ASSERT_MTX_EQ_SPARSITY(result, trans_expected);
    GKO_ASSERT_MTX_NEAR(result, trans_expected, r<value_type>::value);
}


TYPED_TEST(Isai, KernelGenerateLsparse)
{
    using Csr = typename TestFixture::Csr;
    using value_type = typename TestFixture::value_type;
    auto trans_expected = this->transpose(gko::lend(this->l_sparse_inv));
    auto l_trans = this->transpose(gko::lend(this->l_sparse));
    auto result = this->clone_allocations(gko::lend(l_trans));

    gko::kernels::reference::isai::generate_l(this->exec, gko::lend(l_trans),
                                              gko::lend(result));

    GKO_ASSERT_MTX_EQ_SPARSITY(result, trans_expected);
    GKO_ASSERT_MTX_NEAR(result, trans_expected, r<value_type>::value);
}


}  // namespace
