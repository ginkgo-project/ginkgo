/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <algorithm>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/constraints/constraints_handler_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


namespace {
template <typename ValueIndexType>
class ConsKernels : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using dense = gko::matrix::Dense<value_type>;

    ConsKernels() : ref(gko::ReferenceExecutor::create()) {}


    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(ConsKernels, gko::test::ValueIndexTypes);

TYPED_TEST(ConsKernels, FillSubsetEmpty)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    std::vector<value_type> ones(10, gko::one<value_type>());
    gko::Array<index_type> empty_subset{this->ref};

    gko::kernels::reference::cons::fill_subset(
        this->ref, empty_subset, ones.data(), gko::zero<value_type>());

    for (auto v : ones) {
        ASSERT_EQ(v, gko::one<value_type>());
    }
}

TYPED_TEST(ConsKernels, FillSubsetFull)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    std::vector<value_type> ones(5, gko::one<value_type>());
    gko::Array<index_type> empty_subset{this->ref, {0, 1, 2, 3, 4}};

    gko::kernels::reference::cons::fill_subset(
        this->ref, empty_subset, ones.data(), gko::zero<value_type>());

    for (auto v : ones) {
        ASSERT_EQ(v, gko::zero<value_type>());
    }
}

TYPED_TEST(ConsKernels, FillSubset)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto one = gko::one<value_type>();
    auto zero = gko::zero<value_type>();
    gko::Array<index_type> empty_subset{this->ref, {0, 4, 5, 7, 9}};
    auto ones = gko::initialize<gko::matrix::Dense<value_type>>(
        one, gko::dim<2>{10, 1}, this->ref);
    auto result = gko::initialize<gko::matrix::Dense<value_type>>(
        {zero, one, one, one, zero, zero, one, zero, one, zero}, this->ref);

    gko::kernels::reference::cons::fill_subset(this->ref, empty_subset,
                                               ones->get_values(), zero);

    GKO_ASSERT_MTX_NEAR(ones, result, 0);
}

TYPED_TEST(ConsKernels, CopySubsetEmpty)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto one = gko::one<value_type>();
    auto zero = gko::zero<value_type>();
    gko::Array<index_type> empty_subset{this->ref};
    auto src = gko::initialize<gko::matrix::Dense<value_type>>(
        {zero, one, one, one, zero, zero, one, zero, one, zero}, this->ref);
    auto dst = gko::initialize<gko::matrix::Dense<value_type>>(
        one, gko::dim<2>{10, 1}, this->ref);
    auto result = gko::clone(dst);

    gko::kernels::reference::cons::copy_subset(
        this->ref, empty_subset, src->get_const_values(), dst->get_values());

    GKO_ASSERT_MTX_NEAR(dst, result, 0);
}

TYPED_TEST(ConsKernels, CopySubsetFull)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto one = gko::one<value_type>();
    auto zero = gko::zero<value_type>();
    gko::Array<index_type> empty_subset{this->ref, {0, 1, 2, 3, 4}};
    auto src = gko::initialize<gko::matrix::Dense<value_type>>(
        {zero, one, one, one, zero}, this->ref);
    auto dst = gko::initialize<gko::matrix::Dense<value_type>>(
        one, gko::dim<2>{5, 1}, this->ref);
    auto result = gko::clone(src);

    gko::kernels::reference::cons::copy_subset(
        this->ref, empty_subset, src->get_const_values(), dst->get_values());

    GKO_ASSERT_MTX_NEAR(dst, result, 0);
}

TYPED_TEST(ConsKernels, CopySubset)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto one = gko::one<value_type>();
    auto zero = gko::zero<value_type>();
    gko::Array<index_type> empty_subset{this->ref, {0, 4, 5, 7, 9}};
    auto src = gko::initialize<gko::matrix::Dense<value_type>>(
        zero, gko::dim<2>{10, 1}, this->ref);
    auto dst = gko::initialize<gko::matrix::Dense<value_type>>(
        one, gko::dim<2>{10, 1}, this->ref);
    auto result = gko::initialize<gko::matrix::Dense<value_type>>(
        {zero, one, one, one, zero, zero, one, zero, one, zero}, this->ref);

    gko::kernels::reference::cons::copy_subset(
        this->ref, empty_subset, src->get_const_values(), dst->get_values());

    GKO_ASSERT_MTX_NEAR(dst, result, 0);
}

TYPED_TEST(ConsKernels, SetUnitRowEmptySubset)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using mtx = gko::matrix::Csr<value_type, index_type>;
    auto csr = gko::initialize<mtx>(
        {{1, 0, 2, 3}, {0, 0, 4, 0}, {5, 6, 0, 0}, {7, 0, 0, 8}}, this->ref);
    auto result = gko::clone(csr);
    gko::Array<index_type> empty_subset{this->ref};

    gko::kernels::reference::cons::set_unit_rows(
        this->ref, empty_subset, csr->get_const_row_ptrs(),
        csr->get_const_col_idxs(), csr->get_values());

    GKO_ASSERT_MTX_NEAR(csr, result, 0);
}

TYPED_TEST(ConsKernels, SetUnitRowFullSubset)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using mtx = gko::matrix::Csr<value_type, index_type>;
    auto csr = gko::initialize<mtx>(
        {{1, 0, 2, 3}, {0, 4, 0, 0}, {0, 5, 6, 0}, {7, 0, 0, 8}}, this->ref);
    auto result = gko::initialize<mtx>(
        {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}, this->ref);
    ;
    gko::Array<index_type> full_subset{this->ref, {0, 1, 2, 3}};

    gko::kernels::reference::cons::set_unit_rows(
        this->ref, full_subset, csr->get_const_row_ptrs(),
        csr->get_const_col_idxs(), csr->get_values());

    GKO_ASSERT_MTX_NEAR(csr, result, 0);
}

TYPED_TEST(ConsKernels, SetUnitRowSubset)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using mtx = gko::matrix::Csr<value_type, index_type>;
    auto csr = gko::initialize<mtx>(
        {{1, 0, 2, 3}, {0, 4, 0, 0}, {0, 5, 6, 0}, {7, 0, 0, 8}}, this->ref);
    auto result = gko::initialize<mtx>(
        {{1, 0, 0, 0}, {0, 4, 0, 0}, {0, 0, 1, 0}, {7, 0, 0, 8}}, this->ref);
    gko::Array<index_type> subset{this->ref, {0, 2}};

    gko::kernels::reference::cons::set_unit_rows(
        this->ref, subset, csr->get_const_row_ptrs(), csr->get_const_col_idxs(),
        csr->get_values());

    GKO_ASSERT_MTX_NEAR(csr, result, 0);
}

}  // namespace
