/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include "core/components/format_conversion_kernels.hpp"


#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


template <typename IndexType>
class FormatConversion : public ::testing::Test {
protected:
    FormatConversion() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(FormatConversion, gko::test::IndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(FormatConversion, ConvertsEmptyPtrsToIdxs)
{
    std::vector<TypeParam> ptrs(10);
    TypeParam* out = nullptr;

    gko::kernels::reference::components::convert_ptrs_to_idxs(
        this->ref, ptrs.data(), 9, out);

    // mustn't segfault
}


TYPED_TEST(FormatConversion, ConvertsPtrsToIdxs)
{
    std::vector<TypeParam> ptrs{0, 1, 3, 5, 5};
    std::vector<TypeParam> idxs(5);
    std::vector<TypeParam> reference{0, 1, 1, 2, 2};

    gko::kernels::reference::components::convert_ptrs_to_idxs(
        this->ref, ptrs.data(), 4, idxs.data());

    ASSERT_EQ(idxs, reference);
}


TYPED_TEST(FormatConversion, ConvertsEmptyIdxsToPtrs)
{
    std::vector<TypeParam> idxs;
    std::vector<TypeParam> ptrs(10);
    std::vector<TypeParam> reference(10);

    gko::kernels::reference::components::convert_idxs_to_ptrs(
        this->ref, idxs.data(), 0, 9, ptrs.data());

    ASSERT_EQ(ptrs, reference);
}


TYPED_TEST(FormatConversion, ConvertsIdxsToPtrs)
{
    std::vector<TypeParam> idxs{1, 1, 1, 2, 2, 4};
    std::vector<TypeParam> ptrs(6);
    std::vector<TypeParam> reference{0, 0, 3, 5, 5, 6};

    gko::kernels::reference::components::convert_idxs_to_ptrs(
        this->ref, idxs.data(), 6, 5, ptrs.data());

    ASSERT_EQ(ptrs, reference);
}


TYPED_TEST(FormatConversion, ConvertsPtrsToSizes)
{
    std::vector<TypeParam> ptrs{0, 1, 3, 5, 5};
    std::vector<gko::size_type> sizes(4);
    std::vector<gko::size_type> reference{1, 2, 2, 0};

    gko::kernels::reference::components::convert_ptrs_to_sizes(
        this->ref, ptrs.data(), 4, sizes.data());

    ASSERT_EQ(sizes, reference);
}


}  // namespace
