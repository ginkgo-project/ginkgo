/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#include <memory>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"

namespace {


template <typename IndexType>
class FormatConversion : public ::testing::Test {
protected:
    FormatConversion() : rand(293), size(42793) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
        sizes.set_executor(ref);
        ptrs.set_executor(ref);
        idxs.set_executor(ref);
        std::uniform_int_distribution<int> row_dist{0, 10};
        sizes.resize_and_reset(size);
        ptrs.resize_and_reset(size + 1);
        ptrs.get_data()[0] = 0;
        for (gko::size_type i = 0; i < size; i++) {
            sizes.get_data()[i] = row_dist(rand);
            ptrs.get_data()[i + 1] = ptrs.get_data()[i] + sizes.get_data()[i];
        }
        idxs.resize_and_reset(ptrs.get_const_data()[size]);
        for (gko::size_type i = 0; i < size; i++) {
            auto begin = ptrs.get_const_data()[i];
            auto end = ptrs.get_const_data()[i + 1];
            for (auto j = begin; j < end; j++) {
                idxs.get_data()[j] = i;
            }
        }
        sizes.set_executor(exec);
        ptrs.set_executor(exec);
        idxs.set_executor(exec);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;
    gko::size_type size;
    std::default_random_engine rand;
    gko::array<gko::size_type> sizes;
    gko::array<IndexType> ptrs;
    gko::array<IndexType> idxs;
};

TYPED_TEST_SUITE(FormatConversion, gko::test::IndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(FormatConversion, ConvertsEmptyPtrsToIdxs)
{
    gko::array<TypeParam> ptrs(this->exec, this->size + 1);
    ptrs.fill(0);
    TypeParam* output = nullptr;

    gko::kernels::EXEC_NAMESPACE::components::convert_ptrs_to_idxs(
        this->exec, ptrs.get_const_data(), this->size, output);

    // mustn't segfault
}


TYPED_TEST(FormatConversion, ConvertPtrsToIdxs)
{
    auto ref_idxs = this->idxs;
    this->idxs.fill(-1);

    gko::kernels::EXEC_NAMESPACE::components::convert_ptrs_to_idxs(
        this->exec, this->ptrs.get_const_data(), this->size,
        this->idxs.get_data());

    GKO_ASSERT_ARRAY_EQ(this->idxs, ref_idxs);
}


TYPED_TEST(FormatConversion, ConvertsEmptyIdxsToPtrs)
{
    this->ptrs.fill(0);
    auto ref_ptrs = this->ptrs;
    this->ptrs.fill(-1);
    TypeParam* input = nullptr;

    gko::kernels::EXEC_NAMESPACE::components::convert_idxs_to_ptrs(
        this->exec, input, 0, this->size, this->ptrs.get_data());

    GKO_ASSERT_ARRAY_EQ(this->ptrs, ref_ptrs);
}


TYPED_TEST(FormatConversion, ConvertIdxsToPtrsIsEquivalentToRef)
{
    auto ref_ptrs = this->ptrs;
    this->ptrs.fill(-1);

    gko::kernels::EXEC_NAMESPACE::components::convert_idxs_to_ptrs(
        this->exec, this->idxs.get_const_data(), this->idxs.get_num_elems(),
        this->size, this->ptrs.get_data());

    GKO_ASSERT_ARRAY_EQ(this->ptrs, ref_ptrs);
}


TYPED_TEST(FormatConversion, ConvertPtrsToSizesIsEquivalentToRef)
{
    auto ref_sizes = this->sizes;
    this->sizes.fill(12345);

    gko::kernels::EXEC_NAMESPACE::components::convert_ptrs_to_sizes(
        this->exec, this->ptrs.get_const_data(), this->size,
        this->sizes.get_data());

    GKO_ASSERT_ARRAY_EQ(this->sizes, ref_sizes);
}


}  // namespace
