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

#include "core/distributed/partition_helpers_kernels.hpp"


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename IndexType>
class PartitionHelpers : public CommonTestFixture {
protected:
    using index_type = IndexType;
};

TYPED_TEST_SUITE(PartitionHelpers, gko::test::IndexTypes);


TYPED_TEST(PartitionHelpers, CanCompressStartEndsWithOneRange)
{
    using itype = typename TestFixture::index_type;
    gko::array<itype> start_ends{this->exec, {0, 3}};
    gko::array<itype> expects{this->exec, {0, 3}};
    gko::array<itype> result{this->exec, expects.get_num_elems()};

    gko::kernels::EXEC_NAMESPACE::partition_helpers::compress_start_ends(
        this->exec, start_ends, result);

    GKO_ASSERT_ARRAY_EQ(result, expects);
}


TYPED_TEST(PartitionHelpers, CanCompressStartEndsWithMultipleRanges)
{
    using itype = typename TestFixture::index_type;
    gko::array<itype> start_ends{this->exec, {0, 3, 3, 7, 7, 10}};
    gko::array<itype> expects{this->exec, {0, 3, 7, 10}};
    gko::array<itype> result{this->exec, expects.get_num_elems()};

    gko::kernels::EXEC_NAMESPACE::partition_helpers::compress_start_ends(
        this->exec, start_ends, result);

    GKO_ASSERT_ARRAY_EQ(result, expects);
}


TYPED_TEST(PartitionHelpers, CanCompressStartEndsWithZeroRange)
{
    using itype = typename TestFixture::index_type;
    gko::array<itype> start_ends{this->exec};
    gko::array<itype> expects{this->exec, {0}};
    gko::array<itype> result{this->exec, {0}};

    gko::kernels::EXEC_NAMESPACE::partition_helpers::compress_start_ends(
        this->exec, start_ends, result);

    GKO_ASSERT_ARRAY_EQ(result, expects);
}
