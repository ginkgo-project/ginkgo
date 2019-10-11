/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/reorder/metis_fill_reduce.hpp>


#include <algorithm>
#include <fstream>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/metis_types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "matrices/config.hpp"


namespace {


class MetisFillReduce : public ::testing::Test {
protected:
    using v_type = double;
    using i_type = metis_indextype;
    using Mtx = gko::matrix::Dense<v_type>;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    using reorder_type = gko::reorder::MetisFillReduce<v_type, i_type>;
    MetisFillReduce()
        : ref(gko::ReferenceExecutor::create()),
          omp(gko::OmpExecutor::create()),
          ani4_mtx(gko::read<CsrMtx>(
              std::ifstream(gko::matrices::location_ani4_mtx, std::ios::in),
              ref)),
          d_ani4_mtx(gko::read<CsrMtx>(
              std::ifstream(gko::matrices::location_ani4_mtx, std::ios::in),
              omp))
    {}

    static void assert_equal_permutations(
        const gko::matrix::Permutation<i_type> *to_check,
        const gko::matrix::Permutation<i_type> *orig)
    {
        auto o_p_size = orig->get_permutation_size();
        auto d_p_size = to_check->get_permutation_size();
        ASSERT_EQ(d_p_size, o_p_size);

        for (auto i = 0; i < d_p_size; ++i) {
            ASSERT_EQ(to_check->get_const_permutation()[i],
                      orig->get_const_permutation()[i]);
        }
    }

    std::shared_ptr<const gko::Executor> ref;
    std::shared_ptr<const gko::Executor> omp;
    std::shared_ptr<CsrMtx> ani4_mtx;
    std::shared_ptr<CsrMtx> d_ani4_mtx;
    std::unique_ptr<reorder_type> reorder_op;
    std::unique_ptr<reorder_type> d_reorder_op;
};

TEST_F(MetisFillReduce, OmpPermutationIsEquivalentToRef)
{
    auto reorder_op = reorder_type::build().on(ref)->generate(ani4_mtx);
    auto d_reorder_op = reorder_type::build().on(omp)->generate(ani4_mtx);

    auto perm = reorder_op->get_permutation();
    auto d_perm = d_reorder_op->get_permutation();

    assert_equal_permutations(d_perm.get(), perm.get());
}

}  // namespace
