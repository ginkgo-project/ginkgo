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

#include <ginkgo/core/reorder/rcm.hpp>


#include <gtest/gtest.h>


#include "core/test/utils/assertions.hpp"
#include "cuda/test/utils.hpp"


namespace {


class Rcm : public CudaTestFixture {
protected:
    using v_type = double;
    using i_type = int;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    using reorder_type = gko::reorder::Rcm<v_type, i_type>;
    using new_reorder_type = gko::experimental::reorder::Rcm<i_type>;
    using perm_type = gko::matrix::Permutation<i_type>;


    Rcm()
        : p_mtx(gko::initialize<CsrMtx>({{1.0, 2.0, 0.0, -1.3, 2.1},
                                         {2.0, 5.0, 1.5, 0.0, 0.0},
                                         {0.0, 1.5, 1.5, 1.1, 0.0},
                                         {-1.3, 0.0, 1.1, 2.0, 0.0},
                                         {2.1, 0.0, 0.0, 0.0, 1.0}},
                                        exec))
    {}

    std::shared_ptr<CsrMtx> p_mtx;
};


TEST_F(Rcm, IsEquivalentToRef)
{
    auto reorder_op = reorder_type::build().on(ref)->generate(p_mtx);
    auto dreorder_op = reorder_type::build().on(exec)->generate(p_mtx);

    GKO_ASSERT_ARRAY_EQ(dreorder_op->get_permutation_array(),
                        reorder_op->get_permutation_array());
}


TEST_F(Rcm, IsEquivalentToRefNewInterface)
{
    auto reorder_op = new_reorder_type::build().on(ref)->generate(p_mtx);
    auto dreorder_op = new_reorder_type::build().on(exec)->generate(p_mtx);

    GKO_ASSERT_MTX_EQ_SPARSITY(dreorder_op, reorder_op);
}


}  // namespace
