// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/reorder/rcm.hpp"


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
