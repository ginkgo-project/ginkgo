// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
    using perm_type = gko::matrix::Permutation<i_type>;


    Rcm()
        :  // clang-format off
          p_mtx(gko::initialize<CsrMtx>({{1.0, 2.0, 0.0, -1.3, 2.1},
                                         {2.0, 5.0, 1.5, 0.0, 0.0},
                                         {0.0, 1.5, 1.5, 1.1, 0.0},
                                         {-1.3, 0.0, 1.1, 2.0, 0.0},
                                         {2.1, 0.0, 0.0, 0.0, 1.0}},
                                        exec)),
          // clang-format on
          rcm_factory(reorder_type::build().on(exec)),
          reorder_op(rcm_factory->generate(p_mtx))
    {}

    std::unique_ptr<reorder_type::Factory> rcm_factory;
    std::shared_ptr<CsrMtx> p_mtx;
    std::unique_ptr<reorder_type> reorder_op;
};


TEST_F(Rcm, IsExecutedOnCpuExecutor)
{
    // This only executes successfully if computed on cpu executor.
    auto p = reorder_op->get_permutation();

    ASSERT_TRUE(true);
}


}  // namespace
