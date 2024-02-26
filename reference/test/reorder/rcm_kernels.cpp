// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/reorder/rcm.hpp>


#include <algorithm>
#include <fstream>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


#include "core/test/utils/assertions.hpp"


namespace {


class Rcm : public ::testing::Test {
protected:
    using v_type = double;
    using i_type = int;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    using reorder_type = gko::reorder::Rcm<v_type, i_type>;
    using new_reorder_type = gko::experimental::reorder::Rcm<i_type>;
    using perm_type = gko::matrix::Permutation<i_type>;

    Rcm()
        : exec(gko::ReferenceExecutor::create()),
          // clang-format off
          p_mtx_0(gko::initialize<CsrMtx>(
                                        {{1.0, 2.0, 0.0, -1.3, 2.1},
                                         {2.0, 5.0, 1.5, 0.0, 0.0},
                                         {0.0, 1.5, 1.5, 1.1, 0.0},
                                         {-1.3, 0.0, 1.1, 2.0, 0.0},
                                         {2.1, 0.0, 0.0, 0.0, 1.0}},
                                        exec)),
        p_mtx_1(gko::initialize<CsrMtx>(
                                        {{1., 0., 0., 1., 1., 1., 0., 1., 1.},
                                         {0., 1., 1., 1., 0., 1., 1., 0., 0.},
                                         {0., 1., 1., 0., 0., 1., 1., 0., 0.},
                                         {1., 1., 0., 1., 1., 1., 1., 0., 1.},
                                         {1., 0., 0., 1., 1., 1., 1., 1., 1.},
                                         {1., 1., 1., 1., 1., 1., 1., 0., 0.},
                                         {0., 1., 1., 1., 1., 1., 1., 0., 0.},
                                         {1., 0., 0., 0., 1., 0., 0., 1., 1.},
                                         {1., 0., 0., 1., 1., 0., 0., 1., 1.}},
                                        exec)),
        p_mtx_1_lower(gko::initialize<CsrMtx>(
                                        {{1., 0., 0., 0., 0., 0., 0., 0., 0.},
                                         {0., 1., 0., 0., 0., 0., 0., 0., 0.},
                                         {0., 1., 1., 0., 0., 0., 0., 0., 0.},
                                         {1., 1., 0., 1., 0., 0., 0., 0., 0.},
                                         {1., 0., 0., 1., 1., 0., 0., 0., 0.},
                                         {1., 1., 1., 1., 1., 1., 0., 0., 0.},
                                         {0., 1., 1., 1., 1., 1., 1., 0., 0.},
                                         {1., 0., 0., 0., 1., 0., 0., 1., 0.},
                                         {1., 0., 0., 1., 1., 0., 0., 1., 1.}},
                                        exec)),
          // clang-format on
          rcm_factory(reorder_type::build().on(exec)),
          reorder_op_0(rcm_factory->generate(p_mtx_0)),
          reorder_op_1(rcm_factory->generate(p_mtx_1))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<reorder_type::Factory> rcm_factory;
    std::shared_ptr<CsrMtx> p_mtx_0;
    std::unique_ptr<reorder_type> reorder_op_0;
    std::shared_ptr<CsrMtx> p_mtx_1;
    std::unique_ptr<reorder_type> reorder_op_1;
    std::shared_ptr<CsrMtx> p_mtx_1_lower;

    static bool is_permutation(const perm_type* input_perm)
    {
        const auto perm_size = input_perm->get_size()[0];
        auto perm_sorted = std::vector<i_type>(perm_size);
        std::copy_n(input_perm->get_const_permutation(), perm_size,
                    perm_sorted.begin());
        std::sort(perm_sorted.begin(), perm_sorted.end());
        auto identity = std::vector<i_type>(perm_size);
        std::iota(identity.begin(), identity.end(), 0);
        return identity == perm_sorted;
    }
};


TEST_F(Rcm, CreatesAPermutation)
{
    auto p = gko::as<perm_type>(reorder_op_0->get_permutation());

    ASSERT_PRED1(is_permutation, p.get());
}


TEST_F(Rcm, CreatesCorrectPermutation)
{
    std::vector<i_type> correct = {2, 3, 1, 0, 4};

    auto p = gko::as<perm_type>(reorder_op_0->get_permutation())
                 ->get_const_permutation();

    ASSERT_TRUE(std::equal(p, p + correct.size(), correct.begin()));
}


TEST_F(Rcm, PermutesPerfectFullBand)
{
    std::vector<i_type> correct = {7, 8, 0, 4, 3, 5, 6, 1, 2};

    auto p = gko::as<perm_type>(reorder_op_1->get_permutation())
                 ->get_const_permutation();

    ASSERT_TRUE(std::equal(p, p + correct.size(), correct.begin()));
}


TEST_F(Rcm, NewInterfaceWorksOnSymmetric)
{
    std::vector<i_type> correct = {7, 8, 0, 4, 3, 5, 6, 1, 2};

    auto permutation =
        new_reorder_type::build().with_skip_symmetrize(true).on(exec)->generate(
            p_mtx_1);

    auto p = permutation->get_const_permutation();
    ASSERT_TRUE(std::equal(p, p + correct.size(), correct.begin()));
}


TEST_F(Rcm, NewInterfaceWorksOnNonsymmetric)
{
    std::vector<i_type> correct = {7, 8, 0, 4, 3, 5, 6, 1, 2};

    auto permutation =
        new_reorder_type::build().on(exec)->generate(p_mtx_1_lower);

    auto p = permutation->get_const_permutation();
    ASSERT_TRUE(std::equal(p, p + correct.size(), correct.begin()));
}


}  // namespace
