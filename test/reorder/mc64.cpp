// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/reorder/mc64.hpp>


#include "core/test/utils/assertions.hpp"
#include "test/utils/executor.hpp"


namespace {


class Mc64 : public CommonTestFixture {
protected:
    using CsrMtx = gko::matrix::Csr<value_type, index_type>;
    using reorder_type =
        gko::experimental::reorder::Mc64<value_type, index_type>;
    using result_type = gko::Composition<value_type>;
    using perm_type = gko::matrix::ScaledPermutation<value_type, index_type>;

    Mc64()
        : mtx(gko::initialize<CsrMtx>({{1.0, 2.0, 0.0, -1.3, 2.1},
                                       {2.0, 5.0, 1.5, 0.0, 0.0},
                                       {0.0, 1.5, 1.5, 1.1, 0.0},
                                       {-1.3, 0.0, 1.1, 2.0, 0.0},
                                       {2.1, 0.0, 0.0, 0.0, 1.0}},
                                      ref)),
          dmtx(mtx->clone(exec)),
          mc64_factory(reorder_type::build().on(ref)),
          dmc64_factory(reorder_type::build().on(exec))
    {}

    std::pair<std::shared_ptr<const perm_type>,
              std::shared_ptr<const perm_type>>
    unpack(const result_type* result)
    {
        GKO_ASSERT_EQ(result->get_operators().size(), 2);
        return std::make_pair(gko::as<perm_type>(result->get_operators()[0]),
                              gko::as<perm_type>(result->get_operators()[1]));
    }

    std::unique_ptr<reorder_type> mc64_factory;
    std::unique_ptr<reorder_type> dmc64_factory;
    std::shared_ptr<CsrMtx> mtx;
    std::shared_ptr<CsrMtx> dmtx;
};


TEST_F(Mc64, IsEquivalentToReference)
{
    auto perm = mc64_factory->generate(mtx);
    auto dperm = dmc64_factory->generate(dmtx);

    auto ops = unpack(perm.get());
    auto dops = unpack(dperm.get());
    GKO_ASSERT_MTX_NEAR(ops.first, dops.first, 0.0);
    GKO_ASSERT_MTX_NEAR(ops.second, dops.second, 0.0);
}


}  // namespace
