// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class SparsityCsr : public ::testing::Test {
protected:
    using v_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using i_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::SparsityCsr<v_type, i_type>;
    using Csr = gko::matrix::Csr<v_type, i_type>;
    using DenseMtx = gko::matrix::Dense<v_type>;

    SparsityCsr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, gko::dim<2>{2, 3}, 4))
    {
        i_type* c = mtx->get_col_idxs();
        i_type* r = mtx->get_row_ptrs();
        r[0] = 0;
        r[1] = 3;
        r[2] = 4;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 1;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;
};

TYPED_TEST_SUITE(SparsityCsr, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(SparsityCsr, CanBeCreatedFromExistingCsrMatrix)
{
    using Csr = typename TestFixture::Csr;
    using DenseMtx = typename TestFixture::DenseMtx;
    using Mtx = typename TestFixture::Mtx;
    auto csr_mtx = gko::initialize<Csr>(
        {{2.0, 3.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, -3.0}}, this->exec);
    auto comp_mtx = gko::initialize<DenseMtx>(
        {{1.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, 1.0}}, this->exec);

    auto mtx = Mtx::create(this->exec, std::move(csr_mtx));

    GKO_ASSERT_MTX_NEAR(comp_mtx, mtx, 0.0);
}


TYPED_TEST(SparsityCsr, CanBeCreatedFromExistingDenseMatrix)
{
    using DenseMtx = typename TestFixture::DenseMtx;
    using Mtx = typename TestFixture::Mtx;
    auto dense_mtx = gko::initialize<DenseMtx>(
        {{2.0, 3.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, -3.0}}, this->exec);
    auto comp_mtx = gko::initialize<DenseMtx>(
        {{1.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, 1.0}}, this->exec);

    auto mtx = Mtx::create(this->exec, std::move(dense_mtx));

    GKO_ASSERT_MTX_NEAR(comp_mtx, mtx, 0.0);
}


}  // namespace
