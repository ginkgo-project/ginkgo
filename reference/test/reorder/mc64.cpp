// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/reorder/mc64.hpp>


#include <algorithm>
#include <fstream>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/scaled_permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"


namespace {


template <typename ValueIndexType>
class Mc64 : public ::testing::Test {
protected:
    using v_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using i_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using real_type = gko::remove_complex<v_type>;
    using reorder_type = gko::experimental::reorder::Mc64<v_type, i_type>;
    using perm_type = gko::matrix::ScaledPermutation<v_type, i_type>;
    using result_type = gko::Composition<v_type>;
    using Mtx = gko::matrix::Dense<v_type>;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    Mc64()
        : exec(gko::ReferenceExecutor::create()),
          mc64_factory(reorder_type::build().on(exec)),
          id3_mtx(gko::initialize<CsrMtx>(
              {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}, exec)),
          not_id3_mtx(gko::initialize<CsrMtx>(
              {{1.0, 0.0, 2.0}, {0.0, 1.0, 0.0}, {2.0, 0.0, 1.0}}, exec))
    {}

    std::pair<std::shared_ptr<const perm_type>,
              std::shared_ptr<const perm_type>>
    unpack(const result_type* result)
    {
        GKO_ASSERT_EQ(result->get_operators().size(), 2);
        return std::make_pair(gko::as<perm_type>(result->get_operators()[0]),
                              gko::as<perm_type>(result->get_operators()[1]));
    }

    void assert_correct_permutation(const result_type* mc64)
    {
        auto perm = unpack(mc64).first->get_const_permutation();

        ASSERT_EQ(perm[0], 0);
        ASSERT_EQ(perm[1], 1);
        ASSERT_EQ(perm[2], 2);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<CsrMtx> id3_mtx;
    std::shared_ptr<CsrMtx> not_id3_mtx;
    std::unique_ptr<reorder_type> mc64_factory;
};

TYPED_TEST_SUITE(Mc64, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Mc64, HasSensibleDefaults)
{
    using real_type = typename TestFixture::real_type;

    ASSERT_EQ(this->mc64_factory->get_parameters().strategy,
              gko::experimental::reorder::mc64_strategy::max_diagonal_product);
    ASSERT_EQ(this->mc64_factory->get_parameters().tolerance, real_type{1e-14});
}


TYPED_TEST(Mc64, CanBeCreatedWithReorderingStrategy)
{
    using reorder_type = typename TestFixture::reorder_type;

    auto mc64 =
        reorder_type::build()
            .with_strategy(
                gko::experimental::reorder::mc64_strategy::max_diagonal_sum)
            .on(this->exec)
            ->generate(this->not_id3_mtx);

    auto perm = this->unpack(mc64.get()).first->get_const_permutation();
    ASSERT_EQ(perm[0], 2);
    ASSERT_EQ(perm[1], 1);
    ASSERT_EQ(perm[2], 0);
}


TYPED_TEST(Mc64, CanBeCreatedEmpty)
{
    using reorder_type = typename TestFixture::reorder_type;
    using matrix_type = typename TestFixture::CsrMtx;

    auto mc64 =
        reorder_type::build()
            .with_strategy(
                gko::experimental::reorder::mc64_strategy::max_diagonal_sum)
            .on(this->exec)
            ->generate(matrix_type::create(this->exec));

    ASSERT_FALSE(mc64->get_size());
}


TYPED_TEST(Mc64, CanBeCreatedWithTolerance)
{
    using reorder_type = typename TestFixture::reorder_type;
    using real_type = typename TestFixture::real_type;

    auto mc64 = reorder_type::build()
                    .with_tolerance(real_type{1e-10})
                    .on(this->exec)
                    ->generate(this->id3_mtx);

    this->assert_correct_permutation(mc64.get());
}


}  // namespace
