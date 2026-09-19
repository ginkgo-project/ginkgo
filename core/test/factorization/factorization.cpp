// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/factorization.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Factorization : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using fact_type =
        gko::experimental::factorization::Factorization<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using diag_type = gko::matrix::Diagonal<value_type>;
    using storage_type = gko::experimental::factorization::storage_type;

protected:
    Factorization() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(Factorization, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(Factorization, ZeroDiagonal)
{
    using Csr = typename TestFixture::Csr;
    using fact_type = typename TestFixture::fact_type;
    auto mtx = Csr::create(this->ref, gko::dim<2>{2, 2}, 2);
    auto row_ptrs = mtx->get_row_ptrs();
    auto col_idxs = mtx->get_col_idxs();
    auto values = mtx->get_values();
    row_ptrs[0] = 0;
    row_ptrs[1] = 1;
    row_ptrs[2] = 2;
    col_idxs[0] = 0;
    col_idxs[1] = 1;
    values[0] = 1.0;
    values[1] = 0.0;

    ASSERT_THROW(
        fact_type::create_from_combined_lu(std::move(mtx))->validate_data(),
        gko::InvalidData);
}


}  // namespace
