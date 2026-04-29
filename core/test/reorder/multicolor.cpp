// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/reorder/multicolor.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/test/utils/reordering.hpp"


template <typename IndexType>
class Multicolor : public ::testing::Test {
protected:
    using v_type = float;
    using i_type = IndexType;
    using reorder_type = gko::reorder::Multicolor<v_type, i_type>;

    Multicolor()
        : exec(gko::ReferenceExecutor::create()),
          mc_factory(reorder_type::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename reorder_type::Factory> mc_factory;
};

TYPED_TEST_SUITE(Multicolor, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(Multicolor, MulticolorFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->mc_factory->get_executor(), this->exec);
}

TYPED_TEST(Multicolor, GeneratesCorrectOrderingWithCsrInput)
{
    using v_type = typename TestFixture::v_type;
    using i_type = typename TestFixture::i_type;
    const gko::dim<2> grid{5, 5};
    auto expected =
        gko::test::compute_multicolor_ordering_regular_star<i_type>(grid);
    const auto size = 25u;
    auto mdata =
        gko::test::generate_laplacian_2d_5point_matrix_data<v_type, i_type>(
            grid);
    auto mat = gko::share(gko::matrix::Csr<v_type, i_type>::create(this->exec));
    mat->read(mdata);

    auto mc = this->mc_factory->generate(mat);

    auto color_ptrs = mc->get_color_pointers();
    auto perm = mc->get_permutation()->get_const_permutation();
    auto iperm = mc->get_inverse_permutation()->get_const_permutation();
    auto permv = std::vector<i_type>(perm, perm + size);
    auto ipermv = std::vector<i_type>(iperm, iperm + size);
    EXPECT_EQ(color_ptrs, expected.color_ptrs);
    EXPECT_EQ(permv, expected.old_to_new);
    EXPECT_EQ(ipermv, expected.new_to_old);
}

TYPED_TEST(Multicolor, GeneratesCorrectOrderingWithSparsityCsrInput)
{
    using v_type = typename TestFixture::v_type;
    using i_type = typename TestFixture::i_type;
    const gko::dim<2> grid{5, 5};
    auto expected =
        gko::test::compute_multicolor_ordering_regular_star<i_type>(grid);
    const auto size = 25u;
    auto mdata =
        gko::test::generate_laplacian_2d_5point_matrix_data<v_type, i_type>(
            grid);
    auto mat = gko::matrix::Csr<v_type, i_type>::create(this->exec);
    mat->read(mdata);
    auto smat = gko::share(
        gko::matrix::SparsityCsr<v_type, i_type>::create(this->exec));
    mat->convert_to(smat.get());

    auto mc = this->mc_factory->generate(smat);

    auto color_ptrs = mc->get_color_pointers();
    auto perm = mc->get_permutation()->get_const_permutation();
    auto iperm = mc->get_inverse_permutation()->get_const_permutation();
    const auto permv = std::vector<i_type>(perm, perm + size);
    const auto ipermv = std::vector<i_type>(iperm, iperm + size);
    EXPECT_EQ(color_ptrs, expected.color_ptrs);
    EXPECT_EQ(permv, expected.old_to_new);
    EXPECT_EQ(ipermv, expected.new_to_old);
}
