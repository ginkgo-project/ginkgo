// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <fstream>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/reorder/nested_dissection.hpp>


#include "core/test/utils.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


template <typename IndexType>
class NestedDissection : public CommonTestFixture {
protected:
    using index_type = IndexType;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;
    using reorder_type =
        gko::experimental::reorder::NestedDissection<value_type, index_type>;
    using perm_type = gko::matrix::Permutation<index_type>;


    NestedDissection()
        : nd_factory(reorder_type::build().on(ref)),
          dnd_factory(reorder_type::build().on(exec))
    {
        std::ifstream stream{gko::matrices::location_ani1_mtx};
        mtx = gko::read<matrix_type>(stream, ref);
        dmtx = gko::clone(exec, mtx);
    }

    std::unique_ptr<reorder_type> nd_factory;
    std::unique_ptr<reorder_type> dnd_factory;
    std::shared_ptr<matrix_type> mtx;
    std::shared_ptr<matrix_type> dmtx;
};

TYPED_TEST_SUITE(NestedDissection, gko::test::IndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(NestedDissection, ResultIsEquivalentToRef)
{
    auto perm = this->nd_factory->generate(this->mtx);
    auto dperm = this->dnd_factory->generate(this->dmtx);

    auto perm_array = gko::make_array_view(this->ref, this->mtx->get_size()[0],
                                           perm->get_permutation());
    auto dperm_array = gko::make_array_view(
        this->exec, this->mtx->get_size()[0], dperm->get_permutation());
    GKO_ASSERT_ARRAY_EQ(perm_array, dperm_array);
}
