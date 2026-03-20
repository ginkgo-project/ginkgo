// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/*@GKO_PREPROCESSOR_FILENAME_HELPER@*/

#include <type_traits>

#include <gtest/gtest.h>

#include "common/unified/base/kernel_launch.hpp"
#include "core/base/array_access.hpp"
#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


template <typename ValueType>
class DenseView : public CommonTestFixture {
public:
    using value_type = ValueType;
    using view_type = gko::matrix::view::dense<value_type>;
};

TYPED_TEST_SUITE(DenseView, gko::test::ValueTypes, TypenameNameGenerator);


template <typename ValueType>
void assert_dense_view(std::shared_ptr<const gko::EXEC_TYPE> exec)
{
    gko::array<bool> correct{exec, {false}};
    gko::array<ValueType> values{exec, 6};
    values.fill(gko::one<ValueType>());
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto values, auto correct) {
            using device_type = std::decay_t<decltype(values[0])>;
            gko::matrix::view::dense<device_type> view{gko::dim<2>{2, 2}, 3,
                                                       values};
            if (view.size == gko::dim<2>(2, 2) && view.stride == 3 &&
                view.values == values && &view(0, 0) == &values[0] &&
                &view(1, 0) == &values[3] && &view(1, 1) == &values[4] &&
                view(1, 1) == gko::one(view(1, 1))) {
                *correct = true;
            }
        },
        1, values, correct);
    ASSERT_TRUE(get_element(correct, 0));
}

TYPED_TEST(DenseView, WorksOnDevice)
{
    assert_dense_view<TypeParam>(this->exec);
}
